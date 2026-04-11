"""Case configurations for the studied microgrids."""



from __future__ import annotations



from dataclasses import dataclass, field, replace

import numpy as np



from .models.battery import BatteryParams
from .models.cigre_components import DispatchableUnitParams

RESIDENTIAL_GRID_TOU_PRICE_SPREAD_MULTIPLIER = 1.75
RESIDENTIAL_GRID_MONTHLY_DEMAND_CHARGE_PER_KW = 16.0
RESIDENTIAL_GRID_MONTHLY_DEMAND_CHARGE_THRESHOLD_W = 6_000.0
RESIDENTIAL_GRID_BATTERY_REPLACEMENT_COST_PER_KWH = 420.0
RESIDENTIAL_GRID_BATTERY_EQUIVALENT_FULL_CYCLES = 2_200.0
RESIDENTIAL_GRID_BATTERY_END_OF_LIFE_FRACTION = 0.80
RESIDENTIAL_GRID_BATTERY_DEGRADATION_COST_MULTIPLIER = 1.10
RESIDENTIAL_GRID_BATTERY_THROUGHPUT_PENALTY_PER_KWH = (
    RESIDENTIAL_GRID_BATTERY_REPLACEMENT_COST_PER_KWH
    * RESIDENTIAL_GRID_BATTERY_DEGRADATION_COST_MULTIPLIER
    / (2.0 * RESIDENTIAL_GRID_BATTERY_EQUIVALENT_FULL_CYCLES * RESIDENTIAL_GRID_BATTERY_END_OF_LIFE_FRACTION)
)


@dataclass(frozen=True)
class RewardConfig:
    """Reward coefficients and clipping ranges."""



    w_cost: float = 15.0

    w_soc_violation: float = 150.0

    w_voltage_violation: float = 100.0
    w_line_overload: float = 100.0
    w_transformer_overload: float = 80.0

    w_band: float = 5.0

    w_edge: float = 10.0

    soc_center: float = 0.5

    soc_sigma: float = 0.2

    soc_band_min: float = 0.2

    soc_band_max: float = 0.8

    reward_min: float = -500.0

    reward_max: float = 200.0

    valley_price: float = 0.39073

    peak_price: float = 0.51373


def make_loss_only_battery_params(base: BatteryParams) -> BatteryParams:
    """Return a Thevenin-compatible battery param set focused on nonlinear loss effects only."""

    return BatteryParams(
        cell_capacity_ah=base.cell_capacity_ah,
        num_cells_series=base.num_cells_series,
        num_cells_parallel=base.num_cells_parallel,
        nominal_energy_kwh=base.nominal_energy_kwh,
        soc_min=base.soc_min,
        soc_max=base.soc_max,
        soc_init=base.soc_init,
        p_charge_max=base.p_charge_max,
        p_discharge_max=base.p_discharge_max,
        eta_charge=base.eta_charge,
        eta_discharge=base.eta_discharge,
        soc_breakpoints=np.asarray(base.soc_breakpoints, dtype=float).copy() if base.soc_breakpoints is not None else None,
        ocv_values=np.asarray(base.ocv_values, dtype=float).copy() if base.ocv_values is not None else None,
        ocv_charge_values=np.asarray(base.ocv_values, dtype=float).copy() if base.ocv_values is not None else None,
        ocv_discharge_values=np.asarray(base.ocv_values, dtype=float).copy() if base.ocv_values is not None else None,
        r_int_values=np.asarray(base.r_int_values, dtype=float).copy() if base.r_int_values is not None else None,
        thermal_dynamics_enabled=False,
        ambient_temperature_c=base.ambient_temperature_c,
        temperature_init_c=base.reference_temperature_c,
        reference_temperature_c=base.reference_temperature_c,
        r_int_temp_coeff_per_c=0.0,
        min_r_int_temp_factor=1.0,
        max_r_int_temp_factor=1.0,
        thermal_resistance_k_per_w=base.thermal_resistance_k_per_w,
        thermal_capacitance_j_per_k=base.thermal_capacitance_j_per_k,
        temperature_min_c=base.temperature_min_c,
        temperature_max_c=base.temperature_max_c,
        low_soc_r_int_boost_enabled=False,
        low_soc_r_int_boost_threshold=base.low_soc_r_int_boost_threshold,
        low_soc_r_int_boost_factor=1.0,
        low_soc_r_int_boost_exponent=1.0,
        power_stress_r_int_boost_enabled=False,
        power_stress_r_int_boost_start_fraction=base.power_stress_r_int_boost_start_fraction,
        power_stress_r_int_boost_factor=1.0,
        power_stress_r_int_boost_exponent=1.0,
        paper_soc_nonlinearity_enabled=True,
        paper_soc_nonlinearity_gain=max(float(getattr(base, "paper_soc_nonlinearity_gain", 0.0)), 0.8),
        paper_soc_discharge_nonlinearity_gain=max(
            float(getattr(base, "paper_soc_discharge_nonlinearity_gain", 0.0) or getattr(base, "paper_soc_nonlinearity_gain", 0.0)),
            0.8,
        ),
        paper_soc_charge_nonlinearity_gain=max(
            float(getattr(base, "paper_soc_charge_nonlinearity_gain", 0.0) or getattr(base, "paper_soc_nonlinearity_gain", 0.0)),
            0.6,
        ),
        paper_soc_nonlinearity_reference_soc=base.paper_soc_nonlinearity_reference_soc,
        paper_soc_nonlinearity_charge_shift=base.paper_soc_nonlinearity_charge_shift,
        paper_soc_nonlinearity_floor=base.paper_soc_nonlinearity_floor,
        paper_soc_nonlinearity_exponent=max(float(base.paper_soc_nonlinearity_exponent), 1.0),
        ocv_hysteresis_enabled=False,
        ocv_hysteresis_transition_tau_seconds=base.ocv_hysteresis_transition_tau_seconds,
        ocv_hysteresis_relaxation_tau_seconds=base.ocv_hysteresis_relaxation_tau_seconds,
        ocv_hysteresis_deadband_a=base.ocv_hysteresis_deadband_a,
        rc_branch_1_resistance_values=None,
        rc_branch_1_capacitance_values=None,
        rc_branch_2_resistance_values=None,
        rc_branch_2_capacitance_values=None,
    )


def make_no_dispatch_battery_params(base: BatteryParams) -> BatteryParams:
    """Return a battery parameter set with storage dispatch disabled."""

    return replace(base, p_charge_max=0.0, p_discharge_max=0.0)


def make_paper_aligned_reward_config(base: RewardConfig | None = None) -> RewardConfig:
    """Reward settings closer to the reference nonlinear-loss paper."""

    template = base or RewardConfig()
    return RewardConfig(
        w_cost=1.0,
        w_soc_violation=template.w_soc_violation,
        w_voltage_violation=140.0,
        w_line_overload=140.0,
        w_transformer_overload=80.0,
        w_band=0.4,
        w_edge=2.5,
        soc_center=template.soc_center,
        soc_sigma=max(float(template.soc_sigma), 0.18),
        soc_band_min=max(0.10, float(template.soc_band_min) - 0.10),
        soc_band_max=min(0.95, float(template.soc_band_max) + 0.05),
        reward_min=-500.0,
        reward_max=200.0,
        valley_price=template.valley_price,
        peak_price=template.peak_price,
    )


def make_paper_balanced_reward_config(base: RewardConfig | None = None) -> RewardConfig:
    """Cost-led reward with moderate shaping to reduce boundary saturation and overactive dispatch."""

    template = base or RewardConfig()
    return RewardConfig(
        w_cost=1.0,
        w_soc_violation=template.w_soc_violation,
        w_voltage_violation=140.0,
        w_line_overload=140.0,
        w_transformer_overload=80.0,
        w_band=1.5,
        w_edge=8.0,
        soc_center=template.soc_center,
        soc_sigma=max(float(template.soc_sigma), 0.16),
        soc_band_min=max(0.18, float(template.soc_band_min) - 0.05),
        soc_band_max=min(0.88, float(template.soc_band_max)),
        reward_min=-500.0,
        reward_max=200.0,
        valley_price=template.valley_price,
        peak_price=template.peak_price,
    )


def large_format_lfp_r_int_values() -> np.ndarray:
    """Return a study-oriented large-format LFP cell resistance envelope in ohms.

    The profile is centered on the <=0.25 mOhm AC-impedance class reported by
    large-format 280 Ah LiFePO4 product specifications at 25 C and moderate SOC,
    while retaining a mild U-shaped rise near SOC boundaries for ECM use.
    """

    return np.array([
        0.36,
        0.32,
        0.29,
        0.26,
        0.24,
        0.23,
        0.22,
        0.23,
        0.24,
        0.26,
        0.29,
        0.32,
        0.36,
    ]) / 1000.0


LFP_280AH_CELL_MASS_KG = 5.4
LFP_EFFECTIVE_HEAT_CAPACITY_J_PER_KG_K = 900.0


def network_scale_lfp_pack_thermal_params(
    *,
    num_cells_series: int,
    num_cells_parallel: int,
    thermal_resistance_k_per_w: float,
) -> dict[str, float]:
    """Return pack-level thermal parameters for large-format 280 Ah LFP ESS packs.

    The effective heat capacity is scaled from the cell count so community- and
    distribution-scale BESS cases no longer inherit unrealistically small
    thermal inertia from the generic BatteryParams defaults.
    """

    cell_count = float(num_cells_series * num_cells_parallel)
    thermal_capacitance_j_per_k = (
        cell_count * LFP_280AH_CELL_MASS_KG * LFP_EFFECTIVE_HEAT_CAPACITY_J_PER_KG_K
    )
    return {
        "ambient_temperature_c": 25.0,
        "temperature_init_c": 25.0,
        "reference_temperature_c": 25.0,
        "r_int_temp_coeff_per_c": 0.010,
        "min_r_int_temp_factor": 0.80,
        "max_r_int_temp_factor": 1.80,
        "thermal_resistance_k_per_w": thermal_resistance_k_per_w,
        "thermal_capacitance_j_per_k": thermal_capacitance_j_per_k,
        "temperature_min_c": 0.0,
        "temperature_max_c": 55.0,
    }





def residential_battery_params() -> BatteryParams:
    ocv_base = np.array([
        2.50,
        2.90,
        3.05,
        3.20,
        3.22,
        3.24,
        3.26,
        3.27,
        3.29,
        3.32,
        3.38,
        3.50,
        3.65,
    ])
    ocv_hysteresis_delta = np.array([
        0.000,
        0.006,
        0.009,
        0.011,
        0.012,
        0.012,
        0.011,
        0.010,
        0.009,
        0.007,
        0.005,
        0.003,
        0.000,
    ])
    return BatteryParams(
        cell_capacity_ah=64.0,
        num_cells_series=16,
        num_cells_parallel=1,
        nominal_energy_kwh=3.3,
        soc_min=0.10,
        soc_max=0.95,
        soc_init=0.50,
        p_charge_max=3_000.0,
        p_discharge_max=3_300.0,
        eta_charge=0.90,
        eta_discharge=0.90,
        r_int_values=np.array([
            0.0112,
            0.0106,
            0.0099,
            0.0092,
            0.0088,
            0.0084,
            0.0081,
            0.0082,
            0.0086,
            0.0091,
            0.0098,
            0.0105,
            0.0112,
        ]),
        ocv_values=ocv_base,
        ocv_charge_values=ocv_base + ocv_hysteresis_delta,
        ocv_discharge_values=ocv_base - ocv_hysteresis_delta,
        thermal_dynamics_enabled=True,
        ambient_temperature_c=20.0,
        temperature_init_c=20.0,
        reference_temperature_c=25.0,
        r_int_temp_coeff_per_c=0.020,
        min_r_int_temp_factor=0.85,
        max_r_int_temp_factor=2.20,
        thermal_resistance_k_per_w=0.10,
        thermal_capacitance_j_per_k=80_000.0,
        temperature_min_c=0.0,
        temperature_max_c=55.0,
        paper_soc_nonlinearity_enabled=False,
        paper_soc_nonlinearity_gain=0.0,
        paper_soc_discharge_nonlinearity_gain=0.0,
        paper_soc_charge_nonlinearity_gain=0.0,
        paper_soc_nonlinearity_reference_soc=0.55,
        paper_soc_nonlinearity_charge_shift=1.10,
        paper_soc_nonlinearity_floor=0.15,
        paper_soc_nonlinearity_exponent=1.0,
        ocv_hysteresis_enabled=True,
        ocv_hysteresis_transition_tau_seconds=5_400.0,
        ocv_hysteresis_relaxation_tau_seconds=21_600.0,
        ocv_hysteresis_deadband_a=2.0,
        rc_branch_1_resistance_values=np.array([
            0.0018,
            0.0016,
            0.0014,
            0.0012,
            0.00105,
            0.00095,
            0.00090,
            0.00095,
            0.00105,
            0.00120,
            0.00140,
            0.00160,
            0.00180,
        ]),
        rc_branch_1_capacitance_values=np.array([
            2.8e6,
            3.2e6,
            3.7e6,
            4.2e6,
            4.8e6,
            5.2e6,
            5.5e6,
            5.2e6,
            4.8e6,
            4.2e6,
            3.7e6,
            3.2e6,
            2.8e6,
        ]),
        rc_branch_2_resistance_values=np.array([
            0.00120,
            0.00110,
            0.00100,
            0.00090,
            0.00082,
            0.00075,
            0.00070,
            0.00075,
            0.00082,
            0.00090,
            0.00100,
            0.00110,
            0.00120,
        ]),
        rc_branch_2_capacitance_values=np.array([
            1.50e7,
            1.65e7,
            1.80e7,
            2.00e7,
            2.20e7,
            2.35e7,
            2.45e7,
            2.35e7,
            2.20e7,
            2.00e7,
            1.80e7,
            1.65e7,
            1.50e7,
        ]),
    )




def residential_generator_params() -> DispatchableUnitParams:
    return DispatchableUnitParams(
        name="gen",
        p_min_w=0.0,
        p_max_w=1_000.0,
        ramp_up_w=1_000.0,
        ramp_down_w=1_000.0,
        min_up_steps=1,
        min_down_steps=1,
        start_threshold_w=100.0,
        cost_a0=0.0157,
        cost_a1=0.1080,
        cost_a2=0.3100,
        startup_cost=0.0,
        low_load_threshold_fraction=0.0,
        low_load_cost_scale=0.0,
        low_load_cost_exponent=2.0,
    )




def cigre_battery_params() -> BatteryParams:

    soc_floor = 4.0 / 33.0

    return BatteryParams(

        num_cells_series=37,

        num_cells_parallel=1,

        nominal_energy_kwh=33.0,

        soc_min=soc_floor,

        soc_max=1.0,

        soc_init=soc_floor,

        p_charge_max=30_000.0,

        p_discharge_max=33_000.0,

        r_int_values=large_format_lfp_r_int_values(),

        thermal_dynamics_enabled=True,

        low_soc_r_int_boost_enabled=True,

        low_soc_r_int_boost_threshold=0.60,

        low_soc_r_int_boost_factor=12.0,

        low_soc_r_int_boost_exponent=1.0,

        power_stress_r_int_boost_enabled=True,

        power_stress_r_int_boost_start_fraction=0.35,

        power_stress_r_int_boost_factor=2.5,

        power_stress_r_int_boost_exponent=1.3,

    )


def cigre_lv_bess_params() -> BatteryParams:
    """Community-scale LV BESS for the CIGRE European LV case."""

    thermal_params = network_scale_lfp_pack_thermal_params(
        num_cells_series=200,
        num_cells_parallel=1,
        thermal_resistance_k_per_w=0.020,
    )
    ocv_base = np.array([2.50, 2.90, 3.05, 3.20, 3.22, 3.24, 3.26, 3.27, 3.29, 3.32, 3.38, 3.50, 3.65])
    ocv_hysteresis_delta = np.array([0.000, 0.004, 0.006, 0.008, 0.009, 0.010, 0.010, 0.009, 0.008, 0.006, 0.004, 0.002, 0.000])
    return BatteryParams(
        cell_capacity_ah=280.0,
        num_cells_series=200,
        num_cells_parallel=1,
        nominal_energy_kwh=179.2,
        soc_min=0.10,
        soc_max=0.90,
        soc_init=0.50,
        p_charge_max=100_000.0,
        p_discharge_max=100_000.0,
        ocv_values=ocv_base,
        ocv_charge_values=ocv_base + ocv_hysteresis_delta,
        ocv_discharge_values=ocv_base - ocv_hysteresis_delta,
        r_int_values=large_format_lfp_r_int_values(),
        thermal_dynamics_enabled=True,
        ambient_temperature_c=thermal_params["ambient_temperature_c"],
        temperature_init_c=thermal_params["temperature_init_c"],
        reference_temperature_c=thermal_params["reference_temperature_c"],
        r_int_temp_coeff_per_c=thermal_params["r_int_temp_coeff_per_c"],
        min_r_int_temp_factor=thermal_params["min_r_int_temp_factor"],
        max_r_int_temp_factor=thermal_params["max_r_int_temp_factor"],
        thermal_resistance_k_per_w=thermal_params["thermal_resistance_k_per_w"],
        thermal_capacitance_j_per_k=thermal_params["thermal_capacitance_j_per_k"],
        temperature_min_c=thermal_params["temperature_min_c"],
        temperature_max_c=thermal_params["temperature_max_c"],
        low_soc_r_int_boost_enabled=True,
        low_soc_r_int_boost_threshold=0.25,
        low_soc_r_int_boost_factor=2.2,
        low_soc_r_int_boost_exponent=1.4,
        power_stress_r_int_boost_enabled=True,
        power_stress_r_int_boost_start_fraction=0.55,
        power_stress_r_int_boost_factor=1.6,
        power_stress_r_int_boost_exponent=1.2,
        ocv_hysteresis_enabled=True,
        rc_branch_1_resistance_values=np.array([0.00075, 0.00068, 0.00060, 0.00053, 0.00048, 0.00044, 0.00042, 0.00044, 0.00048, 0.00053, 0.00060, 0.00068, 0.00075]),
        rc_branch_1_capacitance_values=np.array([8.0e5, 8.8e5, 9.5e5, 1.05e6, 1.15e6, 1.22e6, 1.28e6, 1.22e6, 1.15e6, 1.05e6, 9.5e5, 8.8e5, 8.0e5]),
    )


def ieee33_dess_params() -> BatteryParams:
    """Distribution-scale BESS for the IEEE 33-bus case."""

    thermal_params = network_scale_lfp_pack_thermal_params(
        num_cells_series=250,
        num_cells_parallel=4,
        thermal_resistance_k_per_w=0.012,
    )
    ocv_base = np.array([2.50, 2.90, 3.05, 3.20, 3.22, 3.24, 3.26, 3.27, 3.29, 3.32, 3.38, 3.50, 3.65])
    ocv_hysteresis_delta = np.array([0.000, 0.005, 0.007, 0.009, 0.010, 0.011, 0.011, 0.010, 0.009, 0.007, 0.005, 0.003, 0.000])
    return BatteryParams(
        cell_capacity_ah=280.0,
        num_cells_series=250,
        num_cells_parallel=4,
        nominal_energy_kwh=896.0,
        soc_min=0.10,
        soc_max=0.90,
        soc_init=0.50,
        p_charge_max=500_000.0,
        p_discharge_max=500_000.0,
        ocv_values=ocv_base,
        ocv_charge_values=ocv_base + ocv_hysteresis_delta,
        ocv_discharge_values=ocv_base - ocv_hysteresis_delta,
        r_int_values=large_format_lfp_r_int_values(),
        thermal_dynamics_enabled=True,
        ambient_temperature_c=thermal_params["ambient_temperature_c"],
        temperature_init_c=thermal_params["temperature_init_c"],
        reference_temperature_c=thermal_params["reference_temperature_c"],
        r_int_temp_coeff_per_c=thermal_params["r_int_temp_coeff_per_c"],
        min_r_int_temp_factor=thermal_params["min_r_int_temp_factor"],
        max_r_int_temp_factor=thermal_params["max_r_int_temp_factor"],
        thermal_resistance_k_per_w=thermal_params["thermal_resistance_k_per_w"],
        thermal_capacitance_j_per_k=thermal_params["thermal_capacitance_j_per_k"],
        temperature_min_c=thermal_params["temperature_min_c"],
        temperature_max_c=thermal_params["temperature_max_c"],
        low_soc_r_int_boost_enabled=True,
        low_soc_r_int_boost_threshold=0.20,
        low_soc_r_int_boost_factor=1.8,
        low_soc_r_int_boost_exponent=1.2,
        power_stress_r_int_boost_enabled=True,
        power_stress_r_int_boost_start_fraction=0.60,
        power_stress_r_int_boost_factor=1.5,
        power_stress_r_int_boost_exponent=1.15,
        ocv_hysteresis_enabled=True,
        rc_branch_1_resistance_values=np.array([0.00055, 0.00050, 0.00045, 0.00040, 0.00036, 0.00033, 0.00031, 0.00033, 0.00036, 0.00040, 0.00045, 0.00050, 0.00055]),
        rc_branch_1_capacitance_values=np.array([2.8e6, 3.1e6, 3.4e6, 3.8e6, 4.2e6, 4.5e6, 4.7e6, 4.5e6, 4.2e6, 3.8e6, 3.4e6, 3.1e6, 2.8e6]),
    )





@dataclass

class MicrogridConfig:
    """Unified config for the PCC-based battery scheduling environment."""

    case_name: str = "MG-RES"
    case_key: str = "mg_res"
    simulation_days: int = 30

    dt_seconds: float = 900.0

    battery_model: str = "thevenin"

    use_real_data: bool = True

    strict_reproduction: bool = True
    data_dir: str | None = None

    data_year: int | None = None

    episode_start_hour: int = 0

    random_episode_start: bool = False
    full_year_random_start_stride_hours: int = 1
    full_year_random_start_hours: tuple[int, ...] = field(default_factory=tuple)

    random_initial_soc: bool = False
    initial_soc_min: float = 0.35
    initial_soc_max: float = 0.80
    observation_stack_steps: int = 1
    seed: int = 42
    pv_max_power: float = 6_000.0
    load_max_power: float = 2_100.0
    price_max: float = 0.51373

    feed_in_tariff: float = 0.0
    grid_import_max: float = float("inf")
    grid_export_max: float = float("inf")
    grid_slack_enabled: bool = True
    tou_price_spread_multiplier: float = RESIDENTIAL_GRID_TOU_PRICE_SPREAD_MULTIPLIER
    peak_import_penalty_per_kw: float = 0.0
    peak_import_penalty_threshold_w: float = float("inf")
    monthly_demand_charge_per_kw: float = RESIDENTIAL_GRID_MONTHLY_DEMAND_CHARGE_PER_KW
    monthly_demand_charge_threshold_w: float = RESIDENTIAL_GRID_MONTHLY_DEMAND_CHARGE_THRESHOLD_W
    monthly_peak_increment_penalty_per_kw: float = 0.0
    nse_penalty_per_kwh: float = 0.0
    curtailment_penalty_per_kwh: float = 0.0
    battery_throughput_penalty_per_kwh: float = 0.0
    battery_loss_penalty_per_kwh: float = 0.0
    battery_stress_penalty_per_kwh: float = 0.0
    midday_pv_boost_multiplier: float = 1.0

    midday_pv_boost_start_hour: int = 10

    midday_pv_boost_end_hour: int = 15

    evening_load_boost_multiplier: float = 1.0

    evening_load_boost_start_hour: int = 17

    evening_load_boost_end_hour: int = 22

    stress_episode_sampling: bool = False

    stress_sampling_strength: float = 0.0

    component_commitment_enabled: bool = False

    include_component_cost_in_objective: bool = True

    generator_enabled: bool = False
    generator_params: DispatchableUnitParams | None = field(default_factory=residential_generator_params)

    reward_mode: str = "legacy"

    battery_params: BatteryParams = field(default_factory=residential_battery_params)

    reward: RewardConfig = field(default_factory=RewardConfig)





@dataclass

class CIGREConfig(MicrogridConfig):

    """MG-CIGRE configuration tuned for PBM-vs-EBM comparison."""



    case_name: str = "MG-CIGRE"

    case_key: str = "mg_cigre"

    pv_max_power: float = 13_000.0

    load_max_power: float = 40_000.0

    data_year: int | None = 2024

    component_commitment_enabled: bool = False

    include_component_cost_in_objective: bool = False

    generator_enabled: bool = False

    generator_params: DispatchableUnitParams | None = None

    reward_mode: str = "cost"

    tou_price_spread_multiplier: float = 8.0

    peak_import_penalty_per_kw: float = 1.50

    peak_import_penalty_threshold_w: float = 10_000.0

    midday_pv_boost_multiplier: float = 1.25

    evening_load_boost_multiplier: float = 1.35

    stress_episode_sampling: bool = True

    stress_sampling_strength: float = 6.0

    battery_params: BatteryParams = field(default_factory=cigre_battery_params)


@dataclass
class NetworkCaseConfig:
    """Shared config for pandapower-backed network cases."""

    case_name: str = "CIGRE-EU-LV"
    case_key: str = "cigre_eu_lv_network"
    benchmark_name: str = "CIGRE European LV"
    simulation_days: int = 30
    dt_seconds: float = 900.0
    battery_model: str = "thevenin"
    reward_profile: str = "network"
    use_real_data: bool = False
    strict_reproduction: bool = False
    data_dir: str | None = None
    data_year: int | None = 2024
    episode_start_hour: int = 0
    random_episode_start: bool = False
    regime: str = "base"
    random_initial_soc: bool = False
    initial_soc_min: float = 0.35
    initial_soc_max: float = 0.80
    observation_stack_steps: int = 1
    seed: int = 42
    pv_max_power: float = 15_000.0
    load_max_power: float = 120_000.0
    price_max: float = 1.50
    feed_in_tariff: float = 0.0
    grid_import_max: float = float("inf")
    grid_export_max: float = float("inf")
    grid_limit_violation_penalty_per_kwh: float = 2.0
    tou_price_spread_multiplier: float = 1.0
    nse_penalty_per_kwh: float = 0.0
    curtailment_penalty_per_kwh: float = 0.0
    battery_throughput_penalty_per_kwh: float = 0.0
    battery_loss_penalty_per_kwh: float = 0.0
    battery_stress_penalty_per_kwh: float = 0.0
    terminal_soc_target: float | None = None
    terminal_soc_tolerance: float = 0.05
    terminal_soc_penalty_per_unit: float = 0.0
    network_voltage_min_pu: float = 0.95
    network_voltage_max_pu: float = 1.05
    network_line_loading_limit_pct: float = 100.0
    network_transformer_loading_limit_pct: float = 100.0
    battery_bus_name: str = "Battery Bus"
    pv_bus_names: tuple[str, ...] = field(default_factory=tuple)
    battery_params: BatteryParams = field(default_factory=cigre_battery_params)
    reward: RewardConfig = field(default_factory=RewardConfig)


@dataclass
class CIGREEuropeanLVConfig(NetworkCaseConfig):
    case_name: str = "CIGRE-EU-LV"
    case_key: str = "cigre_eu_lv_network"
    benchmark_name: str = "CIGRE European LV"
    pv_max_power: float = 18_000.0
    load_max_power: float = 120_000.0
    feed_in_tariff: float = 0.18
    grid_import_max: float = 0.15
    grid_export_max: float = 0.05
    grid_limit_violation_penalty_per_kwh: float = 2.5
    tou_price_spread_multiplier: float = 2.5
    battery_bus_name: str = "Bus R18"
    pv_bus_names: tuple[str, ...] = ("Bus R11", "Bus R15", "Bus R17")
    battery_params: BatteryParams = field(default_factory=cigre_lv_bess_params)

    def __post_init__(self):
        if self.battery_model == "none":
            self.battery_params = make_no_dispatch_battery_params(self.battery_params)
        if self.battery_model == "thevenin_loss_only":
            self.battery_params = make_loss_only_battery_params(self.battery_params)
        if self.reward_profile == "paper_aligned":
            self.reward = make_paper_aligned_reward_config(self.reward)
            self.battery_throughput_penalty_per_kwh = 0.0
            self.battery_loss_penalty_per_kwh = 0.0
            self.battery_stress_penalty_per_kwh = 0.0
        if self.reward_profile == "paper_balanced":
            self.reward = make_paper_balanced_reward_config(self.reward)
            self.battery_throughput_penalty_per_kwh = 0.001
            self.battery_loss_penalty_per_kwh = 0.01
            self.battery_stress_penalty_per_kwh = 0.002
            self.terminal_soc_target = float(self.battery_params.soc_init)
            self.terminal_soc_tolerance = 0.05
            self.terminal_soc_penalty_per_unit = 400.0


@dataclass
class IEEE33Config(NetworkCaseConfig):
    case_name: str = "IEEE33"
    case_key: str = "ieee33_network"
    benchmark_name: str = "IEEE 33-bus"
    pv_max_power: float = 450_000.0
    load_max_power: float = 4_000_000.0
    feed_in_tariff: float = 0.20
    grid_import_max: float = 5.0
    grid_export_max: float = 0.25
    grid_limit_violation_penalty_per_kwh: float = 1.5
    tou_price_spread_multiplier: float = 3.0
    battery_bus_name: str = "Bus 33"
    pv_bus_names: tuple[str, ...] = ("Bus 25", "Bus 30", "Bus 31")
    battery_params: BatteryParams = field(default_factory=ieee33_dess_params)
    network_voltage_min_pu: float = 0.90
    network_line_loading_limit_pct: float = 100.0
    battery_throughput_penalty_per_kwh: float = 0.010
    battery_loss_penalty_per_kwh: float = 0.020
    battery_stress_penalty_per_kwh: float = 0.010
    reward: RewardConfig = field(
        default_factory=lambda: RewardConfig(
            w_cost=0.02,
            w_voltage_violation=220.0,
            w_line_overload=180.0,
            w_transformer_overload=80.0,
            w_band=8.0,
            w_edge=40.0,
            soc_center=0.55,
            soc_sigma=0.16,
            soc_band_min=0.30,
            soc_band_max=0.85,
            reward_min=-120.0,
            reward_max=200.0,
        )
    )

    def __post_init__(self):
        if self.battery_model == "none":
            self.battery_params = make_no_dispatch_battery_params(self.battery_params)
        if self.battery_model == "thevenin_loss_only":
            self.battery_params = make_loss_only_battery_params(self.battery_params)
        if self.reward_profile == "paper_aligned":
            self.reward = make_paper_aligned_reward_config(self.reward)
            self.battery_throughput_penalty_per_kwh = 0.0
            self.battery_loss_penalty_per_kwh = 0.0
            self.battery_stress_penalty_per_kwh = 0.0
        if self.reward_profile == "paper_balanced":
            self.reward = make_paper_balanced_reward_config(self.reward)
            self.battery_throughput_penalty_per_kwh = 0.001
            self.battery_loss_penalty_per_kwh = 0.01
            self.battery_stress_penalty_per_kwh = 0.002
            self.terminal_soc_target = float(self.battery_params.soc_init)
            self.terminal_soc_tolerance = 0.05
            self.terminal_soc_penalty_per_unit = 400.0

