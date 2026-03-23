"""Case configurations for the studied microgrids."""



from __future__ import annotations



from dataclasses import dataclass, field

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

    w_soh: float = 2.0

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





@dataclass

class MicrogridConfig:
    """Unified config for the PCC-based battery scheduling environment."""

    case_name: str = "MG-RES"
    case_key: str = "mg_res"
    simulation_days: int = 30

    dt_seconds: float = 3600.0

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

