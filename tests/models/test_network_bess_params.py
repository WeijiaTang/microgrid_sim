import numpy as np
import pytest

from microgrid_sim.cases import (
    CIGREEuropeanLVConfig,
    IEEE33Config,
    cigre_lv_bess_params,
    ieee33_dess_params,
    make_loss_only_battery_params,
    make_paper_aligned_reward_config,
    make_paper_balanced_reward_config,
)
from microgrid_sim.models.battery import BatteryParams, SimpleBattery, TheveninBattery


def test_cigre_lv_bess_enables_effective_hysteresis_and_1rc():
    params = cigre_lv_bess_params()
    assert params.ocv_hysteresis_enabled is True
    assert not np.allclose(params.ocv_charge_values, params.ocv_discharge_values)
    assert params.rc_branch_1_resistance_values is not None
    assert params.rc_branch_1_capacitance_values is not None


def test_ieee33_dess_enables_effective_hysteresis_and_1rc():
    params = ieee33_dess_params()
    assert params.ocv_hysteresis_enabled is True
    assert not np.allclose(params.ocv_charge_values, params.ocv_discharge_values)
    assert params.rc_branch_1_resistance_values is not None
    assert params.rc_branch_1_capacitance_values is not None


def test_loss_only_battery_params_disable_full_physics_but_keep_soc_nonlinearity():
    params = make_loss_only_battery_params(ieee33_dess_params())
    assert params.thermal_dynamics_enabled is False
    assert params.low_soc_r_int_boost_enabled is False
    assert params.power_stress_r_int_boost_enabled is False
    assert params.ocv_hysteresis_enabled is False
    assert params.rc_branch_1_resistance_values is None
    assert params.rc_branch_1_capacitance_values is None
    assert params.paper_soc_nonlinearity_enabled is True
    assert params.paper_soc_discharge_nonlinearity_gain > 0.0


def test_paper_aligned_reward_config_prioritizes_cost_over_soc_shaping():
    reward = make_paper_aligned_reward_config()
    assert reward.w_cost == 1.0
    assert reward.w_band < 1.0
    assert reward.w_edge < 5.0


def test_paper_balanced_reward_config_reintroduces_moderate_shaping():
    reward = make_paper_balanced_reward_config()
    assert reward.w_cost == 1.0
    assert reward.w_band > 1.0
    assert reward.w_edge >= 8.0
    assert reward.soc_band_min >= 0.18
    assert reward.soc_band_max <= 0.88


def test_battery_reset_restores_pack_state_without_soh_tracking():
    params = ieee33_dess_params()

    for battery_cls in (SimpleBattery, TheveninBattery):
        battery = battery_cls(params)
        battery.reset(soc=0.5)
        _, _, info = battery.step(200_000.0, 3600.0)
        assert "soh" not in info
        battery.reset(soc=0.5)
        assert battery.soc == 0.5


def test_battery_params_reject_large_nominal_energy_mismatch():
    with pytest.raises(ValueError, match="inconsistent with the configured cell stack energy"):
        BatteryParams(
            cell_capacity_ah=280.0,
            num_cells_series=200,
            num_cells_parallel=1,
            nominal_energy_kwh=200.0,
        )


def test_battery_params_reject_lookup_length_mismatch():
    with pytest.raises(ValueError, match="does not match soc_breakpoints length"):
        BatteryParams(
            soc_breakpoints=np.array([0.0, 0.5, 1.0]),
            ocv_values=np.array([3.0, 3.2, 3.4]),
            ocv_charge_values=np.array([3.0, 3.2, 3.4]),
            ocv_discharge_values=np.array([3.0, 3.2, 3.4]),
            r_int_values=np.array([0.0002, 0.0003]),
        )


def test_battery_params_reject_invalid_thermal_configuration():
    with pytest.raises(ValueError, match="thermal_capacitance_j_per_k must be strictly positive"):
        BatteryParams(thermal_capacitance_j_per_k=0.0)

    with pytest.raises(ValueError, match="temperature_init_c must lie within"):
        BatteryParams(temperature_min_c=0.0, temperature_max_c=55.0, temperature_init_c=60.0)


def test_network_bess_cases_use_consistent_energy_and_large_format_resistance():
    for config in (CIGREEuropeanLVConfig(), IEEE33Config()):
        params = config.battery_params
        stack_energy_kwh = params.cell_capacity_ah * 3.2 * params.num_cells_series * params.num_cells_parallel / 1000.0

        assert params.nominal_energy_kwh == stack_energy_kwh
        assert np.all(np.asarray(params.r_int_values, dtype=float) > 0.0)
        assert float(np.max(np.asarray(params.r_int_values, dtype=float) * 1000.0)) < 1.0


def test_network_bess_cases_use_explicit_pack_scale_thermal_calibration():
    cigre = cigre_lv_bess_params()
    ieee33 = ieee33_dess_params()

    assert cigre.thermal_capacitance_j_per_k == pytest.approx(1_944_000.0)
    assert ieee33.thermal_capacitance_j_per_k == pytest.approx(4_860_000.0)
    assert cigre.thermal_resistance_k_per_w == pytest.approx(0.020)
    assert ieee33.thermal_resistance_k_per_w == pytest.approx(0.012)
    assert cigre.r_int_temp_coeff_per_c == pytest.approx(0.010)
    assert ieee33.r_int_temp_coeff_per_c == pytest.approx(0.010)
    assert cigre.temperature_max_c == pytest.approx(55.0)
    assert ieee33.temperature_max_c == pytest.approx(55.0)


def test_cigre_lv_config_defaults_to_enhanced_r11_bess():
    config = CIGREEuropeanLVConfig()

    assert config.battery_bus_name == "Bus R11"
    assert config.battery_params.nominal_energy_kwh == pytest.approx(358.4)
    assert config.battery_params.p_discharge_max == pytest.approx(200_000.0)


def test_temperature_resistance_factor_is_monotone_for_network_scale_bess():
    battery = TheveninBattery(ieee33_dess_params())

    battery.temperature_c = 5.0
    cold_factor = battery._temperature_r_int_factor()
    battery.temperature_c = 25.0
    nominal_factor = battery._temperature_r_int_factor()
    battery.temperature_c = 45.0
    hot_factor = battery._temperature_r_int_factor()

    assert cold_factor == pytest.approx(1.20)
    assert nominal_factor == pytest.approx(1.00)
    assert hot_factor == pytest.approx(0.80)
    assert cold_factor > nominal_factor > hot_factor


def test_network_scale_thevenin_bess_does_not_saturate_temperature_in_one_full_power_hour():
    for params in (cigre_lv_bess_params(), ieee33_dess_params()):
        battery = TheveninBattery(params)
        battery.reset(soc=params.soc_init)
        _, _, info = battery.step(params.p_discharge_max, 3600.0)

        assert info["temperature_c"] > params.temperature_init_c
        assert info["temperature_c"] < 35.0
        assert info["temperature_c"] < params.temperature_max_c
