import numpy as np

from microgrid_sim.cases import (
    CIGREEuropeanLVConfig,
    IEEE33ModifiedConfig,
    cigre_lv_bess_params,
    ieee33_dess_params,
    make_loss_only_battery_params,
    make_paper_aligned_reward_config,
    make_paper_balanced_reward_config,
)
from microgrid_sim.models.battery import SimpleBattery, TheveninBattery


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
    assert reward.w_soh == 0.0


def test_paper_balanced_reward_config_reintroduces_moderate_shaping():
    reward = make_paper_balanced_reward_config()
    assert reward.w_cost == 1.0
    assert reward.w_band > 1.0
    assert reward.w_edge >= 8.0
    assert reward.w_soh == 0.0
    assert reward.soc_band_min >= 0.18
    assert reward.soc_band_max <= 0.88


def test_battery_reset_restores_soh_to_fresh_state():
    params = ieee33_dess_params()

    for battery_cls in (SimpleBattery, TheveninBattery):
        battery = battery_cls(params)
        battery.reset(soc=0.5)
        battery.step(200_000.0, 3600.0)
        assert battery.soh < 1.0

        battery.reset(soc=0.5)
        assert battery.soh == 1.0


def test_network_bess_cases_use_consistent_energy_and_large_format_resistance():
    for config in (CIGREEuropeanLVConfig(), IEEE33ModifiedConfig()):
        params = config.battery_params
        stack_energy_kwh = params.cell_capacity_ah * 3.2 * params.num_cells_series * params.num_cells_parallel / 1000.0

        assert params.nominal_energy_kwh == stack_energy_kwh
        assert np.all(np.asarray(params.r_int_values, dtype=float) > 0.0)
        assert float(np.max(np.asarray(params.r_int_values, dtype=float) * 1000.0)) < 1.0
