import pytest

from microgrid_sim.cases import CIGREEuropeanLVConfig, IEEE33Config, NetworkCaseConfig
from microgrid_sim.envs import network_microgrid as network_microgrid_module
from microgrid_sim.envs.network_microgrid import NetworkMicrogridEnv


def test_network_microgrid_env_reset_and_step():
    env = NetworkMicrogridEnv(CIGREEuropeanLVConfig(simulation_days=1))
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape
    assert "min_bus_voltage_pu" in info
    assert "power_flow_converged" in info
    assert env.config.battery_bus_name == "Bus R11"
    assert env.net.user_metadata["battery_bus_index"] == 12
    assert env.net.user_metadata["storage_power_mw"] == 0.2
    assert env.net.user_metadata["storage_energy_mwh"] == 0.3584

    next_obs, reward, terminated, truncated, step_info = env.step(env.action_space.sample())
    assert next_obs.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "max_line_loading_pct" in step_info
    assert "battery_loss_kwh" in step_info
    assert "battery_stress_kwh" in step_info
    assert "net_energy_cost" in step_info
    assert "total_grid_cost" in step_info
    assert "grid_limit_penalty_cost" in step_info
    assert "power_flow_failed" in step_info


def test_network_microgrid_env_horizon_end_is_truncated_not_terminated():
    env = NetworkMicrogridEnv(CIGREEuropeanLVConfig(simulation_days=1))
    try:
        env.reset(seed=7)
        terminated = False
        truncated = False
        for _ in range(env.total_steps):
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if terminated or truncated:
                break
        assert terminated is False
        assert truncated is True
    finally:
        env.close()


def test_network_microgrid_env_power_flow_failure_is_terminated(monkeypatch: pytest.MonkeyPatch):
    env = NetworkMicrogridEnv(CIGREEuropeanLVConfig(simulation_days=1))
    try:
        env.reset(seed=7)

        def _fail_power_flow(_net):
            return {"converged": False, "failed": True, "failure_reason": "synthetic_test_failure"}

        monkeypatch.setattr(network_microgrid_module, "run_power_flow", _fail_power_flow)
        _, _, terminated, truncated, info = env.step(env.action_space.sample())
        assert terminated is True
        assert truncated is False
        assert info["power_flow_failed"] is True
        assert info["power_flow_failure_reason"] == "synthetic_test_failure"
    finally:
        env.close()


def test_ieee33_env_uses_distribution_scale_bess():
    config = IEEE33Config(simulation_days=1)
    assert config.battery_params.nominal_energy_kwh == 896.0
    assert config.battery_params.p_discharge_max == 500_000.0
    assert config.pv_max_power == 450_000.0
    assert config.network_voltage_min_pu == 0.90
    assert config.network_line_loading_limit_pct == 100.0

    env = NetworkMicrogridEnv(config)
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape
    assert info["case_name"] == "IEEE33"
    assert env.net.user_metadata["storage_role"] == "distribution_scale_dess"
    assert env.net.user_metadata["battery_bus_index"] == 32
    assert "grid_import_mw" in info
    assert config.battery_bus_name == "Bus 33"


def test_ieee33_battery_action_sign_changes_soc_and_network_response():
    baseline_config = IEEE33Config(simulation_days=1, random_initial_soc=False)

    for battery_model in ("simple", "thevenin", "thevenin_loss_only"):
        config = IEEE33Config(
            simulation_days=baseline_config.simulation_days,
            random_initial_soc=baseline_config.random_initial_soc,
            battery_model=battery_model,
        )

        idle_env = NetworkMicrogridEnv(config)
        idle_env.reset(seed=123)
        _, _, _, _, idle_info = idle_env.step([0.0])
        idle_env.close()

        charge_env = NetworkMicrogridEnv(config)
        charge_env.reset(seed=123)
        _, _, _, _, charge_info = charge_env.step([-1.0])
        charge_env.close()

        discharge_env = NetworkMicrogridEnv(config)
        discharge_env.reset(seed=123)
        _, _, _, _, discharge_info = discharge_env.step([1.0])
        discharge_env.close()

        assert charge_info["battery_power_mw"] < idle_info["battery_power_mw"]
        assert charge_info["soc"] > idle_info["soc"]
        assert charge_info["grid_import_mw"] > idle_info["grid_import_mw"]
        assert charge_info["min_bus_voltage_pu"] < idle_info["min_bus_voltage_pu"]

        assert discharge_info["battery_power_mw"] > idle_info["battery_power_mw"]
        assert discharge_info["soc"] < idle_info["soc"]
        assert discharge_info["grid_import_mw"] < idle_info["grid_import_mw"]
        assert discharge_info["min_bus_voltage_pu"] > idle_info["min_bus_voltage_pu"]


def test_battery_actions_scale_to_current_power_bounds():
    env = NetworkMicrogridEnv(IEEE33Config(simulation_days=1, seed=42, battery_model="simple", regime="base"))
    try:
        env.reset(seed=42)
        env.battery.soc = float(env.config.battery_params.soc_max)
        assert env._battery_power_command([-1.0]) == pytest.approx(0.0)

        env.battery.soc = float(env.config.battery_params.soc_min)
        assert env._battery_power_command([1.0]) == pytest.approx(0.0)

        env.battery.soc = 0.5
        min_command_w, max_command_w = env.battery.power_command_bounds(dt=float(env.config.dt_seconds))
        assert env._battery_power_command([1.0]) == pytest.approx(max_command_w)
        assert env._battery_power_command([-1.0]) == pytest.approx(min_command_w)
    finally:
        env.close()


def test_none_battery_model_disables_storage_dispatch_effects():
    config = IEEE33Config(simulation_days=1, random_initial_soc=False, battery_model="none", reward_profile="paper_balanced")
    assert config.battery_params.p_charge_max == 0.0
    assert config.battery_params.p_discharge_max == 0.0

    idle_env = NetworkMicrogridEnv(config)
    idle_env.reset(seed=123)
    _, idle_reward, _, _, idle_info = idle_env.step([0.0])
    idle_env.close()

    charge_env = NetworkMicrogridEnv(config)
    charge_env.reset(seed=123)
    _, charge_reward, _, _, charge_info = charge_env.step([-1.0])
    charge_env.close()

    discharge_env = NetworkMicrogridEnv(config)
    discharge_env.reset(seed=123)
    _, discharge_reward, _, _, discharge_info = discharge_env.step([1.0])
    discharge_env.close()

    assert idle_info["battery_power_mw"] == 0.0
    assert charge_info["battery_power_mw"] == 0.0
    assert discharge_info["battery_power_mw"] == 0.0
    assert charge_info["soc"] == idle_info["soc"] == discharge_info["soc"]
    assert charge_info["grid_import_mw"] == idle_info["grid_import_mw"] == discharge_info["grid_import_mw"]
    assert charge_info["battery_loss_kwh"] == 0.0
    assert charge_info["battery_stress_kwh"] == 0.0
    assert charge_reward == idle_reward == discharge_reward


def test_cigre_export_revenue_and_export_limit_penalty_are_reported():
    config = CIGREEuropeanLVConfig(
        simulation_days=1,
        random_initial_soc=False,
        battery_model="simple",
        regime="high_pv",
        feed_in_tariff=0.20,
        grid_export_max=0.01,
        load_max_power=120_000.0,
        pv_max_power=200_000.0,
        tou_price_spread_multiplier=1.0,
    )
    env = NetworkMicrogridEnv(config)
    try:
        env.reset(seed=123)
        _, _, _, _, info = env.step([1.0])
        assert info["grid_export_mw"] > 0.0
        assert info["export_revenue"] > 0.0
        assert info["grid_export_limit_violation_mw"] > 0.0
        assert info["grid_limit_penalty_cost"] > 0.0
        assert info["total_grid_cost"] >= info["net_energy_cost"]
    finally:
        env.close()


def test_ieee33_import_limit_penalty_is_reported_on_heavy_charging():
    config = IEEE33Config(
        simulation_days=1,
        random_initial_soc=False,
        battery_model="simple",
        regime="network_stress",
        grid_import_max=1.0,
    )
    env = NetworkMicrogridEnv(config)
    try:
        env.reset(seed=123)
        _, _, _, _, info = env.step([-1.0])
        assert info["grid_import_mw"] > 1.0
        assert info["grid_import_limit_violation_mw"] > 0.0
        assert info["grid_limit_penalty_cost"] > 0.0
        assert info["total_grid_cost"] > info["import_cost"]
    finally:
        env.close()


def test_paper_aligned_reward_profile_disables_explicit_battery_penalties():
    config = IEEE33Config(simulation_days=1, battery_model="thevenin_loss_only", reward_profile="paper_aligned")
    env = NetworkMicrogridEnv(config)
    try:
        env.reset(seed=123)
        assert env.config.reward.w_cost == 1.0
        assert env.config.reward.w_band < 1.0
        assert env.config.battery_loss_penalty_per_kwh == 0.0
        assert env.config.battery_stress_penalty_per_kwh == 0.0
        assert env.config.battery_throughput_penalty_per_kwh == 0.0
        assert env.battery.params.paper_soc_nonlinearity_enabled is True
        assert env.battery.params.thermal_dynamics_enabled is False
    finally:
        env.close()


def test_paper_balanced_reward_profile_uses_moderate_battery_shaping():
    config = IEEE33Config(simulation_days=1, battery_model="thevenin", reward_profile="paper_balanced")
    env = NetworkMicrogridEnv(config)
    try:
        env.reset(seed=123)
        assert env.config.reward.w_cost == 1.0
        assert env.config.reward.w_band > 1.0
        assert env.config.reward.w_edge >= 8.0
        assert env.config.battery_throughput_penalty_per_kwh > 0.0
        assert env.config.battery_loss_penalty_per_kwh > 0.0
        assert env.config.battery_stress_penalty_per_kwh > 0.0
        assert env.config.terminal_soc_penalty_per_kwh > 0.0
        assert env.config.terminal_soc_target == env.config.battery_params.soc_init
        assert env.battery.params.thermal_dynamics_enabled is True
    finally:
        env.close()


def test_network_microgrid_env_rejects_unknown_case_key():
    with pytest.raises(ValueError, match="Unsupported network case_key"):
        NetworkMicrogridEnv(NetworkCaseConfig(case_key="unsupported_case"))


def test_network_microgrid_env_rejects_unknown_battery_model():
    with pytest.raises(ValueError, match="Unsupported battery_model"):
        NetworkMicrogridEnv(CIGREEuropeanLVConfig(simulation_days=1, battery_model="unsupported_model"))
