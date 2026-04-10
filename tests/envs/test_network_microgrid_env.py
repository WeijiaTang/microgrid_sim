from microgrid_sim.cases import CIGREEuropeanLVConfig, IEEE33ModifiedConfig
from microgrid_sim.envs.network_microgrid import NetworkMicrogridEnv


def test_network_microgrid_env_reset_and_step():
    env = NetworkMicrogridEnv(CIGREEuropeanLVConfig(simulation_days=1))
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape
    assert "min_bus_voltage_pu" in info
    assert "power_flow_converged" in info

    next_obs, reward, terminated, truncated, step_info = env.step(env.action_space.sample())
    assert next_obs.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "max_line_loading_pct" in step_info
    assert "battery_loss_kwh" in step_info
    assert "battery_stress_kwh" in step_info
    assert "power_flow_failed" in step_info


def test_ieee33_env_uses_distribution_scale_bess():
    config = IEEE33ModifiedConfig(simulation_days=1)
    assert config.battery_params.nominal_energy_kwh == 1_000.0
    assert config.battery_params.p_discharge_max == 500_000.0
    assert config.pv_max_power == 450_000.0
    assert config.network_voltage_min_pu == 0.94
    assert config.network_line_loading_limit_pct == 95.0

    env = NetworkMicrogridEnv(config)
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape
    assert info["case_name"] == "IEEE33-MOD"
    assert env.net.user_metadata["storage_role"] == "distribution_scale_dess"
    assert env.net.user_metadata["battery_bus_index"] == 32
    assert "grid_import_mw" in info
    assert config.battery_bus_name == "Bus 33"


def test_ieee33_battery_action_sign_changes_soc_and_network_response():
    baseline_config = IEEE33ModifiedConfig(simulation_days=1, random_initial_soc=False)

    for battery_model in ("simple", "thevenin", "thevenin_loss_only"):
        config = IEEE33ModifiedConfig(
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


def test_none_battery_model_disables_storage_dispatch_effects():
    config = IEEE33ModifiedConfig(simulation_days=1, random_initial_soc=False, battery_model="none", reward_profile="paper_balanced")
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


def test_paper_aligned_reward_profile_disables_explicit_battery_penalties():
    config = IEEE33ModifiedConfig(simulation_days=1, battery_model="thevenin_loss_only", reward_profile="paper_aligned")
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
    config = IEEE33ModifiedConfig(simulation_days=1, battery_model="thevenin", reward_profile="paper_balanced")
    env = NetworkMicrogridEnv(config)
    try:
        env.reset(seed=123)
        assert env.config.reward.w_cost == 1.0
        assert env.config.reward.w_band > 1.0
        assert env.config.reward.w_edge >= 8.0
        assert env.config.battery_throughput_penalty_per_kwh > 0.0
        assert env.config.battery_loss_penalty_per_kwh > 0.0
        assert env.config.battery_stress_penalty_per_kwh > 0.0
        assert env.battery.params.thermal_dynamics_enabled is True
    finally:
        env.close()
