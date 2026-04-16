from microgrid_sim.cases import IEEE33Config
from microgrid_sim.envs.reward_builder import build_network_reward


def test_reward_builder_penalizes_low_soc_edge_more_than_mid_soc():
    config = IEEE33Config()
    metrics = {
        "min_bus_voltage_pu": 1.0,
        "max_bus_voltage_pu": 1.0,
        "max_line_loading_pct": 0.0,
        "max_transformer_loading_pct": 0.0,
    }
    low_soc_reward, low_penalties = build_network_reward(
        config,
        battery_info={"soc": 0.10, "soc_violation": 0.0, "effective_power": 0.0, "power_loss": 0.0, "r_int_power_factor": 1.0},
        metrics=metrics,
        import_cost=0.0,
    )
    mid_soc_reward, mid_penalties = build_network_reward(
        config,
        battery_info={"soc": 0.55, "soc_violation": 0.0, "effective_power": 0.0, "power_loss": 0.0, "r_int_power_factor": 1.0},
        metrics=metrics,
        import_cost=0.0,
    )
    assert mid_penalties["soc_edge_penalty"] < low_penalties["soc_edge_penalty"]
    assert mid_penalties["soc_center_penalty"] < low_penalties["soc_center_penalty"]
    assert mid_soc_reward > low_soc_reward


def test_reward_builder_soft_edge_penalty_is_active_inside_soc_band():
    config = IEEE33Config()
    metrics = {
        "min_bus_voltage_pu": 1.0,
        "max_bus_voltage_pu": 1.0,
        "max_line_loading_pct": 0.0,
        "max_transformer_loading_pct": 0.0,
    }
    centered_reward, centered_penalties = build_network_reward(
        config,
        battery_info={"soc": 0.55, "soc_violation": 0.0, "effective_power": 0.0, "power_loss": 0.0, "r_int_power_factor": 1.0},
        metrics=metrics,
        import_cost=0.0,
    )
    off_center_reward, off_center_penalties = build_network_reward(
        config,
        battery_info={"soc": 0.70, "soc_violation": 0.0, "effective_power": 0.0, "power_loss": 0.0, "r_int_power_factor": 1.0},
        metrics=metrics,
        import_cost=0.0,
    )
    assert centered_penalties["soc_edge_penalty"] == 0.0
    assert off_center_penalties["soc_edge_penalty"] > 0.0
    assert off_center_reward < centered_reward


def test_reward_builder_applies_terminal_soc_penalty_only_on_terminal_step():
    config = IEEE33Config(reward_profile="paper_balanced")
    metrics = {
        "min_bus_voltage_pu": 1.0,
        "max_bus_voltage_pu": 1.0,
        "max_line_loading_pct": 0.0,
        "max_transformer_loading_pct": 0.0,
    }
    battery_info = {"soc": 0.20, "soc_violation": 0.0, "effective_power": 0.0, "power_loss": 0.0, "r_int_power_factor": 1.0}
    non_terminal_reward, non_terminal_penalties = build_network_reward(
        config,
        battery_info=battery_info,
        metrics=metrics,
        import_cost=0.0,
        is_terminal=False,
    )
    terminal_reward, terminal_penalties = build_network_reward(
        config,
        battery_info=battery_info,
        metrics=metrics,
        import_cost=0.0,
        is_terminal=True,
    )
    assert non_terminal_penalties["terminal_soc_penalty"] == 0.0
    assert terminal_penalties["terminal_soc_penalty"] > 0.0
    assert terminal_penalties["terminal_soc_deviation"] > terminal_penalties["terminal_soc_tolerance"]
    expected_excess_kwh = (
        (terminal_penalties["terminal_soc_deviation"] - terminal_penalties["terminal_soc_tolerance"])
        * config.battery_params.nominal_energy_wh
        / 1000.0
    )
    assert abs(terminal_penalties["terminal_soc_excess_kwh"] - expected_excess_kwh) < 1e-9
    assert abs(
        terminal_penalties["terminal_soc_penalty"]
        - terminal_penalties["terminal_soc_excess_kwh"] * config.terminal_soc_penalty_per_kwh
    ) < 1e-9
    assert non_terminal_penalties["battery_shaping_penalty"] > 0.0
    assert non_terminal_penalties["reward_after_battery_shaping"] == non_terminal_reward
    assert terminal_penalties["reward_after_battery_shaping"] == non_terminal_reward
    assert non_terminal_penalties["step_reward_after_clip"] > non_terminal_reward
    assert terminal_penalties["reward_after_terminal_penalty"] == terminal_reward
    assert terminal_reward == non_terminal_reward - terminal_penalties["terminal_soc_penalty"]
    assert terminal_reward < config.reward.reward_min
    assert terminal_reward < non_terminal_reward


def test_reward_builder_penalizes_low_peak_reserve_only_during_high_price_steps():
    config = IEEE33Config(reward_profile="paper_balanced")
    metrics = {
        "min_bus_voltage_pu": 1.0,
        "max_bus_voltage_pu": 1.0,
        "max_line_loading_pct": 0.0,
        "max_transformer_loading_pct": 0.0,
    }
    battery_info = {
        "soc": 0.15,
        "soc_violation": 0.0,
        "effective_power": 0.0,
        "power_loss": 0.0,
        "r_int_power_factor": 1.0,
        "battery_discharge_power_limit": 50_000.0,
    }
    off_peak_reward, off_peak_penalties = build_network_reward(
        config,
        battery_info=battery_info,
        metrics=metrics,
        import_cost=0.0,
        price=config.reward.peak_price - 0.05,
    )
    peak_reward, peak_penalties = build_network_reward(
        config,
        battery_info=battery_info,
        metrics=metrics,
        import_cost=0.0,
        price=config.reward.peak_price + 0.05,
    )
    assert off_peak_penalties["peak_reserve_penalty"] == 0.0
    assert peak_penalties["discharge_limit_ratio"] < config.reward.peak_reserve_power_floor
    assert peak_penalties["peak_reserve_shortfall"] > 0.0
    assert peak_penalties["peak_reserve_penalty"] > 0.0
    assert peak_reward < off_peak_reward


def test_reward_builder_applies_peak_reserve_penalty_after_step_clipping():
    config = IEEE33Config(reward_profile="paper_balanced")
    metrics = {
        "min_bus_voltage_pu": 1.0,
        "max_bus_voltage_pu": 1.0,
        "max_line_loading_pct": 0.0,
        "max_transformer_loading_pct": 0.0,
    }
    battery_info = {
        "soc": 0.12,
        "soc_violation": 0.0,
        "effective_power": 0.0,
        "power_loss": 0.0,
        "r_int_power_factor": 1.0,
        "battery_discharge_power_limit": 100_000.0,
    }
    reward, penalties = build_network_reward(
        config,
        battery_info=battery_info,
        metrics=metrics,
        import_cost=550.0,
        price=config.reward.peak_price + 0.05,
        is_terminal=False,
    )

    assert penalties["step_reward_before_clip"] < config.reward.reward_min
    assert penalties["step_reward_after_clip"] == config.reward.reward_min
    assert penalties["peak_reserve_penalty"] > 0.0
    assert penalties["reward_after_peak_reserve_penalty"] == penalties["step_reward_after_clip"] - penalties["peak_reserve_penalty"]
    assert penalties["battery_shaping_penalty"] >= penalties["peak_reserve_penalty"]
    assert penalties["reward_after_battery_shaping"] == penalties["step_reward_after_clip"] - penalties["battery_shaping_penalty"]
    assert penalties["reward_after_terminal_penalty"] == reward
    assert reward < penalties["step_reward_after_clip"]
