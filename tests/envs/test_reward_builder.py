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
    assert non_terminal_penalties["step_reward_after_clip"] == non_terminal_reward
    assert terminal_penalties["step_reward_after_clip"] == non_terminal_reward
    assert terminal_penalties["reward_after_terminal_penalty"] == terminal_reward
    assert terminal_reward == non_terminal_reward - terminal_penalties["terminal_soc_penalty"]
    assert terminal_reward < config.reward.reward_min
    assert terminal_reward < non_terminal_reward
