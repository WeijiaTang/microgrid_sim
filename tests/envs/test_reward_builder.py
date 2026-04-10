from microgrid_sim.cases import IEEE33ModifiedConfig
from microgrid_sim.envs.reward_builder import build_network_reward


def test_reward_builder_penalizes_low_soc_edge_more_than_mid_soc():
    config = IEEE33ModifiedConfig()
    metrics = {
        "min_bus_voltage_pu": 1.0,
        "max_bus_voltage_pu": 1.0,
        "max_line_loading_pct": 0.0,
        "max_transformer_loading_pct": 0.0,
    }
    low_soc_reward, low_penalties = build_network_reward(
        config,
        battery_info={"soc": 0.10, "soh": 1.0, "soc_violation": 0.0, "effective_power": 0.0, "power_loss": 0.0, "r_int_power_factor": 1.0},
        metrics=metrics,
        import_cost=0.0,
    )
    mid_soc_reward, mid_penalties = build_network_reward(
        config,
        battery_info={"soc": 0.55, "soh": 1.0, "soc_violation": 0.0, "effective_power": 0.0, "power_loss": 0.0, "r_int_power_factor": 1.0},
        metrics=metrics,
        import_cost=0.0,
    )
    assert mid_penalties["soc_edge_penalty"] < low_penalties["soc_edge_penalty"]
    assert mid_penalties["soc_center_penalty"] < low_penalties["soc_center_penalty"]
    assert mid_soc_reward > low_soc_reward
