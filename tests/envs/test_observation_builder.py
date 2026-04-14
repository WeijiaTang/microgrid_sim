from __future__ import annotations

import numpy as np

from microgrid_sim.cases import IEEE33Config
from microgrid_sim.envs.observation_builder import OBSERVATION_SIZE, build_network_observation
from microgrid_sim.models.battery import SimpleBattery


def test_network_observation_includes_soc_target_and_energy_room_features():
    config = IEEE33Config(simulation_days=1, reward_profile="paper_balanced")
    battery = SimpleBattery(config.battery_params)
    battery.reset(soc=0.40)
    metrics = {
        "min_bus_voltage_pu": 0.98,
        "max_bus_voltage_pu": 1.01,
        "max_line_loading_pct": 45.0,
        "max_transformer_loading_pct": 20.0,
        "slack_active_power_mw": 0.5,
    }

    obs = build_network_observation(
        config,
        battery,
        load_w=1_000_000.0,
        pv_w=200_000.0,
        price=0.8,
        step=0,
        total_steps=96,
        metrics=metrics,
        battery_info={"actual_power": 0.0, "effective_power": 0.0, "current": 0.0, "power_loss": 0.0, "p_max": 500_000.0, "r_int": 0.02},
        timestamp="2024-01-01 00:00:00",
    )

    assert obs.shape == (OBSERVATION_SIZE,)
    assert OBSERVATION_SIZE == 30
    assert np.isclose(obs[19], battery.soc - config.terminal_soc_target)
    assert np.isclose(obs[20], config.battery_params.soc_max - battery.soc)
    assert np.isclose(obs[21], battery.soc - config.battery_params.soc_min)
    assert np.isclose(obs[22], 0.0, atol=1e-6)
    assert np.isclose(obs[23], 1.0, atol=1e-6)
    assert np.isclose(obs[24], -1.0)
    assert np.isclose(obs[25], 1.0)
    assert np.isclose(obs[26], 1.0)
    assert np.isclose(obs[27], 1.0)
    assert np.isclose(obs[28], 1.0)
    assert np.isclose(obs[29], 1.0)


def test_network_observation_falls_back_to_initial_soc_target_when_terminal_target_is_unset():
    config = IEEE33Config(simulation_days=1, terminal_soc_target=None)
    battery = SimpleBattery(config.battery_params)
    battery.reset(soc=0.65)

    obs = build_network_observation(
        config,
        battery,
        load_w=0.0,
        pv_w=0.0,
        price=0.0,
        step=0,
        total_steps=96,
        metrics={},
        battery_info={},
    )

    assert np.isclose(obs[19], battery.soc - config.battery_params.soc_init)
