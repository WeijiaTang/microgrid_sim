from __future__ import annotations

import math

import pytest

from microgrid_sim.baselines.dispatch import run_milp_baseline, run_rule_based_baseline
from microgrid_sim.cases import CIGREEuropeanLVConfig, IEEE33Config
from microgrid_sim.envs.network_microgrid import NetworkMicrogridEnv
from microgrid_sim.time_utils import steps_per_day


@pytest.mark.parametrize("config_cls", [CIGREEuropeanLVConfig, IEEE33Config])
def test_milp_baseline_runs_on_current_network_env(config_cls):
    env = NetworkMicrogridEnv(config_cls(simulation_days=1, battery_model="simple"))

    result = run_milp_baseline(env, simulation_days=1, horizon=24, efficiency_model="simple")
    expected_steps = steps_per_day(env.config.dt_seconds)

    assert result["name"] == "MILP Oracle"
    assert len(result["steps"]) == expected_steps
    assert len(result["soc"]) == expected_steps
    assert len(result["battery_power"]) == expected_steps
    assert math.isfinite(result["total_cost"])
    assert math.isfinite(result["cost"][-1])


def test_rule_based_baseline_runs_on_current_network_env():
    env = NetworkMicrogridEnv(IEEE33Config(simulation_days=1, battery_model="simple"))

    result = run_rule_based_baseline(env, simulation_days=1)
    expected_steps = steps_per_day(env.config.dt_seconds)

    assert result["name"] == "Rule Based"
    assert len(result["steps"]) == expected_steps
    assert len(result["pv"]) == expected_steps
    assert len(result["load"]) == expected_steps
    assert math.isfinite(result["total_cost"])
