from __future__ import annotations

import math

import pytest

from microgrid_sim.baselines.dispatch import MILPOptimizer, run_milp_baseline, run_rule_based_baseline
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
    assert math.isfinite(result["total_objective_cost"])
    assert math.isfinite(result["cost"][-1])
    assert math.isfinite(result["objective_cost"][-1])


def test_rule_based_baseline_runs_on_current_network_env():
    env = NetworkMicrogridEnv(IEEE33Config(simulation_days=1, battery_model="simple"))

    result = run_rule_based_baseline(env, simulation_days=1)
    expected_steps = steps_per_day(env.config.dt_seconds)

    assert result["name"] == "Rule Based"
    assert len(result["steps"]) == expected_steps
    assert len(result["pv"]) == expected_steps
    assert len(result["load"]) == expected_steps
    assert math.isfinite(result["total_cost"])


def test_milp_baseline_supports_soft_grid_limit_on_cigre():
    env = NetworkMicrogridEnv(CIGREEuropeanLVConfig(simulation_days=1, battery_model="simple", reward_profile="paper_balanced"))

    result = run_milp_baseline(env, simulation_days=1, chunk_days=0, efficiency_model="realistic")

    assert result["name"] == "MILP Oracle"
    assert len(result["steps"]) == steps_per_day(env.config.dt_seconds)
    assert math.isfinite(result["total_cost"])
    assert math.isfinite(result["total_objective_cost"])


def test_milp_baseline_tracks_terminal_soc_objective_cost():
    env = NetworkMicrogridEnv(IEEE33Config(simulation_days=1, battery_model="simple", reward_profile="paper_balanced"))

    result = run_milp_baseline(env, simulation_days=1, chunk_days=0, efficiency_model="realistic")

    assert len(result["objective_cost"]) == len(result["cost"])
    assert result["total_objective_cost"] >= result["total_cost"]


def test_milp_optimizer_can_return_solution_details():
    env = NetworkMicrogridEnv(IEEE33Config(simulation_days=1, battery_model="simple", reward_profile="paper_balanced"))
    try:
        env.reset(seed=42)
        total_steps = steps_per_day(env.config.dt_seconds)
        from microgrid_sim.baselines.dispatch import _extract_network_env_forecasts

        forecasts = _extract_network_env_forecasts(env, total_steps)
        optimizer = MILPOptimizer(
            env.battery.params,
            horizon=total_steps,
            efficiency_model="realistic",
            feed_in_tariff=float(env.config.feed_in_tariff),
            grid_import_max=float(env.config.grid_import_max),
            grid_export_max=float(env.config.grid_export_max),
            grid_limit_violation_penalty_per_kwh=float(env.config.grid_limit_violation_penalty_per_kwh),
            battery_throughput_penalty_per_kwh=float(env.config.battery_throughput_penalty_per_kwh),
            terminal_soc_target=getattr(env.config, "terminal_soc_target", None),
            terminal_soc_tolerance=float(getattr(env.config, "terminal_soc_tolerance", 0.0)),
            terminal_soc_penalty_per_kwh=float(getattr(env.config, "terminal_soc_penalty_per_kwh", 0.0)),
            dt_seconds=float(env.config.dt_seconds),
        )
        schedule, objective_value, details = optimizer.solve(
            forecasts["pv_w"][:total_steps],
            forecasts["load_w"][:total_steps],
            forecasts["price"][:total_steps],
            env.battery.soc,
            other_forecast=forecasts["other_w"][:total_steps],
            month_index=forecasts["month_index"][:total_steps],
            initial_monthly_peak_billed_kw={},
            return_details=True,
        )
    finally:
        env.close()

    assert len(schedule) == total_steps
    assert math.isfinite(objective_value)
    assert len(details["soc_schedule"]) == total_steps
    assert len(details["import_schedule_w"]) == total_steps
    assert len(details["export_schedule_w"]) == total_steps
    assert math.isfinite(float(details["energy_import_cost"]))
    assert math.isfinite(float(details["terminal_soc_penalty_cost"]))
    assert float(details["objective_value"]) == pytest.approx(float(objective_value))
