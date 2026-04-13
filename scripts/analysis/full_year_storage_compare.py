#!/usr/bin/env python3
"""Full-year storage comparison across no-battery, MILP Oracle, and rolling GA baselines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from genetic_dispatch_baseline import optimize_schedule_rolling  # noqa: E402
from microgrid_sim.baselines.dispatch import MILPOptimizer, _battery_command_to_action, _extract_network_env_forecasts  # noqa: E402
from microgrid_sim.cases import CIGREEuropeanLVConfig, IEEE33Config  # noqa: E402
from microgrid_sim.envs.network_microgrid import NetworkMicrogridEnv  # noqa: E402
from microgrid_sim.time_utils import simulation_steps, steps_per_day  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare full-year storage performance on IEEE33 and CIGRE.")
    parser.add_argument("--cases", type=str, default="ieee33,cigre", help="Comma-separated case keys: ieee33, cigre")
    parser.add_argument("--regimes", type=str, default="network_stress", help="Comma-separated regimes")
    parser.add_argument("--reward-profile", type=str, default="paper_balanced", help="Reward profile")
    parser.add_argument("--battery-model", type=str, default="simple", help="Battery model for storage-enabled baselines")
    parser.add_argument("--days", type=int, default=365, help="Simulation days")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--milp-efficiency-model", type=str, default="realistic", help="MILP efficiency model")
    parser.add_argument(
        "--milp-chunk-days",
        type=int,
        default=0,
        help="MILP chunk days. Use 0 for full-year perfect-foresight, >0 for rolling MILP.",
    )
    parser.add_argument("--ga-population-size", type=int, default=6, help="GA population size per rolling window")
    parser.add_argument("--ga-generations", type=int, default=2, help="GA generations per rolling window")
    parser.add_argument("--ga-elite-count", type=int, default=2, help="GA elite count per rolling window")
    parser.add_argument("--ga-mutation-scale", type=float, default=0.12, help="GA mutation scale")
    parser.add_argument("--ga-crossover-rate", type=float, default=0.5, help="GA crossover rate")
    parser.add_argument("--ga-rolling-window-days", type=int, default=7, help="GA planning window in days")
    parser.add_argument("--ga-rolling-stride-days", type=int, default=1, help="GA execution stride in days")
    parser.add_argument("--skip-milp", action="store_true", help="Skip the MILP Oracle baseline")
    parser.add_argument("--skip-ga", action="store_true", help="Skip the rolling GA baseline")
    parser.add_argument("--voltage-penalty", type=float, default=2000.0, help="Analysis-only voltage penalty scale")
    parser.add_argument("--loading-penalty", type=float, default=250.0, help="Analysis-only loading penalty scale")
    parser.add_argument("--soc-penalty", type=float, default=1000.0, help="Analysis-only SOC violation penalty scale")
    parser.add_argument(
        "--save-trajectories",
        action="store_true",
        help="Save per-step trajectories for each baseline and case.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/full_year_storage_compare"),
        help="Output directory",
    )
    return parser


def _parse_csv_arg(raw: str) -> list[str]:
    return [item.strip().lower() for item in str(raw).split(",") if item.strip()]


def build_config(case_key: str, battery_model: str, days: int, seed: int, regime: str, reward_profile: str):
    if case_key == "ieee33":
        return IEEE33Config(
            simulation_days=int(days),
            seed=int(seed),
            battery_model=str(battery_model),
            regime=str(regime),
            reward_profile=str(reward_profile),
        )
    if case_key == "cigre":
        return CIGREEuropeanLVConfig(
            simulation_days=int(days),
            seed=int(seed),
            battery_model=str(battery_model),
            regime=str(regime),
            reward_profile=str(reward_profile),
        )
    raise ValueError(f"Unsupported case '{case_key}'")


def _dwell_fraction(mask: pd.Series | np.ndarray | list[bool]) -> float:
    series = pd.Series(mask, dtype=float)
    if series.empty:
        return 0.0
    return float(series.mean())


def _boundary_note(final_soc: float, lower_dwell: float, upper_dwell: float, soc_min: float, soc_max: float) -> str:
    if lower_dwell >= 0.80:
        return "lower-bound attractor"
    if upper_dwell >= 0.80:
        return "upper-bound attractor"
    if abs(float(final_soc) - float(soc_min)) <= 1e-6:
        return "ends at lower SOC boundary"
    if abs(float(final_soc) - float(soc_max)) <= 1e-6:
        return "ends at upper SOC boundary"
    return "no severe boundary collapse"


def _analysis_penalty_value(
    trajectory: pd.DataFrame,
    *,
    voltage_penalty: float,
    loading_penalty: float,
    soc_penalty: float,
) -> float:
    if trajectory.empty:
        return 0.0
    undervoltage = trajectory["undervoltage"].astype(float)
    overvoltage = trajectory["overvoltage"].astype(float)
    line_overload = trajectory["line_overload_pct"].astype(float)
    trafo_overload = trajectory["transformer_overload_pct"].astype(float)
    soc_violation = trajectory["soc_violation"].astype(float)
    return float(
        (float(voltage_penalty) * (undervoltage + overvoltage)).sum()
        + (float(loading_penalty) * ((line_overload + trafo_overload) / 100.0)).sum()
        + (float(soc_penalty) * soc_violation).sum()
    )


def _summarize_trajectory(
    *,
    trajectory: pd.DataFrame,
    case_key: str,
    regime: str,
    baseline: str,
    battery_model: str,
    days: int,
    seed: int,
    reward_profile: str,
    profile_start_timestamp: str,
    profile_end_timestamp: str,
    soc_min: float,
    soc_max: float,
    voltage_penalty: float,
    loading_penalty: float,
    soc_penalty: float,
    extra_fields: dict[str, float | int | str] | None = None,
) -> dict[str, float | int | str]:
    extra_fields = dict(extra_fields or {})
    final_cost = float(trajectory["cumulative_cost"].iloc[-1]) if not trajectory.empty else 0.0
    final_objective_cost = float(trajectory["cumulative_objective_cost"].iloc[-1]) if not trajectory.empty else final_cost
    final_soc = float(trajectory["soc"].iloc[-1]) if not trajectory.empty else 0.0
    power_flow_failed = (
        trajectory["power_flow_failed"].astype(int)
        if (not trajectory.empty and "power_flow_failed" in trajectory.columns)
        else pd.Series(np.zeros(len(trajectory), dtype=int))
    )
    lower_hit = trajectory["soc"].astype(float) <= float(soc_min) + 1e-6 if not trajectory.empty else pd.Series(dtype=bool)
    upper_hit = trajectory["soc"].astype(float) >= float(soc_max) - 1e-6 if not trajectory.empty else pd.Series(dtype=bool)
    lower_dwell = _dwell_fraction(lower_hit)
    upper_dwell = _dwell_fraction(upper_hit)
    analysis_penalty = _analysis_penalty_value(
        trajectory,
        voltage_penalty=float(voltage_penalty),
        loading_penalty=float(loading_penalty),
        soc_penalty=float(soc_penalty),
    )
    row = {
        "case": str(case_key),
        "regime": str(regime),
        "baseline": str(baseline),
        "battery_model": str(battery_model),
        "days": int(days),
        "seed": int(seed),
        "reward_profile": str(reward_profile),
        "profile_start_timestamp": str(profile_start_timestamp),
        "profile_end_timestamp": str(profile_end_timestamp),
        "steps": int(len(trajectory)),
        "final_cumulative_cost": final_cost,
        "final_cumulative_objective_cost": final_objective_cost,
        "analysis_penalty_value": float(analysis_penalty),
        "analysis_objective_value": float(final_objective_cost + analysis_penalty),
        "total_reward": float(trajectory["reward"].sum()) if not trajectory.empty else 0.0,
        "final_soc": final_soc,
        "total_terminal_soc_penalty": float(trajectory["terminal_soc_penalty"].sum()) if not trajectory.empty else 0.0,
        "min_voltage_worst": float(trajectory["min_bus_voltage_pu"].min()) if not trajectory.empty else 1.0,
        "max_line_loading_peak": float(trajectory["max_line_loading_pct"].max()) if not trajectory.empty else 0.0,
        "mean_grid_import_mw": float(trajectory["grid_import_mw"].mean()) if not trajectory.empty else 0.0,
        "undervoltage_total": float(trajectory["undervoltage"].sum()) if not trajectory.empty else 0.0,
        "overvoltage_total": float(trajectory["overvoltage"].sum()) if not trajectory.empty else 0.0,
        "line_overload_total": float(trajectory["line_overload_pct"].sum()) if not trajectory.empty else 0.0,
        "transformer_overload_total": float(trajectory["transformer_overload_pct"].sum()) if not trajectory.empty else 0.0,
        "soc_violation_total": float(trajectory["soc_violation"].sum()) if not trajectory.empty else 0.0,
        "soc_lower_dwell_fraction": float(lower_dwell),
        "soc_upper_dwell_fraction": float(upper_dwell),
        "power_flow_failure_steps": int(power_flow_failed.sum()) if not trajectory.empty else 0,
        "soc_boundary_note": _boundary_note(final_soc, lower_dwell, upper_dwell, soc_min, soc_max),
    }
    row.update(extra_fields)
    return row


def _reset_env(env: NetworkMicrogridEnv, seed: int) -> tuple[str, str]:
    env.reset(seed=int(seed))
    timestamps = pd.DatetimeIndex(env._profiles.timestamps)
    if timestamps.empty:
        return "", ""
    return str(timestamps[0]), str(timestamps[-1])


def _rollout_env(
    env: NetworkMicrogridEnv,
    *,
    total_steps: int,
    action_fn,
    post_step_fn=None,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for step in range(int(total_steps)):
        idx = min(env.current_step, env.total_steps - 1)
        power_w = float(action_fn(idx))
        action = _battery_command_to_action(env, power_w)
        _, reward, terminated, truncated, info = env.step(action)
        if post_step_fn is not None:
            post_step_fn(step, info)
        rows.append(
            {
                "step": int(step),
                "timestamp": str(info.get("timestamp", "")),
                "requested_power_w": power_w,
                "applied_action": float(np.asarray(action, dtype=float).reshape(-1)[0]),
                "reward": float(reward),
                "soc": float(info.get("soc", 0.0)),
                "battery_power_w": float(info.get("battery_power_w", 0.0)),
                "battery_power_mw": float(info.get("battery_power_mw", 0.0)),
                "grid_import_mw": float(info.get("grid_import_mw", 0.0)),
                "grid_export_mw": float(info.get("grid_export_mw", 0.0)),
                "cumulative_cost": float(info.get("cumulative_cost", 0.0)),
                "cumulative_objective_cost": float(info.get("cumulative_objective_cost", info.get("cumulative_cost", 0.0))),
                "min_bus_voltage_pu": float(info.get("min_bus_voltage_pu", 1.0)),
                "max_line_loading_pct": float(info.get("max_line_loading_pct", 0.0)),
                "undervoltage": float(info.get("undervoltage", 0.0)),
                "overvoltage": float(info.get("overvoltage", 0.0)),
                "line_overload_pct": float(info.get("line_overload_pct", 0.0)),
                "transformer_overload_pct": float(info.get("transformer_overload_pct", 0.0)),
                "soc_violation": float(info.get("soc_violation", 0.0)),
                "terminal_soc_penalty": float(info.get("terminal_soc_penalty", 0.0)),
                "power_flow_failed": int(bool(info.get("power_flow_failed", False))),
            }
        )
        if terminated or truncated:
            break
    return pd.DataFrame(rows)


def run_zero_baseline(config, *, seed: int) -> tuple[pd.DataFrame, dict[str, str]]:
    env = NetworkMicrogridEnv(config)
    try:
        start_ts, end_ts = _reset_env(env, seed=int(seed))
        total_steps = simulation_steps(config.simulation_days, config.dt_seconds)
        trajectory = _rollout_env(env, total_steps=total_steps, action_fn=lambda _: 0.0)
        return trajectory, {"profile_start_timestamp": start_ts, "profile_end_timestamp": end_ts}
    finally:
        env.close()


def run_milp_with_trajectory(
    config,
    *,
    seed: int,
    simulation_days: int,
    efficiency_model: str,
    chunk_days: int,
) -> tuple[pd.DataFrame, dict[str, str | int]]:
    env = NetworkMicrogridEnv(config)
    try:
        start_ts, end_ts = _reset_env(env, seed=int(seed))
        total_steps = simulation_steps(simulation_days, env.config.dt_seconds)
        env_steps_per_day = steps_per_day(env.config.dt_seconds)
        if int(chunk_days) <= 0:
            horizon_steps = total_steps
        else:
            horizon_steps = max(int(chunk_days), 1) * env_steps_per_day
        forecasts = _extract_network_env_forecasts(env, total_steps)
        optimizer = MILPOptimizer(
            env.battery.params,
            horizon=int(horizon_steps),
            efficiency_model=str(efficiency_model),
            feed_in_tariff=float(getattr(env.config, "feed_in_tariff", 0.0)),
            grid_import_max=float(getattr(env.config, "grid_import_max", float("inf"))),
            grid_export_max=float(getattr(env.config, "grid_export_max", float("inf"))),
            grid_limit_violation_penalty_per_kwh=float(getattr(env.config, "grid_limit_violation_penalty_per_kwh", 0.0)),
            peak_import_penalty_per_kw=float(getattr(env.config, "peak_import_penalty_per_kw", 0.0)),
            peak_import_penalty_threshold_w=float(getattr(env.config, "peak_import_penalty_threshold_w", float("inf"))),
            monthly_demand_charge_per_kw=float(getattr(env.config, "monthly_demand_charge_per_kw", 0.0)),
            monthly_demand_charge_threshold_w=float(getattr(env.config, "monthly_demand_charge_threshold_w", float("inf"))),
            battery_throughput_penalty_per_kwh=float(getattr(env.config, "battery_throughput_penalty_per_kwh", 0.0)),
            terminal_soc_target=getattr(env.config, "terminal_soc_target", None),
            terminal_soc_tolerance=float(getattr(env.config, "terminal_soc_tolerance", 0.0)),
            terminal_soc_penalty_per_kwh=float(getattr(env.config, "terminal_soc_penalty_per_kwh", 0.0)),
            dt_seconds=float(getattr(env.config, "dt_seconds", 3600.0)),
        )
        realized_monthly_peak_billed_kw: dict[int, float] = {}
        if int(horizon_steps) >= total_steps:
            schedule, _ = optimizer.solve(
                forecasts["pv_w"][:total_steps],
                forecasts["load_w"][:total_steps],
                forecasts["price"][:total_steps],
                env.battery.soc,
                other_forecast=forecasts["other_w"][:total_steps],
                month_index=forecasts["month_index"][:total_steps],
                initial_monthly_peak_billed_kw={},
            )

            def action_fn(step: int) -> float:
                return float(schedule[step])

            solve_mode = "full_horizon"
        else:

            def action_fn(step: int) -> float:
                lookahead = min(int(horizon_steps), total_steps - step)
                schedule, _ = optimizer.solve(
                    forecasts["pv_w"][step : step + lookahead],
                    forecasts["load_w"][step : step + lookahead],
                    forecasts["price"][step : step + lookahead],
                    env.battery.soc,
                    other_forecast=forecasts["other_w"][step : step + lookahead],
                    month_index=forecasts["month_index"][step : step + lookahead],
                    initial_monthly_peak_billed_kw=realized_monthly_peak_billed_kw,
                )
                return float(schedule[0])

            solve_mode = "rolling_horizon"

        def post_step_fn(step: int, info: dict) -> None:
            month_id = int(forecasts["month_index"][step])
            import_kw = max(float(info.get("grid_import_mw", 0.0)) * 1000.0, 0.0)
            realized_monthly_peak_billed_kw[month_id] = max(realized_monthly_peak_billed_kw.get(month_id, 0.0), import_kw)

        trajectory = _rollout_env(env, total_steps=total_steps, action_fn=action_fn, post_step_fn=post_step_fn)
        return trajectory, {
            "profile_start_timestamp": start_ts,
            "profile_end_timestamp": end_ts,
            "milp_horizon_steps": int(horizon_steps),
            "milp_solve_mode": str(solve_mode),
        }
    finally:
        env.close()


def _profile_window_metadata(config, *, seed: int) -> tuple[str, str]:
    env = NetworkMicrogridEnv(config)
    try:
        return _reset_env(env, seed=int(seed))
    finally:
        env.close()


def build_protocol_summaries(detail_df: pd.DataFrame, *, protocol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_rows: list[dict[str, float | int | str]] = []
    objective_rows: list[dict[str, float | int | str]] = []
    for (case_key, regime), group in detail_df.groupby(["case", "regime"], dropna=False):
        by_baseline = {str(row["baseline"]): row for _, row in group.iterrows()}
        none_row = by_baseline.get("none")
        oracle_row = by_baseline.get("milp_oracle")
        if none_row is None or oracle_row is None:
            continue
        for baseline_name, algorithm_label in (("milp_oracle", "MILP Oracle"), ("ga_dispatch_rolling", "GA rolling")):
            algorithm_row = by_baseline.get(baseline_name)
            if algorithm_row is None:
                continue
            raw_none = float(none_row["final_cumulative_cost"])
            raw_oracle = float(oracle_row["final_cumulative_cost"])
            raw_algorithm = float(algorithm_row["final_cumulative_cost"])
            raw_oracle_savings = raw_none - raw_oracle
            raw_savings = raw_none - raw_algorithm
            raw_rows.append(
                {
                    "protocol": str(protocol),
                    "case": str(case_key),
                    "regime": str(regime),
                    "algorithm": str(algorithm_label),
                    "metric": "final_cumulative_cost",
                    "none_cost": raw_none,
                    "oracle_cost": raw_oracle,
                    "algorithm_cost": raw_algorithm,
                    "savings_vs_none": float(raw_savings),
                    "oracle_savings": float(raw_oracle_savings),
                    "recovery_fraction": float(raw_savings / raw_oracle_savings) if abs(raw_oracle_savings) > 1e-9 else np.nan,
                    "gap_to_oracle": float(raw_algorithm - raw_oracle),
                    "final_soc": float(algorithm_row["final_soc"]),
                    "soc_boundary_note": str(algorithm_row["soc_boundary_note"]),
                }
            )

            obj_none = float(none_row["final_cumulative_objective_cost"])
            obj_oracle = float(oracle_row["final_cumulative_objective_cost"])
            obj_algorithm = float(algorithm_row["final_cumulative_objective_cost"])
            obj_oracle_savings = obj_none - obj_oracle
            obj_savings = obj_none - obj_algorithm
            objective_rows.append(
                {
                    "protocol": str(protocol),
                    "case": str(case_key),
                    "regime": str(regime),
                    "algorithm": str(algorithm_label),
                    "metric": "final_cumulative_objective_cost",
                    "none_cost": obj_none,
                    "oracle_cost": obj_oracle,
                    "algorithm_cost": obj_algorithm,
                    "savings_vs_none": float(obj_savings),
                    "oracle_savings": float(obj_oracle_savings),
                    "recovery_fraction": float(obj_savings / obj_oracle_savings) if abs(obj_oracle_savings) > 1e-9 else np.nan,
                    "gap_to_oracle": float(obj_algorithm - obj_oracle),
                    "final_soc": float(algorithm_row["final_soc"]),
                    "soc_boundary_note": str(algorithm_row["soc_boundary_note"]),
                }
            )
    return pd.DataFrame(raw_rows), pd.DataFrame(objective_rows)


def main() -> int:
    args = build_parser().parse_args()
    case_keys = _parse_csv_arg(args.cases)
    regimes = _parse_csv_arg(args.regimes)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectories_dir = output_dir / "trajectories"
    if bool(args.save_trajectories):
        trajectories_dir.mkdir(parents=True, exist_ok=True)

    detail_rows: list[dict[str, float | int | str]] = []
    for case_key in case_keys:
        for regime in regimes:
            print(f"[full-year] case={case_key} regime={regime} days={args.days}")

            none_config = build_config(case_key, "none", args.days, args.seed, regime, args.reward_profile)
            none_traj, none_meta = run_zero_baseline(none_config, seed=int(args.seed))
            none_row = _summarize_trajectory(
                trajectory=none_traj,
                case_key=case_key,
                regime=regime,
                baseline="none",
                battery_model="none",
                days=int(args.days),
                seed=int(args.seed),
                reward_profile=str(args.reward_profile),
                profile_start_timestamp=str(none_meta["profile_start_timestamp"]),
                profile_end_timestamp=str(none_meta["profile_end_timestamp"]),
                soc_min=float(none_config.battery_params.soc_min),
                soc_max=float(none_config.battery_params.soc_max),
                voltage_penalty=float(args.voltage_penalty),
                loading_penalty=float(args.loading_penalty),
                soc_penalty=float(args.soc_penalty),
            )
            detail_rows.append(none_row)
            if bool(args.save_trajectories):
                none_traj.to_csv(trajectories_dir / f"{case_key}_{regime}_none_trajectory.csv", index=False)

            storage_config = build_config(case_key, args.battery_model, args.days, args.seed, regime, args.reward_profile)
            if not bool(args.skip_milp):
                milp_traj, milp_meta = run_milp_with_trajectory(
                    storage_config,
                    seed=int(args.seed),
                    simulation_days=int(args.days),
                    efficiency_model=str(args.milp_efficiency_model),
                    chunk_days=int(args.milp_chunk_days),
                )
                milp_row = _summarize_trajectory(
                    trajectory=milp_traj,
                    case_key=case_key,
                    regime=regime,
                    baseline="milp_oracle",
                    battery_model=str(args.battery_model),
                    days=int(args.days),
                    seed=int(args.seed),
                    reward_profile=str(args.reward_profile),
                    profile_start_timestamp=str(milp_meta["profile_start_timestamp"]),
                    profile_end_timestamp=str(milp_meta["profile_end_timestamp"]),
                    soc_min=float(storage_config.battery_params.soc_min),
                    soc_max=float(storage_config.battery_params.soc_max),
                    voltage_penalty=float(args.voltage_penalty),
                    loading_penalty=float(args.loading_penalty),
                    soc_penalty=float(args.soc_penalty),
                    extra_fields={
                        "milp_efficiency_model": str(args.milp_efficiency_model),
                        "milp_chunk_days": int(args.milp_chunk_days),
                        "milp_horizon_steps": int(milp_meta["milp_horizon_steps"]),
                        "milp_solve_mode": str(milp_meta["milp_solve_mode"]),
                    },
                )
                detail_rows.append(milp_row)
                if bool(args.save_trajectories):
                    milp_traj.to_csv(trajectories_dir / f"{case_key}_{regime}_milp_oracle_trajectory.csv", index=False)

            if not bool(args.skip_ga):
                ga_actions, ga_summary, ga_traj = optimize_schedule_rolling(
                    case_key=case_key,
                    battery_model=str(args.battery_model),
                    regime=regime,
                    days=int(args.days),
                    seed=int(args.seed),
                    population_size=int(args.ga_population_size),
                    generations=int(args.ga_generations),
                    elite_count=int(args.ga_elite_count),
                    mutation_scale=float(args.ga_mutation_scale),
                    crossover_rate=float(args.ga_crossover_rate),
                    reward_profile=str(args.reward_profile),
                    voltage_penalty=float(args.voltage_penalty),
                    loading_penalty=float(args.loading_penalty),
                    soc_penalty=float(args.soc_penalty),
                    rolling_window_days=int(args.ga_rolling_window_days),
                    rolling_stride_days=int(args.ga_rolling_stride_days),
                )
                ga_start_ts, ga_end_ts = _profile_window_metadata(storage_config, seed=int(args.seed))
                ga_row = _summarize_trajectory(
                    trajectory=ga_traj,
                    case_key=case_key,
                    regime=regime,
                    baseline="ga_dispatch_rolling",
                    battery_model=str(args.battery_model),
                    days=int(args.days),
                    seed=int(args.seed),
                    reward_profile=str(args.reward_profile),
                    profile_start_timestamp=str(ga_start_ts),
                    profile_end_timestamp=str(ga_end_ts),
                    soc_min=float(storage_config.battery_params.soc_min),
                    soc_max=float(storage_config.battery_params.soc_max),
                    voltage_penalty=float(args.voltage_penalty),
                    loading_penalty=float(args.loading_penalty),
                    soc_penalty=float(args.soc_penalty),
                    extra_fields={
                        "objective_value": float(ga_summary.get("objective_value", 0.0)),
                        "penalty_value": float(ga_summary.get("penalty_value", 0.0)),
                        "ga_population_size": int(args.ga_population_size),
                        "ga_generations": int(args.ga_generations),
                        "ga_elite_count": int(args.ga_elite_count),
                        "ga_mutation_scale": float(args.ga_mutation_scale),
                        "ga_crossover_rate": float(args.ga_crossover_rate),
                        "ga_rolling_window_days": int(args.ga_rolling_window_days),
                        "ga_rolling_stride_days": int(args.ga_rolling_stride_days),
                        "ga_rolling_window_count": int(ga_summary.get("rolling_window_count", 0)),
                        "optimized_action_count": int(len(ga_actions)),
                    },
                )
                detail_rows.append(ga_row)
                if bool(args.save_trajectories):
                    ga_traj.to_csv(trajectories_dir / f"{case_key}_{regime}_ga_dispatch_rolling_trajectory.csv", index=False)
                    rolling_plans = ga_traj.attrs.get("rolling_plans")
                    if rolling_plans:
                        pd.DataFrame(rolling_plans).to_csv(
                            trajectories_dir / f"{case_key}_{regime}_ga_dispatch_rolling_windows.csv",
                            index=False,
                        )

    detail_df = pd.DataFrame(detail_rows)
    protocol = f"full_year_{int(args.days)}d_start_profile"
    raw_df, objective_df = build_protocol_summaries(detail_df, protocol=protocol)

    detail_csv = output_dir / "detail.csv"
    raw_csv = output_dir / "raw_cost_protocol_summary.csv"
    objective_csv = output_dir / "objective_cost_protocol_summary.csv"
    detail_json = output_dir / "detail.json"
    detail_df.to_csv(detail_csv, index=False)
    raw_df.to_csv(raw_csv, index=False)
    objective_df.to_csv(objective_csv, index=False)
    detail_json.write_text(json.dumps(detail_rows, indent=2), encoding="utf-8")

    print("\n=== Full-Year Detail ===")
    print(
        detail_df[
            [
                "case",
                "regime",
                "baseline",
                "battery_model",
                "final_cumulative_cost",
                "final_cumulative_objective_cost",
                "analysis_objective_value",
                "final_soc",
                "soc_lower_dwell_fraction",
                "soc_upper_dwell_fraction",
            ]
        ].round(6).to_string(index=False)
    )
    print("\n=== Raw-Cost Protocol Summary ===")
    if not raw_df.empty:
        print(raw_df.round(6).to_string(index=False))
    print("\n=== Objective-Cost Protocol Summary ===")
    if not objective_df.empty:
        print(objective_df.round(6).to_string(index=False))
    print(f"\nSaved detail CSV: {detail_csv}")
    print(f"Saved raw-cost protocol summary: {raw_csv}")
    print(f"Saved objective-cost protocol summary: {objective_csv}")
    print(f"Saved detail JSON: {detail_json}")
    if bool(args.save_trajectories):
        print(f"Saved trajectories: {trajectories_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
