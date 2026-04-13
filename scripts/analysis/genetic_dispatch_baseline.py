#!/usr/bin/env python3
"""Genetic-algorithm dispatch baseline for network-first microgrids."""
# Ref: docs/spec/task.md (Task-ID: SPEC-FIDELITY-MISMATCH-001)

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from microgrid_sim.cases import CIGREEuropeanLVConfig, IEEE33Config
from microgrid_sim.data.network_profiles import load_network_profiles
from microgrid_sim.envs.network_microgrid import NetworkMicrogridEnv
from microgrid_sim.models.battery import SimpleBattery
from microgrid_sim.time_utils import hours_to_steps, simulation_steps, steps_per_day


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GA dispatch baseline for none/simple/thevenin battery comparison.")
    parser.add_argument("--cases", type=str, default="ieee33", help="Comma-separated case keys: cigre, ieee33")
    parser.add_argument("--regimes", type=str, default="network_stress", help="Comma-separated regimes")
    parser.add_argument("--battery-models", type=str, default="simple,thevenin", help="Comma-separated battery models")
    parser.add_argument("--reward-profile", type=str, default="network", help="Reward profile: network, paper_aligned, or paper_balanced")
    parser.add_argument("--days", type=int, default=1, help="Simulation days for the deterministic dispatch horizon")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--population-size", type=int, default=32, help="GA population size")
    parser.add_argument("--generations", type=int, default=25, help="Number of GA generations")
    parser.add_argument("--elite-count", type=int, default=4, help="Number of elites kept each generation")
    parser.add_argument("--mutation-scale", type=float, default=0.12, help="Gaussian mutation standard deviation")
    parser.add_argument("--crossover-rate", type=float, default=0.5, help="Blend crossover ratio")
    parser.add_argument(
        "--rolling-window-days",
        type=int,
        default=0,
        help="Optional rolling-horizon planning window in days. When > 0, GA replans over this window and executes only the stride segment.",
    )
    parser.add_argument(
        "--rolling-stride-days",
        type=int,
        default=0,
        help="Optional rolling-horizon execution stride in days. Defaults to --rolling-window-days when unset.",
    )
    parser.add_argument("--voltage-penalty", type=float, default=2_000.0, help="Penalty multiplier for undervoltage/overvoltage")
    parser.add_argument("--loading-penalty", type=float, default=250.0, help="Penalty multiplier for line/transformer overload")
    parser.add_argument("--soc-penalty", type=float, default=1_000.0, help="Penalty multiplier for SOC violation")
    parser.add_argument("--output-dir", type=str, default="results/genetic_dispatch_baseline", help="Output directory")
    return parser


def _parse_csv_arg(raw: str) -> list[str]:
    return [item.strip().lower() for item in str(raw).split(",") if item.strip()]


def build_config(
    case_key: str,
    battery_model: str,
    days: int,
    seed: int,
    regime: str,
    reward_profile: str,
    *,
    episode_start_hour: int = 0,
):
    if case_key == "ieee33":
        return IEEE33Config(
            simulation_days=days,
            seed=seed,
            battery_model=battery_model,
            regime=regime,
            reward_profile=reward_profile,
            episode_start_hour=int(episode_start_hour),
        )
    if case_key == "cigre":
        return CIGREEuropeanLVConfig(
            simulation_days=days,
            seed=seed,
            battery_model=battery_model,
            regime=regime,
            reward_profile=reward_profile,
            episode_start_hour=int(episode_start_hour),
        )
    raise ValueError(f"Unsupported case '{case_key}'")


def build_env_from_config(config) -> NetworkMicrogridEnv:
    return NetworkMicrogridEnv(config)


def build_env(
    case_key: str,
    battery_model: str,
    days: int,
    seed: int,
    regime: str,
    reward_profile: str,
    *,
    episode_start_hour: int = 0,
) -> NetworkMicrogridEnv:
    return build_env_from_config(
        build_config(
            case_key=case_key,
            battery_model=battery_model,
            days=days,
            seed=seed,
            regime=regime,
            reward_profile=reward_profile,
            episode_start_hour=episode_start_hour,
        )
    )


def _initialize_env_state(
    env: NetworkMicrogridEnv,
    *,
    seed: int,
    initial_soc: float | None = None,
    initial_temperature_c: float | None = None,
) -> None:
    env.reset(seed=seed)
    if initial_soc is not None:
        env.battery.reset(soc=float(initial_soc))
    if initial_temperature_c is not None and hasattr(env.battery, "temperature_c"):
        env.battery.temperature_c = float(initial_temperature_c)


def _rollout_actions(
    env: NetworkMicrogridEnv,
    *,
    actions: np.ndarray,
    voltage_penalty: float,
    loading_penalty: float,
    soc_penalty: float,
) -> tuple[float, dict[str, float], pd.DataFrame]:
    rows: list[dict[str, float | int | str]] = []
    total_penalty = 0.0
    for raw_action in np.asarray(actions, dtype=float).reshape(-1):
        action = float(np.clip(raw_action, -1.0, 1.0))
        _, reward, terminated, truncated, info = env.step(np.array([action], dtype=np.float32))
        undervoltage = float(info.get("undervoltage", 0.0))
        overvoltage = float(info.get("overvoltage", 0.0))
        line_overload = float(info.get("line_overload_pct", 0.0))
        trafo_overload = float(info.get("transformer_overload_pct", 0.0))
        soc_violation = float(info.get("soc_violation", 0.0))
        step_penalty = (
            voltage_penalty * (undervoltage + overvoltage)
            + loading_penalty * ((line_overload + trafo_overload) / 100.0)
            + soc_penalty * soc_violation
        )
        total_penalty += step_penalty
        rows.append(
            {
                "step": int(info.get("step", len(rows))),
                "action": action,
                "reward": float(reward),
                "soc": float(info.get("soc", 0.0)),
                "battery_power_mw": float(info.get("battery_power_mw", 0.0)),
                "grid_import_mw": float(info.get("grid_import_mw", 0.0)),
                "grid_export_mw": float(info.get("grid_export_mw", 0.0)),
                "cumulative_cost": float(info.get("cumulative_cost", 0.0)),
                "cumulative_objective_cost": float(info.get("cumulative_objective_cost", info.get("cumulative_cost", 0.0))),
                "min_bus_voltage_pu": float(info.get("min_bus_voltage_pu", 1.0)),
                "max_line_loading_pct": float(info.get("max_line_loading_pct", 0.0)),
                "undervoltage": undervoltage,
                "overvoltage": overvoltage,
                "line_overload_pct": line_overload,
                "transformer_overload_pct": trafo_overload,
                "soc_violation": soc_violation,
                "terminal_soc_penalty": float(info.get("terminal_soc_penalty", 0.0)),
                "penalty_objective_step": float(step_penalty),
            }
        )
        if terminated or truncated:
            break

    trajectory = pd.DataFrame(rows)
    final_cost = float(trajectory["cumulative_cost"].iloc[-1]) if not trajectory.empty else 0.0
    final_objective_cost = (
        float(trajectory["cumulative_objective_cost"].iloc[-1])
        if not trajectory.empty
        else final_cost
    )
    objective = final_objective_cost + float(total_penalty)
    summary = {
        "steps": int(len(trajectory)),
        "objective_value": float(objective),
        "penalty_value": float(total_penalty),
        "total_reward": float(trajectory["reward"].sum()) if not trajectory.empty else 0.0,
        "final_cumulative_cost": final_cost,
        "final_cumulative_objective_cost": final_objective_cost,
        "final_soc": float(trajectory["soc"].iloc[-1]) if not trajectory.empty else float(getattr(env.battery, "soc", 0.0)),
        "total_terminal_soc_penalty": float(trajectory["terminal_soc_penalty"].sum()) if not trajectory.empty else 0.0,
        "min_voltage_worst": float(trajectory["min_bus_voltage_pu"].min()) if not trajectory.empty else 1.0,
        "max_line_loading_peak": float(trajectory["max_line_loading_pct"].max()) if not trajectory.empty else 0.0,
        "mean_grid_import_mw": float(trajectory["grid_import_mw"].mean()) if not trajectory.empty else 0.0,
        "undervoltage_total": float(trajectory["undervoltage"].sum()) if not trajectory.empty else 0.0,
        "overvoltage_total": float(trajectory["overvoltage"].sum()) if not trajectory.empty else 0.0,
        "line_overload_total": float(trajectory["line_overload_pct"].sum()) if not trajectory.empty else 0.0,
        "transformer_overload_total": float(trajectory["transformer_overload_pct"].sum()) if not trajectory.empty else 0.0,
        "soc_violation_total": float(trajectory["soc_violation"].sum()) if not trajectory.empty else 0.0,
    }
    return objective, summary, trajectory


def evaluate_schedule(
    case_key: str,
    battery_model: str,
    regime: str,
    days: int,
    seed: int,
    actions: np.ndarray,
    reward_profile: str,
    voltage_penalty: float,
    loading_penalty: float,
    soc_penalty: float,
    episode_start_hour: int = 0,
    initial_soc: float | None = None,
    initial_temperature_c: float | None = None,
    terminal_penalty_enabled: bool = True,
) -> tuple[float, dict[str, float], pd.DataFrame]:
    config = build_config(
        case_key=case_key,
        battery_model=battery_model,
        days=days,
        seed=seed,
        regime=regime,
        reward_profile=reward_profile,
        episode_start_hour=episode_start_hour,
    )
    if not bool(terminal_penalty_enabled):
        config = replace(config, terminal_soc_penalty_per_kwh=0.0)
    env = build_env_from_config(config)
    try:
        _initialize_env_state(
            env,
            seed=seed,
            initial_soc=initial_soc,
            initial_temperature_c=initial_temperature_c,
        )
        return _rollout_actions(
            env,
            actions=actions,
            voltage_penalty=voltage_penalty,
            loading_penalty=loading_penalty,
            soc_penalty=soc_penalty,
        )
    finally:
        env.close()


def tournament_select(rng: np.random.Generator, scores: np.ndarray, tournament_size: int = 3) -> int:
    participants = rng.integers(0, len(scores), size=max(int(tournament_size), 2))
    return int(participants[np.argmin(scores[participants])])


def _power_to_action(power_w: float, charge_limit_w: float, discharge_limit_w: float) -> float:
    if power_w >= 0.0:
        scale = max(float(discharge_limit_w), 1e-9)
    else:
        scale = max(float(charge_limit_w), 1e-9)
    return float(np.clip(power_w / scale, -1.0, 1.0))


def _heuristic_seed_schedules(
    case_key: str,
    battery_model: str,
    regime: str,
    days: int,
    seed: int,
    reward_profile: str,
    *,
    episode_start_hour: int = 0,
    initial_soc: float | None = None,
) -> list[np.ndarray]:
    config = build_config(
        case_key,
        battery_model,
        days,
        seed,
        regime,
        reward_profile,
        episode_start_hour=episode_start_hour,
    )
    horizon = simulation_steps(days, config.dt_seconds)
    zero_schedule = np.zeros(horizon, dtype=np.float32)
    if str(battery_model).lower() == "none":
        return [zero_schedule]

    profiles = load_network_profiles(config)
    start_step = max(hours_to_steps(getattr(config, "episode_start_hour", 0), config.dt_seconds), 0)
    end_step = start_step + horizon
    load_w = np.asarray(profiles.load_w[start_step:end_step], dtype=float)
    pv_w = np.asarray(profiles.pv_w[start_step:end_step], dtype=float)
    price = np.asarray(profiles.price[start_step:end_step], dtype=float)
    battery = SimpleBattery(config.battery_params)
    battery.reset(soc=initial_soc)
    valley_price = float(getattr(config.reward, "valley_price", np.min(price)))
    peak_price = float(getattr(config.reward, "peak_price", np.max(price)))
    import_limit_w = float(config.grid_import_max) * 1e6 if np.isfinite(float(config.grid_import_max)) else float("inf")
    export_limit_w = float(config.grid_export_max) * 1e6 if np.isfinite(float(config.grid_export_max)) else float("inf")
    charge_limit_w = float(config.battery_params.p_charge_max)
    discharge_limit_w = float(config.battery_params.p_discharge_max)

    peak_shave_actions: list[float] = []
    tou_actions: list[float] = []
    blended_actions: list[float] = []

    for idx in range(horizon):
        net_demand_w = float(load_w[idx] - pv_w[idx])
        price_step = float(price[idx])

        desired_peak_power_w = 0.0
        if np.isfinite(import_limit_w) and net_demand_w > import_limit_w:
            desired_peak_power_w = min(net_demand_w - import_limit_w, discharge_limit_w)
        elif np.isfinite(export_limit_w) and net_demand_w < -export_limit_w:
            desired_peak_power_w = -min((-net_demand_w) - export_limit_w, charge_limit_w)

        desired_tou_power_w = 0.0
        if price_step <= valley_price and battery.soc < min(config.battery_params.soc_max, 0.75):
            desired_tou_power_w = -0.35 * charge_limit_w
        elif price_step >= peak_price and battery.soc > max(config.battery_params.soc_min, 0.25):
            desired_tou_power_w = 0.35 * discharge_limit_w

        desired_blended_power_w = desired_peak_power_w if abs(desired_peak_power_w) > 0.0 else desired_tou_power_w

        battery.step(desired_blended_power_w, config.dt_seconds)
        peak_shave_actions.append(_power_to_action(desired_peak_power_w, charge_limit_w, discharge_limit_w))
        tou_actions.append(_power_to_action(desired_tou_power_w, charge_limit_w, discharge_limit_w))
        blended_actions.append(_power_to_action(desired_blended_power_w, charge_limit_w, discharge_limit_w))

    return [
        zero_schedule,
        np.asarray(peak_shave_actions, dtype=np.float32),
        np.asarray(tou_actions, dtype=np.float32),
        np.asarray(blended_actions, dtype=np.float32),
    ]


def optimize_schedule(
    case_key: str,
    battery_model: str,
    regime: str,
    days: int,
    seed: int,
    population_size: int,
    generations: int,
    elite_count: int,
    mutation_scale: float,
    crossover_rate: float,
    reward_profile: str,
    voltage_penalty: float,
    loading_penalty: float,
    soc_penalty: float,
    episode_start_hour: int = 0,
    initial_soc: float | None = None,
    initial_temperature_c: float | None = None,
    terminal_penalty_enabled: bool = True,
) -> tuple[np.ndarray, dict[str, float], pd.DataFrame]:
    horizon = simulation_steps(
        days,
        build_config(
            case_key,
            battery_model,
            days,
            seed,
            regime,
            reward_profile,
            episode_start_hour=episode_start_hour,
        ).dt_seconds,
    )
    rng = np.random.default_rng(int(seed))
    population = rng.uniform(-1.0, 1.0, size=(int(population_size), horizon)).astype(np.float32)
    heuristic_seeds = _heuristic_seed_schedules(
        case_key=case_key,
        battery_model=battery_model,
        regime=regime,
        days=days,
        seed=seed,
        reward_profile=reward_profile,
        episode_start_hour=episode_start_hour,
        initial_soc=initial_soc,
    )
    for idx, schedule in enumerate(heuristic_seeds[: int(population_size)]):
        population[idx] = np.asarray(schedule, dtype=np.float32)
    elite_count = min(max(int(elite_count), 1), int(population_size))
    best_actions = population[0].copy()
    best_score = float("inf")
    best_summary: dict[str, float] = {}
    best_trajectory = pd.DataFrame()

    for generation in range(int(generations)):
        scores = np.zeros(int(population_size), dtype=float)
        summaries: list[dict[str, float]] = []
        trajectories: list[pd.DataFrame] = []
        for idx in range(int(population_size)):
            score, summary, trajectory = evaluate_schedule(
                case_key=case_key,
                battery_model=battery_model,
                regime=regime,
                days=days,
                seed=seed,
                actions=population[idx],
                reward_profile=reward_profile,
                voltage_penalty=voltage_penalty,
                loading_penalty=loading_penalty,
                soc_penalty=soc_penalty,
                episode_start_hour=episode_start_hour,
                initial_soc=initial_soc,
                initial_temperature_c=initial_temperature_c,
                terminal_penalty_enabled=terminal_penalty_enabled,
            )
            scores[idx] = score
            summaries.append(summary)
            trajectories.append(trajectory)
            if score < best_score:
                best_score = float(score)
                best_actions = population[idx].copy()
                best_summary = dict(summary)
                best_trajectory = trajectory.copy()

        order = np.argsort(scores)
        elites = population[order[:elite_count]].copy()
        next_population = [elite.copy() for elite in elites]

        while len(next_population) < int(population_size):
            parent_a = population[tournament_select(rng, scores)]
            parent_b = population[tournament_select(rng, scores)]
            blend = float(crossover_rate)
            child = blend * parent_a + (1.0 - blend) * parent_b
            child += rng.normal(0.0, float(mutation_scale), size=child.shape)
            child = np.clip(child, -1.0, 1.0)
            next_population.append(child.astype(np.float32))

        population = np.asarray(next_population, dtype=np.float32)
        print(
            f"[ga] case={case_key} regime={regime} model={battery_model} generation={generation + 1}/{generations} "
            f"best_objective={best_score:.3f}"
        )

    return best_actions, best_summary, best_trajectory


def optimize_schedule_rolling(
    case_key: str,
    battery_model: str,
    regime: str,
    days: int,
    seed: int,
    population_size: int,
    generations: int,
    elite_count: int,
    mutation_scale: float,
    crossover_rate: float,
    reward_profile: str,
    voltage_penalty: float,
    loading_penalty: float,
    soc_penalty: float,
    rolling_window_days: int,
    rolling_stride_days: int,
) -> tuple[np.ndarray, dict[str, float], pd.DataFrame]:
    exec_config = build_config(case_key, battery_model, days, seed, regime, reward_profile)
    exec_env = build_env_from_config(exec_config)
    steps_day = steps_per_day(exec_config.dt_seconds)
    total_steps = simulation_steps(days, exec_config.dt_seconds)
    window_days = max(int(rolling_window_days), 1)
    stride_days = min(max(int(rolling_stride_days), 1), window_days)
    executed_actions: list[float] = []
    trajectory_parts: list[pd.DataFrame] = []
    planning_records: list[dict[str, float | int]] = []

    try:
        _initialize_env_state(exec_env, seed=seed)
        current_step = 0
        while current_step < total_steps:
            remaining_steps = total_steps - current_step
            remaining_days = max(int(np.ceil(remaining_steps / max(steps_day, 1))), 1)
            plan_days = min(window_days, remaining_days)
            execute_days = min(stride_days, remaining_days)
            execute_steps = min(execute_days * steps_day, remaining_steps)
            episode_start_hour = int(current_step * exec_config.dt_seconds / 3600.0)
            terminal_penalty_enabled = execute_steps >= remaining_steps
            start_soc = float(exec_env.battery.soc)
            start_temperature_c = float(getattr(exec_env.battery, "temperature_c", 25.0))
            planned_actions, window_summary, _ = optimize_schedule(
                case_key=case_key,
                battery_model=battery_model,
                regime=regime,
                days=int(plan_days),
                seed=int(seed),
                population_size=int(population_size),
                generations=int(generations),
                elite_count=int(elite_count),
                mutation_scale=float(mutation_scale),
                crossover_rate=float(crossover_rate),
                reward_profile=str(reward_profile),
                voltage_penalty=float(voltage_penalty),
                loading_penalty=float(loading_penalty),
                soc_penalty=float(soc_penalty),
                episode_start_hour=episode_start_hour,
                initial_soc=start_soc,
                initial_temperature_c=start_temperature_c,
                terminal_penalty_enabled=terminal_penalty_enabled,
            )
            executed_chunk = np.asarray(planned_actions[:execute_steps], dtype=np.float32)
            _, _, chunk_trajectory = _rollout_actions(
                exec_env,
                actions=executed_chunk,
                voltage_penalty=float(voltage_penalty),
                loading_penalty=float(loading_penalty),
                soc_penalty=float(soc_penalty),
            )
            if not chunk_trajectory.empty:
                chunk_trajectory = chunk_trajectory.copy()
                chunk_trajectory["step"] = chunk_trajectory["step"].astype(int) + int(current_step)
                trajectory_parts.append(chunk_trajectory)
            executed_actions.extend(float(value) for value in executed_chunk)
            planning_records.append(
                {
                    "window_index": int(len(planning_records)),
                    "start_step": int(current_step),
                    "execute_steps": int(execute_steps),
                    "plan_days": int(plan_days),
                    "execute_days": int(execute_days),
                    "planned_objective_value": float(window_summary.get("objective_value", 0.0)),
                    "start_soc": start_soc,
                    "end_soc": float(chunk_trajectory["soc"].iloc[-1]) if not chunk_trajectory.empty else float(exec_env.battery.soc),
                    "terminal_penalty_enabled": int(terminal_penalty_enabled),
                }
            )
            if chunk_trajectory.empty:
                break
            current_step += int(len(chunk_trajectory))

        full_trajectory = pd.concat(trajectory_parts, ignore_index=True) if trajectory_parts else pd.DataFrame()
        final_cost = float(full_trajectory["cumulative_cost"].iloc[-1]) if not full_trajectory.empty else 0.0
        final_objective_cost = (
            float(full_trajectory["cumulative_objective_cost"].iloc[-1])
            if not full_trajectory.empty
            else final_cost
        )
        total_penalty = float(full_trajectory["penalty_objective_step"].sum()) if not full_trajectory.empty else 0.0
        summary = {
            "steps": int(len(full_trajectory)),
            "objective_value": float(final_objective_cost + total_penalty),
            "penalty_value": float(total_penalty),
            "total_reward": float(full_trajectory["reward"].sum()) if not full_trajectory.empty else 0.0,
            "final_cumulative_cost": final_cost,
            "final_cumulative_objective_cost": final_objective_cost,
            "final_soc": float(full_trajectory["soc"].iloc[-1]) if not full_trajectory.empty else float(exec_env.battery.soc),
            "total_terminal_soc_penalty": float(full_trajectory["terminal_soc_penalty"].sum()) if not full_trajectory.empty else 0.0,
            "min_voltage_worst": float(full_trajectory["min_bus_voltage_pu"].min()) if not full_trajectory.empty else 1.0,
            "max_line_loading_peak": float(full_trajectory["max_line_loading_pct"].max()) if not full_trajectory.empty else 0.0,
            "mean_grid_import_mw": float(full_trajectory["grid_import_mw"].mean()) if not full_trajectory.empty else 0.0,
            "undervoltage_total": float(full_trajectory["undervoltage"].sum()) if not full_trajectory.empty else 0.0,
            "overvoltage_total": float(full_trajectory["overvoltage"].sum()) if not full_trajectory.empty else 0.0,
            "line_overload_total": float(full_trajectory["line_overload_pct"].sum()) if not full_trajectory.empty else 0.0,
            "transformer_overload_total": float(full_trajectory["transformer_overload_pct"].sum()) if not full_trajectory.empty else 0.0,
            "soc_violation_total": float(full_trajectory["soc_violation"].sum()) if not full_trajectory.empty else 0.0,
            "rolling_window_days": int(window_days),
            "rolling_stride_days": int(stride_days),
            "rolling_window_count": int(len(planning_records)),
        }
        planning_df = pd.DataFrame(planning_records)
        if not planning_df.empty:
            full_trajectory.attrs["rolling_plans"] = planning_df.to_dict(orient="records")
        return np.asarray(executed_actions, dtype=np.float32), summary, full_trajectory
    finally:
        exec_env.close()


def main() -> int:
    args = build_parser().parse_args()
    case_keys = _parse_csv_arg(args.cases)
    regimes = _parse_csv_arg(args.regimes)
    battery_models = _parse_csv_arg(args.battery_models)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectories_dir = output_dir / "trajectories"
    trajectories_dir.mkdir(parents=True, exist_ok=True)
    resolved_stride_days = int(args.rolling_stride_days) if int(args.rolling_stride_days) > 0 else int(args.rolling_window_days)

    summary_rows: list[dict[str, float | int | str]] = []
    for case_key in case_keys:
        for regime in regimes:
            for battery_model in battery_models:
                baseline_name = "ga_dispatch"
                if int(args.rolling_window_days) > 0:
                    actions, summary, trajectory = optimize_schedule_rolling(
                        case_key=case_key,
                        battery_model=battery_model,
                        regime=regime,
                        days=int(args.days),
                        seed=int(args.seed),
                        population_size=int(args.population_size),
                        generations=int(args.generations),
                        elite_count=int(args.elite_count),
                        mutation_scale=float(args.mutation_scale),
                        crossover_rate=float(args.crossover_rate),
                        reward_profile=str(args.reward_profile),
                        voltage_penalty=float(args.voltage_penalty),
                        loading_penalty=float(args.loading_penalty),
                        soc_penalty=float(args.soc_penalty),
                        rolling_window_days=int(args.rolling_window_days),
                        rolling_stride_days=int(resolved_stride_days),
                    )
                    baseline_name = "ga_dispatch_rolling"
                else:
                    actions, summary, trajectory = optimize_schedule(
                        case_key=case_key,
                        battery_model=battery_model,
                        regime=regime,
                        days=int(args.days),
                        seed=int(args.seed),
                        population_size=int(args.population_size),
                        generations=int(args.generations),
                        elite_count=int(args.elite_count),
                        mutation_scale=float(args.mutation_scale),
                        crossover_rate=float(args.crossover_rate),
                        reward_profile=str(args.reward_profile),
                        voltage_penalty=float(args.voltage_penalty),
                        loading_penalty=float(args.loading_penalty),
                        soc_penalty=float(args.soc_penalty),
                    )
                row = {
                    "case": case_key,
                    "regime": regime,
                    "baseline": baseline_name,
                    "seed": int(args.seed),
                    "battery_model": battery_model,
                    "days": int(args.days),
                    "population_size": int(args.population_size),
                    "generations": int(args.generations),
                    "elite_count": int(args.elite_count),
                    "mutation_scale": float(args.mutation_scale),
                    "crossover_rate": float(args.crossover_rate),
                    "rolling_window_days": int(args.rolling_window_days),
                    "rolling_stride_days": int(resolved_stride_days),
                    "reward_profile": str(args.reward_profile),
                    "voltage_penalty": float(args.voltage_penalty),
                    "loading_penalty": float(args.loading_penalty),
                    "soc_penalty": float(args.soc_penalty),
                    **summary,
                }
                summary_rows.append(row)

                stem = f"{case_key}_{regime}_ga_{battery_model}_seed{args.seed}"
                action_df = pd.DataFrame({"step": np.arange(len(actions), dtype=int), "optimized_action": np.asarray(actions, dtype=float)})
                action_df.to_csv(trajectories_dir / f"{stem}_actions.csv", index=False)
                trajectory.to_csv(trajectories_dir / f"{stem}_trajectory.csv", index=False)
                rolling_plans = trajectory.attrs.get("rolling_plans")
                if rolling_plans:
                    pd.DataFrame(rolling_plans).to_csv(trajectories_dir / f"{stem}_rolling_windows.csv", index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = output_dir / "summary.csv"
    summary_json = output_dir / "summary.json"
    summary_df.to_csv(summary_csv, index=False)
    summary_json.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    print(summary_df.round(6).to_string(index=False))
    print(f"\nSaved summary CSV: {summary_csv}")
    print(f"Saved summary JSON: {summary_json}")
    print(f"Saved trajectories: {trajectories_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
