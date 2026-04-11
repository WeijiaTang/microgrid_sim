#!/usr/bin/env python3
"""Genetic-algorithm dispatch baseline for network-first microgrids."""
# Ref: docs/spec/task.md (Task-ID: SPEC-FIDELITY-MISMATCH-001)

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

from microgrid_sim.cases import CIGREEuropeanLVConfig, IEEE33Config
from microgrid_sim.envs.network_microgrid import NetworkMicrogridEnv
from microgrid_sim.time_utils import simulation_steps


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
    parser.add_argument("--voltage-penalty", type=float, default=2_000.0, help="Penalty multiplier for undervoltage/overvoltage")
    parser.add_argument("--loading-penalty", type=float, default=250.0, help="Penalty multiplier for line/transformer overload")
    parser.add_argument("--soc-penalty", type=float, default=1_000.0, help="Penalty multiplier for SOC violation")
    parser.add_argument("--output-dir", type=str, default="results/genetic_dispatch_baseline", help="Output directory")
    return parser


def _parse_csv_arg(raw: str) -> list[str]:
    return [item.strip().lower() for item in str(raw).split(",") if item.strip()]


def build_config(case_key: str, battery_model: str, days: int, seed: int, regime: str, reward_profile: str):
    if case_key == "ieee33":
        return IEEE33Config(simulation_days=days, seed=seed, battery_model=battery_model, regime=regime, reward_profile=reward_profile)
    if case_key == "cigre":
        return CIGREEuropeanLVConfig(simulation_days=days, seed=seed, battery_model=battery_model, regime=regime, reward_profile=reward_profile)
    raise ValueError(f"Unsupported case '{case_key}'")


def build_env(case_key: str, battery_model: str, days: int, seed: int, regime: str, reward_profile: str) -> NetworkMicrogridEnv:
    return NetworkMicrogridEnv(
        build_config(
            case_key=case_key,
            battery_model=battery_model,
            days=days,
            seed=seed,
            regime=regime,
            reward_profile=reward_profile,
        )
    )


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
) -> tuple[float, dict[str, float], pd.DataFrame]:
    env = build_env(case_key=case_key, battery_model=battery_model, days=days, seed=seed, regime=regime, reward_profile=reward_profile)
    rows: list[dict[str, float | int | str]] = []
    try:
        obs, info = env.reset(seed=seed)
        del obs, info
        total_penalty = 0.0
        for step, action in enumerate(np.asarray(actions, dtype=float).reshape(-1)):
            _, reward, terminated, truncated, info = env.step(np.array([float(np.clip(action, -1.0, 1.0))], dtype=np.float32))
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
                    "step": int(step),
                    "action": float(np.clip(action, -1.0, 1.0)),
                    "reward": float(reward),
                    "soc": float(info.get("soc", 0.0)),
                    "battery_power_mw": float(info.get("battery_power_mw", 0.0)),
                    "grid_import_mw": float(info.get("grid_import_mw", 0.0)),
                    "grid_export_mw": float(info.get("grid_export_mw", 0.0)),
                    "cumulative_cost": float(info.get("cumulative_cost", 0.0)),
                    "min_bus_voltage_pu": float(info.get("min_bus_voltage_pu", 1.0)),
                    "max_line_loading_pct": float(info.get("max_line_loading_pct", 0.0)),
                    "undervoltage": undervoltage,
                    "overvoltage": overvoltage,
                    "line_overload_pct": line_overload,
                    "transformer_overload_pct": trafo_overload,
                    "soc_violation": soc_violation,
                    "penalty_objective_step": float(step_penalty),
                }
            )
            if terminated or truncated:
                break
        trajectory = pd.DataFrame(rows)
        final_cost = float(trajectory["cumulative_cost"].iloc[-1]) if not trajectory.empty else 0.0
        objective = final_cost + float(total_penalty)
        summary = {
            "steps": int(len(trajectory)),
            "objective_value": float(objective),
            "penalty_value": float(total_penalty),
            "total_reward": float(trajectory["reward"].sum()) if not trajectory.empty else 0.0,
            "final_cumulative_cost": final_cost,
            "final_soc": float(trajectory["soc"].iloc[-1]) if not trajectory.empty else 0.0,
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
    finally:
        env.close()


def tournament_select(rng: np.random.Generator, scores: np.ndarray, tournament_size: int = 3) -> int:
    participants = rng.integers(0, len(scores), size=max(int(tournament_size), 2))
    return int(participants[np.argmin(scores[participants])])


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
) -> tuple[np.ndarray, dict[str, float], pd.DataFrame]:
    horizon = simulation_steps(days, build_config(case_key, battery_model, days, seed, regime, reward_profile).dt_seconds)
    rng = np.random.default_rng(int(seed))
    population = rng.uniform(-1.0, 1.0, size=(int(population_size), horizon)).astype(np.float32)
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


def main() -> int:
    args = build_parser().parse_args()
    case_keys = _parse_csv_arg(args.cases)
    regimes = _parse_csv_arg(args.regimes)
    battery_models = _parse_csv_arg(args.battery_models)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectories_dir = output_dir / "trajectories"
    trajectories_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, float | int | str]] = []
    for case_key in case_keys:
        for regime in regimes:
            for battery_model in battery_models:
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
                    "baseline": "ga_dispatch",
                    "seed": int(args.seed),
                    "battery_model": battery_model,
                    "days": int(args.days),
                    "population_size": int(args.population_size),
                    "generations": int(args.generations),
                    "elite_count": int(args.elite_count),
                    "mutation_scale": float(args.mutation_scale),
                    "crossover_rate": float(args.crossover_rate),
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
