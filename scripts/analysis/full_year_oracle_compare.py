#!/usr/bin/env python3
"""Fast full-year Oracle comparison using direct LP solves without pandapower replay."""

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

from microgrid_sim.baselines.dispatch import MILPOptimizer, _extract_network_env_forecasts  # noqa: E402
from microgrid_sim.cases import CIGREEuropeanLVConfig, IEEE33Config  # noqa: E402
from microgrid_sim.data.network_profiles import load_network_profiles  # noqa: E402
from microgrid_sim.envs.network_microgrid import NetworkMicrogridEnv  # noqa: E402
from microgrid_sim.time_utils import simulation_steps, steps_per_day, steps_per_hour  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fast full-year Oracle storage comparison on IEEE33 and CIGRE.")
    parser.add_argument("--cases", type=str, default="ieee33,cigre", help="Comma-separated case keys")
    parser.add_argument("--regimes", type=str, default="network_stress", help="Comma-separated regimes")
    parser.add_argument("--reward-profile", type=str, default="paper_balanced", help="Reward profile")
    parser.add_argument("--battery-model", type=str, default="simple", help="Storage battery model")
    parser.add_argument("--days", type=int, default=365, help="Simulation days")
    parser.add_argument("--year", type=int, default=0, help="Optional calendar year restriction, e.g. 2024")
    parser.add_argument("--offset-days-within-year", type=int, default=0, help="Optional start-day offset inside --year")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--efficiency-model", type=str, default="realistic", help="MILP efficiency model")
    parser.add_argument("--output-dir", type=Path, default=Path("results/full_year_oracle_compare"), help="Output directory")
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


def resolve_year_window(*, case_key: str, year: int, regime: str, reward_profile: str, seed: int) -> dict[str, int | str]:
    probe_config = build_config(
        case_key=case_key,
        battery_model="simple",
        days=1,
        seed=int(seed),
        regime=regime,
        reward_profile=reward_profile,
    )
    full_profiles = load_network_profiles(probe_config)
    timestamps = pd.DatetimeIndex(full_profiles.timestamps)
    mask = timestamps.year == int(year)
    if not mask.any():
        available_years = sorted({int(value) for value in timestamps.year})
        raise ValueError(f"Requested year {year} is unavailable for case '{case_key}'. Available years: {available_years}")
    indices = np.flatnonzero(mask.to_numpy() if hasattr(mask, "to_numpy") else np.asarray(mask, dtype=bool))
    start_step = int(indices[0])
    steps = int(len(indices))
    dt_steps_per_hour = steps_per_hour(probe_config.dt_seconds)
    dt_steps_per_day = steps_per_day(probe_config.dt_seconds)
    if start_step % dt_steps_per_hour != 0:
        raise ValueError(f"Year {year} for case '{case_key}' does not start on an hour boundary.")
    if steps % dt_steps_per_day != 0:
        raise ValueError(f"Year {year} for case '{case_key}' does not span an integer number of days.")
    expected = np.arange(start_step, start_step + steps, dtype=int)
    if not np.array_equal(indices, expected):
        raise ValueError(f"Year {year} for case '{case_key}' is not contiguous in the canonical dataset.")
    return {
        "year": int(year),
        "start_step": int(start_step),
        "start_hour": int(start_step // dt_steps_per_hour),
        "steps": int(steps),
        "days": int(steps // dt_steps_per_day),
        "start_timestamp": str(timestamps[start_step]),
        "end_timestamp": str(timestamps[start_step + steps - 1]),
    }


def solve_case(config, *, efficiency_model: str, seed: int) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    env = NetworkMicrogridEnv(config)
    try:
        env.reset(seed=int(seed))
        total_steps = simulation_steps(config.simulation_days, config.dt_seconds)
        forecasts = _extract_network_env_forecasts(env, total_steps)
        optimizer = MILPOptimizer(
            env.battery.params,
            horizon=total_steps,
            efficiency_model=str(efficiency_model),
            feed_in_tariff=float(getattr(config, "feed_in_tariff", 0.0)),
            grid_import_max=float(getattr(config, "grid_import_max", float("inf"))),
            grid_export_max=float(getattr(config, "grid_export_max", float("inf"))),
            grid_limit_violation_penalty_per_kwh=float(getattr(config, "grid_limit_violation_penalty_per_kwh", 0.0)),
            peak_import_penalty_per_kw=float(getattr(config, "peak_import_penalty_per_kw", 0.0)),
            peak_import_penalty_threshold_w=float(getattr(config, "peak_import_penalty_threshold_w", float("inf"))),
            monthly_demand_charge_per_kw=float(getattr(config, "monthly_demand_charge_per_kw", 0.0)),
            monthly_demand_charge_threshold_w=float(getattr(config, "monthly_demand_charge_threshold_w", float("inf"))),
            battery_throughput_penalty_per_kwh=float(getattr(config, "battery_throughput_penalty_per_kwh", 0.0)),
            terminal_soc_target=getattr(config, "terminal_soc_target", None),
            terminal_soc_tolerance=float(getattr(config, "terminal_soc_tolerance", 0.0)),
            terminal_soc_penalty_per_kwh=float(getattr(config, "terminal_soc_penalty_per_kwh", 0.0)),
            dt_seconds=float(config.dt_seconds),
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
        timeline = pd.DataFrame(
            {
                "step": np.arange(total_steps, dtype=int),
                "timestamp": pd.DatetimeIndex(forecasts["timestamps"]).astype(str),
                "price": np.asarray(forecasts["price"], dtype=float),
                "load_w": np.asarray(forecasts["load_w"], dtype=float),
                "pv_w": np.asarray(forecasts["pv_w"], dtype=float),
                "battery_schedule_w": np.asarray(details["battery_schedule_w"], dtype=float),
                "import_schedule_w": np.asarray(details["import_schedule_w"], dtype=float),
                "export_schedule_w": np.asarray(details["export_schedule_w"], dtype=float),
                "soc": np.asarray(details["soc_schedule"], dtype=float),
            }
        )
        row = {
            "case": str(getattr(config, "benchmark_name", config.case_name)),
            "case_key": str(config.case_key),
            "regime": str(getattr(config, "regime", "")),
            "battery_model": str(getattr(config, "battery_model", "")),
            "days": int(config.simulation_days),
            "seed": int(seed),
            "efficiency_model": str(efficiency_model),
            "profile_start_timestamp": str(forecasts["timestamps"][0]),
            "profile_end_timestamp": str(forecasts["timestamps"][-1]),
            "objective_value": float(objective_value),
            "energy_import_cost": float(details["energy_import_cost"]),
            "feed_in_revenue": float(details["feed_in_revenue"]),
            "grid_limit_penalty_cost": float(details["grid_limit_penalty_cost"]),
            "throughput_penalty_cost": float(details["throughput_penalty_cost"]),
            "peak_import_penalty_cost": float(details["peak_import_penalty_cost"]),
            "monthly_demand_charge_cost": float(details["monthly_demand_charge_cost"]),
            "terminal_soc_penalty_cost": float(details["terminal_soc_penalty_cost"]),
            "final_soc": float(np.asarray(details["soc_schedule"], dtype=float)[-1]) if total_steps > 0 else float(env.battery.soc),
            "min_soc": float(np.min(np.asarray(details["soc_schedule"], dtype=float))) if total_steps > 0 else float(env.battery.soc),
            "max_soc": float(np.max(np.asarray(details["soc_schedule"], dtype=float))) if total_steps > 0 else float(env.battery.soc),
            "total_battery_throughput_mwh": float(
                np.sum(np.abs(np.asarray(details["battery_schedule_w"], dtype=float))) * float(config.dt_seconds) / 3_600_000_000.0
            ),
        }
        return row, timeline
    finally:
        env.close()


def build_protocol_summary(detail_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for (case_key, regime), group in detail_df.groupby(["case_key", "regime"], dropna=False):
        by_model = {str(row["battery_model"]): row for _, row in group.iterrows()}
        none_row = by_model.get("none")
        storage_row = next((row for model, row in by_model.items() if model != "none"), None)
        if none_row is None or storage_row is None:
            continue
        none_objective = float(none_row["objective_value"])
        oracle_objective = float(storage_row["objective_value"])
        savings = none_objective - oracle_objective
        rows.append(
            {
                "protocol": f"full_year_{int(storage_row['days'])}d_lp_oracle",
                "case": str(storage_row["case"]),
                "case_key": str(case_key),
                "regime": str(regime),
                "battery_model": str(storage_row["battery_model"]),
                "none_objective": none_objective,
                "oracle_objective": oracle_objective,
                "oracle_savings_vs_none": float(savings),
                "oracle_savings_pct_vs_none": float(savings / none_objective) if abs(none_objective) > 1e-9 else np.nan,
                "none_final_soc": float(none_row["final_soc"]),
                "oracle_final_soc": float(storage_row["final_soc"]),
                "oracle_min_soc": float(storage_row["min_soc"]),
                "oracle_max_soc": float(storage_row["max_soc"]),
                "oracle_throughput_mwh": float(storage_row["total_battery_throughput_mwh"]),
                "profile_start_timestamp": str(storage_row["profile_start_timestamp"]),
                "profile_end_timestamp": str(storage_row["profile_end_timestamp"]),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    args = build_parser().parse_args()
    case_keys = _parse_csv_arg(args.cases)
    regimes = _parse_csv_arg(args.regimes)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectories_dir = output_dir / "timelines"
    trajectories_dir.mkdir(parents=True, exist_ok=True)

    detail_rows: list[dict[str, float | int | str]] = []
    for case_key in case_keys:
        for regime in regimes:
            print(f"[oracle-fast] case={case_key} regime={regime} days={args.days}")
            for battery_model in ("none", str(args.battery_model)):
                config = build_config(case_key, battery_model, args.days, args.seed, regime, args.reward_profile)
                if int(args.year) > 0:
                    year_window = resolve_year_window(
                        case_key=case_key,
                        year=int(args.year),
                        regime=regime,
                        reward_profile=str(args.reward_profile),
                        seed=int(args.seed),
                    )
                    available_days = int(year_window["days"])
                    offset_days = max(int(args.offset_days_within_year), 0)
                    if offset_days >= available_days:
                        raise ValueError(
                            f"Requested offset_days_within_year={offset_days} exceeds available days={available_days} for year {args.year} case '{case_key}'."
                        )
                    if offset_days + int(args.days) > available_days:
                        raise ValueError(
                            f"Requested days={int(args.days)} with offset_days_within_year={offset_days} exceeds available days={available_days} for year {args.year} case '{case_key}'."
                        )
                    config.episode_start_hour = int(year_window["start_hour"]) + offset_days * 24
                    config.random_episode_start = False
                row, timeline = solve_case(config, efficiency_model=str(args.efficiency_model), seed=int(args.seed))
                detail_rows.append(row)
                timeline.to_csv(trajectories_dir / f"{case_key}_{regime}_{battery_model}_timeline.csv", index=False)

    detail_df = pd.DataFrame(detail_rows)
    protocol_df = build_protocol_summary(detail_df)
    detail_csv = output_dir / "detail.csv"
    protocol_csv = output_dir / "protocol_summary.csv"
    detail_json = output_dir / "detail.json"
    detail_df.to_csv(detail_csv, index=False)
    protocol_df.to_csv(protocol_csv, index=False)
    detail_json.write_text(json.dumps(detail_rows, indent=2), encoding="utf-8")

    print("\n=== Full-Year Oracle Detail ===")
    print(detail_df.round(6).to_string(index=False))
    print("\n=== Full-Year Oracle Summary ===")
    print(protocol_df.round(6).to_string(index=False))
    print(f"\nSaved detail CSV: {detail_csv}")
    print(f"Saved protocol summary CSV: {protocol_csv}")
    print(f"Saved detail JSON: {detail_json}")
    print(f"Saved timelines: {trajectories_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
