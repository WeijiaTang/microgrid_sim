#!/usr/bin/env python3
"""Replay LP oracle schedules through NetworkMicrogridEnv for value-recovery references.

The LP oracle reference is useful for producing schedules quickly, but its objective
is lossless net-load accounting. Reviewer-facing learned-policy costs come from the
network environment's pandapower slack import. This script rebuilds the none/oracle
reference with the same environment accounting used by evaluation trajectories.
"""
# Ref: docs/spec/task.md (Task-ID: SPEC-SAFE-OFFLINE-RL-001)

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.analysis.full_year_oracle_compare import build_config, resolve_year_window  # noqa: E402
from microgrid_sim.envs.network_microgrid import NetworkMicrogridEnv  # noqa: E402
from microgrid_sim.time_utils import steps_per_day  # noqa: E402


CASE_ALIASES = {
    "ieee33": {"ieee33", "ieee33_network", "ieee33network", "ieee33-bus", "ieee 33-bus", "ieee33bus"},
    "cigre": {"cigre", "cigre_eu_lv_network", "cigreeulvnetwork", "cigre european lv", "cigreeuropeanlv"},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read an LP oracle reference suite with per-window timelines and produce a metric-style "
            "none-vs-oracle CSV replayed through NetworkMicrogridEnv/pandapower."
        )
    )
    parser.add_argument(
        "--oracle-reference-csv",
        type=Path,
        default=Path("results/oracle_reference_suite_ieee33_simple_2024/oracle_reference_windows.csv"),
        help="Combined LP oracle reference CSV produced by build_oracle_reference_suite.py.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help=(
            "Output metric-style CSV. Defaults to network_replayed_oracle_reference_windows.csv "
            "next to --oracle-reference-csv."
        ),
    )
    parser.add_argument("--reward-profile", type=str, default="paper_balanced", help="Reward/objective profile for env replay.")
    parser.add_argument("--seed", type=int, default=42, help="Replay seed.")
    parser.add_argument(
        "--metrics",
        type=str,
        default="final_cumulative_cost,final_cumulative_objective_cost",
        help="Comma-separated replay metrics to export for compare_safe_warmstart_sac.py.",
    )
    parser.add_argument(
        "--write-timelines",
        action="store_true",
        help="Also save replayed none/oracle timelines under a sibling network_replayed_timelines directory.",
    )
    return parser


def _parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _canonical_case_key(value: Any) -> str:
    token = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    compact = token.replace("_", "")
    for canonical, aliases in CASE_ALIASES.items():
        normalized_aliases = {alias.replace("-", "_").replace(" ", "_").lower() for alias in aliases}
        normalized_aliases |= {alias.replace("_", "") for alias in normalized_aliases}
        if token in normalized_aliases or compact in normalized_aliases:
            return canonical
    return token


def _window_days(row: pd.Series) -> int:
    for column in ("reference_window_days", "days"):
        if column in row and pd.notna(row[column]):
            return int(row[column])
    start = pd.to_datetime(row.get("profile_start_timestamp"), errors="coerce")
    end = pd.to_datetime(row.get("profile_end_timestamp"), errors="coerce")
    if pd.isna(start) or pd.isna(end):
        raise ValueError("Cannot infer reference window length; missing days and timestamps.")
    return int(round(((end - start).total_seconds() + 900.0) / 86400.0))


def _offset_days(row: pd.Series) -> int:
    if "reference_offset_days_within_year" in row and pd.notna(row["reference_offset_days_within_year"]):
        return int(row["reference_offset_days_within_year"])
    start = pd.to_datetime(row.get("profile_start_timestamp"), errors="coerce")
    if pd.isna(start):
        return 0
    return int((start - pd.Timestamp(year=int(start.year), month=1, day=1)).days)


def _timeline_path(row: pd.Series, *, oracle_reference_csv: Path) -> Path:
    source_dir_raw = str(row.get("reference_source_dir", "")).strip()
    if source_dir_raw:
        source_dir = Path(source_dir_raw)
        if not source_dir.is_absolute():
            source_dir = REPO_ROOT / source_dir
    else:
        source_dir = oracle_reference_csv.parent / "runs" / f"{_window_days(row)}d_off{_offset_days(row)}"
    case_key = _canonical_case_key(row.get("case_key", row.get("case", "")))
    # full_year_oracle_compare writes canonical short case keys in timeline names.
    battery_model = str(row.get("battery_model", "")).strip()
    regime = str(row.get("regime", "")).strip()
    path = source_dir / "timelines" / f"{case_key}_{regime}_{battery_model}_timeline.csv"
    if not path.exists():
        raise FileNotFoundError(f"Oracle timeline not found for network replay: {path}")
    return path


def _configure_window(config: Any, *, case_key: str, regime: str, reward_profile: str, seed: int, year: int, days: int, offset_days: int) -> Any:
    year_window = resolve_year_window(
        case_key=case_key,
        year=int(year),
        regime=str(regime),
        reward_profile=str(reward_profile),
        seed=int(seed),
    )
    available_days = int(year_window["days"])
    if int(offset_days) + int(days) > available_days:
        raise ValueError(f"Requested replay window days={days} offset={offset_days} exceeds available days={available_days}.")
    config.episode_start_hour = int(year_window["start_hour"]) + int(offset_days) * 24
    config.random_episode_start = False
    return config


def normalized_action_for_requested_power(env: NetworkMicrogridEnv, requested_power_w: float) -> float:
    """Map an absolute battery power command to the env's current normalized action."""
    min_command_w, max_command_w = env.battery.power_command_bounds(dt=float(env.config.dt_seconds))
    requested = float(requested_power_w)
    if requested >= 0.0:
        denom = max(float(max_command_w), 0.0)
        return float(np.clip(requested / denom, 0.0, 1.0)) if denom > 1e-12 else 0.0
    denom = max(-float(min_command_w), 0.0)
    return float(np.clip(requested / denom, -1.0, 0.0)) if denom > 1e-12 else 0.0


def replay_env_costs(
    *,
    case_key: str,
    regime: str,
    battery_model: str,
    reward_profile: str,
    seed: int,
    year: int,
    days: int,
    offset_days: int,
    battery_schedule_w: np.ndarray | None,
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    config = build_config(case_key, battery_model, days, seed, regime, reward_profile)
    config = _configure_window(
        config,
        case_key=case_key,
        regime=regime,
        reward_profile=reward_profile,
        seed=seed,
        year=year,
        days=days,
        offset_days=offset_days,
    )
    total_steps = int(days) * steps_per_day(float(config.dt_seconds))
    if battery_schedule_w is not None and len(battery_schedule_w) < total_steps:
        raise ValueError(f"Battery schedule has {len(battery_schedule_w)} rows but replay needs {total_steps}.")

    env = NetworkMicrogridEnv(config)
    rows: list[dict[str, float | int | str]] = []
    try:
        env.reset(seed=int(seed))
        last_info: dict[str, Any] = {}
        for step in range(total_steps):
            requested_power_w = 0.0 if battery_schedule_w is None else float(battery_schedule_w[step])
            action = normalized_action_for_requested_power(env, requested_power_w)
            _, _, terminated, truncated, info = env.step(np.array([action], dtype=np.float32))
            last_info = dict(info)
            rows.append(
                {
                    "step": int(step),
                    "timestamp": str(info.get("timestamp", "")),
                    "requested_battery_schedule_w": float(requested_power_w),
                    "replay_action": float(action),
                    "battery_power_w": float(info.get("battery_power_w", 0.0)),
                    "grid_import_mw": float(info.get("grid_import_mw", 0.0)),
                    "grid_export_mw": float(info.get("grid_export_mw", 0.0)),
                    "total_grid_cost": float(info.get("total_grid_cost", 0.0)),
                    "cumulative_cost": float(info.get("cumulative_cost", 0.0)),
                    "cumulative_objective_cost": float(info.get("cumulative_objective_cost", 0.0)),
                    "soc": float(info.get("soc", 0.0)),
                }
            )
            if bool(terminated or truncated):
                break
        if not last_info:
            raise RuntimeError("Replay produced no environment steps.")
        result = {
            "final_cumulative_cost": float(last_info.get("cumulative_cost", 0.0)),
            "final_cumulative_objective_cost": float(last_info.get("cumulative_objective_cost", 0.0)),
            "final_soc": float(last_info.get("soc", 0.0)),
            "total_battery_throughput_kwh": float(sum(abs(float(row["battery_power_w"])) for row in rows) * float(config.dt_seconds) / 3_600_000.0),
            "profile_start_timestamp": str(rows[0]["timestamp"]),
            "profile_end_timestamp": str(rows[-1]["timestamp"]),
            "steps": int(len(rows)),
        }
        return result, pd.DataFrame(rows)
    finally:
        env.close()


def build_network_replayed_reference(
    reference_df: pd.DataFrame,
    *,
    oracle_reference_csv: Path,
    reward_profile: str,
    seed: int,
    metrics: list[str],
    timelines_output_dir: Path | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    if reference_df.empty:
        return pd.DataFrame()

    for _, source_row in reference_df.iterrows():
        case_key = _canonical_case_key(source_row.get("case_key", source_row.get("case", "")))
        regime = str(source_row.get("regime", "")).strip()
        battery_model = str(source_row.get("battery_model", "")).strip()
        days = _window_days(source_row)
        offset_days = _offset_days(source_row)
        start_ts = pd.to_datetime(source_row.get("profile_start_timestamp"), errors="coerce")
        year = int(start_ts.year) if pd.notna(start_ts) else 2024
        timeline = pd.read_csv(_timeline_path(source_row, oracle_reference_csv=oracle_reference_csv))
        schedule_w = timeline["battery_schedule_w"].astype(float).to_numpy()

        print(f"[network-replay-oracle] case={case_key} regime={regime} model={battery_model} days={days} offset={offset_days}")
        none_result, none_timeline = replay_env_costs(
            case_key=case_key,
            regime=regime,
            battery_model="none",
            reward_profile=reward_profile,
            seed=seed,
            year=year,
            days=days,
            offset_days=offset_days,
            battery_schedule_w=None,
        )
        oracle_result, oracle_timeline = replay_env_costs(
            case_key=case_key,
            regime=regime,
            battery_model=battery_model,
            reward_profile=reward_profile,
            seed=seed,
            year=year,
            days=days,
            offset_days=offset_days,
            battery_schedule_w=schedule_w,
        )
        if timelines_output_dir is not None:
            window_dir = timelines_output_dir / f"{days}d_off{offset_days}"
            window_dir.mkdir(parents=True, exist_ok=True)
            none_timeline.to_csv(window_dir / f"{case_key}_{regime}_none_network_replay.csv", index=False)
            oracle_timeline.to_csv(window_dir / f"{case_key}_{regime}_{battery_model}_oracle_network_replay.csv", index=False)

        for metric in metrics:
            if metric not in none_result or metric not in oracle_result:
                raise KeyError(f"Unsupported replay metric '{metric}'. Available: {sorted(none_result)}")
            none_cost = float(none_result[metric])
            oracle_cost = float(oracle_result[metric])
            rows.append(
                {
                    "protocol": f"network_replayed_{days}d_lp_oracle",
                    "case": str(source_row.get("case", case_key)),
                    "case_key": str(source_row.get("case_key", case_key)),
                    "regime": regime,
                    "battery_model": battery_model,
                    "reference_window_days": int(days),
                    "reference_offset_days_within_year": int(offset_days),
                    "metric": str(metric),
                    "none_cost": none_cost,
                    "oracle_cost": oracle_cost,
                    "oracle_savings_vs_none": float(none_cost - oracle_cost),
                    "oracle_savings_pct_vs_none": float((none_cost - oracle_cost) / none_cost) if abs(none_cost) > 1e-9 else np.nan,
                    "none_final_soc": float(none_result["final_soc"]),
                    "oracle_final_soc": float(oracle_result["final_soc"]),
                    "oracle_throughput_mwh": float(oracle_result["total_battery_throughput_kwh"]) / 1000.0,
                    "profile_start_timestamp": str(oracle_result["profile_start_timestamp"]),
                    "profile_end_timestamp": str(oracle_result["profile_end_timestamp"]),
                    "lp_reference_source_dir": str(source_row.get("reference_source_dir", "")),
                    "reference_source_dir": str(timelines_output_dir or ""),
                    "reward_profile": str(reward_profile),
                    "seed": int(seed),
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    args = build_parser().parse_args(sys.argv[1:])
    oracle_reference_csv = Path(args.oracle_reference_csv)
    if not oracle_reference_csv.exists():
        raise FileNotFoundError(f"Oracle reference CSV not found: {oracle_reference_csv}")
    output_csv = Path(args.output_csv) if args.output_csv is not None else oracle_reference_csv.with_name(
        "network_replayed_oracle_reference_windows.csv"
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    timelines_output_dir = output_csv.parent / "network_replayed_timelines" if bool(args.write_timelines) else None
    reference_df = pd.read_csv(oracle_reference_csv)
    replayed_df = build_network_replayed_reference(
        reference_df,
        oracle_reference_csv=oracle_reference_csv,
        reward_profile=str(args.reward_profile),
        seed=int(args.seed),
        metrics=_parse_csv(str(args.metrics)),
        timelines_output_dir=timelines_output_dir,
    )
    replayed_df.to_csv(output_csv, index=False)
    output_json = output_csv.with_suffix(".json")
    output_json.write_text(json.dumps(replayed_df.to_dict(orient="records"), indent=2), encoding="utf-8")
    print(f"[network-replay-oracle] saved CSV: {output_csv}")
    print(f"[network-replay-oracle] saved JSON: {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
