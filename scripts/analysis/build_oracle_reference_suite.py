#!/usr/bin/env python3
"""Build a multi-window oracle reference suite for reviewer-facing value-recovery normalization."""
# Ref: docs/spec/task.md (Task-ID: SPEC-SAFE-OFFLINE-RL-001)

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run full_year_oracle_compare across multiple window lengths and offsets, then combine the "
            "protocol summaries into one oracle reference CSV for compare/enrich scripts."
        )
    )
    parser.add_argument("--cases", type=str, default="ieee33", help="Comma-separated case keys.")
    parser.add_argument("--regimes", type=str, default="network_stress", help="Comma-separated regimes.")
    parser.add_argument("--battery-model", type=str, default="simple", help="Battery model for oracle reference.")
    parser.add_argument("--reward-profile", type=str, default="network", help="Reward profile passed to the oracle script.")
    parser.add_argument("--year", type=int, default=2024, help="Calendar year used for window extraction.")
    parser.add_argument("--seed", type=int, default=42, help="Seed passed to the oracle script.")
    parser.add_argument(
        "--window-days-list",
        type=str,
        default="30,365",
        help="Comma-separated oracle window lengths in days. Maximum supported window is 365 days.",
    )
    parser.add_argument(
        "--offset-days-list",
        type=str,
        default="0,91,182,273",
        help="Comma-separated start-day offsets within the selected year.",
    )
    parser.add_argument(
        "--max-window-days",
        type=int,
        default=365,
        help="Hard upper bound for generated oracle windows. Defaults to 365 days.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/oracle_reference_suite"),
        help="Output directory for per-window runs and the combined oracle reference CSV.",
    )
    return parser


def _parse_int_csv(raw: str) -> list[int]:
    values = [int(token.strip()) for token in str(raw).split(",") if token.strip()]
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def build_window_jobs(window_days_list: list[int], offset_days_list: list[int], *, max_window_days: int = 365) -> list[dict[str, int]]:
    jobs: list[dict[str, int]] = []
    seen: set[tuple[int, int]] = set()
    for window_days in window_days_list:
        if window_days <= 0:
            raise ValueError(f"Window days must be positive, got {window_days}.")
        if window_days > int(max_window_days):
            raise ValueError(f"Window days {window_days} exceed max_window_days={max_window_days}.")
        candidate_offsets = [0] if int(window_days) >= int(max_window_days) else offset_days_list
        for offset_days in candidate_offsets:
            if offset_days < 0:
                raise ValueError(f"Offset days must be non-negative, got {offset_days}.")
            if offset_days + window_days > int(max_window_days):
                continue
            key = (int(window_days), int(offset_days))
            if key in seen:
                continue
            seen.add(key)
            jobs.append({"window_days": int(window_days), "offset_days_within_year": int(offset_days)})
    jobs.sort(key=lambda item: (item["window_days"], item["offset_days_within_year"]))
    return jobs


def run_oracle_job(
    *,
    cases: str,
    regimes: str,
    battery_model: str,
    reward_profile: str,
    year: int,
    seed: int,
    window_days: int,
    offset_days_within_year: int,
    output_dir: Path,
) -> Path:
    job_output_dir = output_dir / f"{int(window_days)}d_off{int(offset_days_within_year)}"
    job_output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "analysis" / "full_year_oracle_compare.py"),
        "--cases",
        str(cases),
        "--regimes",
        str(regimes),
        "--battery-model",
        str(battery_model),
        "--reward-profile",
        str(reward_profile),
        "--days",
        str(int(window_days)),
        "--year",
        str(int(year)),
        "--offset-days-within-year",
        str(int(offset_days_within_year)),
        "--seed",
        str(int(seed)),
        "--output-dir",
        str(job_output_dir),
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    summary_csv = job_output_dir / "protocol_summary.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"Oracle protocol summary not produced: {summary_csv}")
    return summary_csv


def combine_oracle_summaries(summary_paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in summary_paths:
        df = pd.read_csv(path)
        if df.empty:
            continue
        parent_name = path.parent.name
        window_days = int(parent_name.split("d_off")[0])
        offset_days = int(parent_name.split("d_off")[1])
        df = df.copy()
        df["reference_window_days"] = int(window_days)
        df["reference_offset_days_within_year"] = int(offset_days)
        df["reference_source_dir"] = str(path.parent)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    ordered = [
        "protocol",
        "case",
        "case_key",
        "regime",
        "battery_model",
        "reference_window_days",
        "reference_offset_days_within_year",
        "none_objective",
        "oracle_objective",
        "oracle_savings_vs_none",
        "oracle_savings_pct_vs_none",
        "none_final_soc",
        "oracle_final_soc",
        "oracle_min_soc",
        "oracle_max_soc",
        "oracle_throughput_mwh",
        "profile_start_timestamp",
        "profile_end_timestamp",
        "reference_source_dir",
    ]
    front = [column for column in ordered if column in combined.columns]
    remainder = [column for column in combined.columns if column not in front]
    return combined[front + remainder]


def main() -> int:
    args = build_parser().parse_args(sys.argv[1:])
    output_dir = Path(args.output_dir)
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    jobs = build_window_jobs(
        _parse_int_csv(args.window_days_list),
        _parse_int_csv(args.offset_days_list),
        max_window_days=int(args.max_window_days),
    )
    summary_paths: list[Path] = []
    for job in jobs:
        print(
            "[oracle-suite] "
            f"days={job['window_days']} offset={job['offset_days_within_year']} "
            f"cases={args.cases} regimes={args.regimes}"
        )
        summary_paths.append(
            run_oracle_job(
                cases=str(args.cases),
                regimes=str(args.regimes),
                battery_model=str(args.battery_model),
                reward_profile=str(args.reward_profile),
                year=int(args.year),
                seed=int(args.seed),
                window_days=int(job["window_days"]),
                offset_days_within_year=int(job["offset_days_within_year"]),
                output_dir=runs_dir,
            )
        )

    combined_df = combine_oracle_summaries(summary_paths)
    combined_csv = output_dir / "oracle_reference_windows.csv"
    combined_json = output_dir / "oracle_reference_windows.json"
    combined_df.to_csv(combined_csv, index=False)
    combined_json.write_text(combined_df.to_json(orient="records", indent=2), encoding="utf-8")
    print(f"[oracle-suite] saved combined CSV: {combined_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
