#!/usr/bin/env python3
"""Combine per-result paper tables into consolidated reviewer-facing method tables."""
# Ref: docs/spec/task.md (Task-ID: SPEC-SAFE-OFFLINE-RL-001)

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Combine paper_main_* tables from multiple result directories.")
    parser.add_argument("result_dirs", nargs="+", help="Result directories containing paper_main_value_recovery.csv")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for the consolidated tables.",
    )
    return parser


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Expected table not found: {path}")
    return pd.read_csv(path)


def _read_optional_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    value_frames: list[pd.DataFrame] = []
    behavior_frames: list[pd.DataFrame] = []
    audit_frames: list[pd.DataFrame] = []
    for raw_dir in args.result_dirs:
        result_dir = Path(raw_dir)
        value_df = _read_table(result_dir / "paper_main_value_recovery.csv").copy()
        behavior_df = _read_table(result_dir / "paper_main_battery_behavior.csv").copy()
        audit_df = _read_optional_table(result_dir / "storyline_audit.csv").copy()
        value_df["source_result_dir"] = str(result_dir)
        behavior_df["source_result_dir"] = str(result_dir)
        if not audit_df.empty:
            audit_df["source_result_dir"] = str(result_dir)
        value_frames.append(value_df)
        behavior_frames.append(behavior_df)
        if not audit_df.empty:
            audit_frames.append(audit_df)

    combined_value = pd.concat(value_frames, ignore_index=True) if value_frames else pd.DataFrame()
    combined_behavior = pd.concat(behavior_frames, ignore_index=True) if behavior_frames else pd.DataFrame()
    combined_audit = pd.concat(audit_frames, ignore_index=True) if audit_frames else pd.DataFrame()

    value_csv = output_dir / "paper_main_value_recovery_combined.csv"
    value_json = output_dir / "paper_main_value_recovery_combined.json"
    behavior_csv = output_dir / "paper_main_battery_behavior_combined.csv"
    behavior_json = output_dir / "paper_main_battery_behavior_combined.json"
    audit_csv = output_dir / "storyline_audit_combined.csv"
    audit_json = output_dir / "storyline_audit_combined.json"
    combined_value.to_csv(value_csv, index=False)
    value_json.write_text(json.dumps(combined_value.to_dict(orient="records"), indent=2), encoding="utf-8")
    combined_behavior.to_csv(behavior_csv, index=False)
    behavior_json.write_text(json.dumps(combined_behavior.to_dict(orient="records"), indent=2), encoding="utf-8")
    combined_audit.to_csv(audit_csv, index=False)
    audit_json.write_text(json.dumps(combined_audit.to_dict(orient="records"), indent=2), encoding="utf-8")

    print(f"Saved combined value-recovery table: {value_csv}")
    print(f"Saved combined battery-behavior table: {behavior_csv}")
    print(f"Saved combined storyline audit: {audit_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
