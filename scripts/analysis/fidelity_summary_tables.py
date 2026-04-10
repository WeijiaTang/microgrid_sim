#!/usr/bin/env python3
"""Aggregate fidelity experiment summaries into paper-ready tables."""
# Ref: docs/spec/task.md (Task-ID: SPEC-FIDELITY-MISMATCH-001)

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate cross-fidelity summary CSV files into paper-ready tables.")
    parser.add_argument(
        "--input-dirs",
        type=str,
        required=True,
        help="Comma-separated result directories containing summary.csv files from short_cross_fidelity_probe.py",
    )
    parser.add_argument("--output-dir", type=str, default="results/fidelity_summary_tables", help="Directory for aggregated outputs")
    return parser


def _parse_csv_arg(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _load_summary_dirs(input_dirs: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for raw_dir in input_dirs:
        path = Path(raw_dir)
        summary_csv = path / "summary.csv"
        if not summary_csv.exists():
            raise FileNotFoundError(f"Missing summary.csv in '{path}'")
        frame = pd.read_csv(summary_csv)
        if frame.empty:
            continue
        frame["source_dir"] = str(path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _infer_reward_profile(summary_df: pd.DataFrame) -> pd.Series:
    if "reward_profile" in summary_df.columns:
        return summary_df["reward_profile"].fillna("unknown").astype(str)
    source_dir = summary_df.get("source_dir", pd.Series([""] * len(summary_df), index=summary_df.index)).astype(str).str.lower()
    inferred = pd.Series("unknown", index=summary_df.index, dtype="object")
    inferred = inferred.mask(source_dir.str.contains("paper_balanced"), "paper_balanced")
    inferred = inferred.mask(source_dir.str.contains("paper_aligned"), "paper_aligned")
    inferred = inferred.mask(source_dir.str.contains("network"), "network")
    return inferred


def _annotate_experiment_metadata(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df.copy()
    annotated = summary_df.copy()
    annotated["reward_profile"] = _infer_reward_profile(annotated)
    annotated["regularized_training"] = (
        (annotated.get("action_smoothing_coef", 0.0).astype(float) > 0.0)
        | (annotated.get("action_max_delta", 0.0).astype(float) > 0.0)
        | (annotated.get("action_rate_penalty", 0.0).astype(float) > 0.0)
        | (annotated.get("battery_feasibility_aware", 0).astype(int) > 0)
        | (annotated.get("battery_infeasible_penalty", 0.0).astype(float) > 0.0)
        | (annotated.get("symmetric_battery_action", 0).astype(int) > 0)
    ).astype(int)
    annotated["mixed_fidelity_training"] = annotated["train_model"].astype(str).str.contains(r"\+").astype(int)
    annotated["train_stage_count"] = annotated["train_model"].astype(str).str.count(r"\+") + 1
    return annotated


def _aggregate_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()
    id_cols = [
        "source_dir",
        "case",
        "regime",
        "reward_profile",
        "regularized_training",
        "mixed_fidelity_training",
        "train_stage_count",
        "agent",
        "train_model",
        "test_model",
        "train_steps",
        "eval_steps",
        "action_smoothing_coef",
        "action_max_delta",
        "action_rate_penalty",
        "battery_feasibility_aware",
        "battery_infeasible_penalty",
        "symmetric_battery_action",
        "mixed_fidelity_stage_fractions",
        "mixed_fidelity_stage_learning_rates",
    ]
    numeric_cols = [
        col
        for col in summary_df.columns
        if col not in set(id_cols + ["seed", "source_dir"])
        and pd.api.types.is_numeric_dtype(summary_df[col])
    ]
    aggregated = (
        summary_df.groupby(id_cols, dropna=False)[numeric_cols]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    aggregated.columns = [
        f"{col}_{stat}" if stat else str(col)
        for col, stat in aggregated.columns.to_flat_index()
    ]
    return aggregated


def _gap_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, float | int | str]] = []
    group_cols = [
        "source_dir",
        "case",
        "regime",
        "agent",
        "seed",
        "train_model",
        "train_steps",
        "eval_steps",
        "mixed_fidelity_stage_fractions",
        "mixed_fidelity_stage_learning_rates",
    ]
    gap_cols = [
        "final_cumulative_cost",
        "min_voltage_worst",
        "max_line_loading_peak",
        "max_line_current_peak_ka",
        "mean_grid_import_mw",
        "total_battery_loss_kwh",
        "final_temperature_c",
        "soc_upper_dwell_fraction",
        "soc_lower_dwell_fraction",
        "infeasible_action_dwell_fraction",
    ]
    for keys, group in summary_df.groupby(group_cols, dropna=False):
        by_test = {str(row["test_model"]): row for _, row in group.iterrows()}
        if "simple" not in by_test or "thevenin" not in by_test:
            continue
        simple_row = by_test["simple"]
        thevenin_row = by_test["thevenin"]
        row = {col: value for col, value in zip(group_cols, keys)}
        for column in gap_cols:
            row[f"{column}_gap"] = float(thevenin_row[column]) - float(simple_row[column])
        rows.append(row)
    gap_df = pd.DataFrame(rows)
    if gap_df.empty:
        return gap_df
    agg_cols = [col for col in gap_df.columns if col.endswith("_gap")]
    grouped = gap_df.groupby(["case", "regime", "train_model"], dropna=False)[agg_cols].agg(["mean", "std", "min", "max"]).reset_index()
    grouped.columns = [
        f"{col}_{stat}" if stat else str(col)
        for col, stat in grouped.columns.to_flat_index()
    ]
    return grouped


def _paper_key_metrics_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()
    group_cols = [
        "source_dir",
        "case",
        "regime",
        "reward_profile",
        "regularized_training",
        "mixed_fidelity_training",
        "train_stage_count",
        "train_model",
        "test_model",
        "train_steps",
        "eval_steps",
    ]
    metric_cols = [
        "final_cumulative_cost",
        "final_soc",
        "min_voltage_worst",
        "max_line_loading_peak",
        "mean_grid_import_mw",
        "total_battery_loss_kwh",
        "total_battery_stress_kwh",
        "total_battery_throughput_kwh",
        "final_temperature_c",
        "soc_lower_dwell_fraction",
        "infeasible_action_dwell_fraction",
    ]
    paper_df = summary_df.groupby(group_cols, dropna=False)[metric_cols].mean().reset_index()
    paper_df["cost_rank_within_test_env"] = (
        paper_df.groupby(["source_dir", "case", "regime", "test_model"], dropna=False)["final_cumulative_cost"]
        .rank(method="dense", ascending=True)
        .astype(int)
    )
    return paper_df.sort_values(
        by=["source_dir", "case", "regime", "test_model", "cost_rank_within_test_env", "train_model"],
        kind="stable",
    ).reset_index(drop=True)


def _best_train_family_table(paper_df: pd.DataFrame) -> pd.DataFrame:
    if paper_df.empty:
        return pd.DataFrame()
    selectors = ["source_dir", "case", "regime", "test_model"]
    best_idx = (
        paper_df.sort_values(
            by=[
                "source_dir",
                "case",
                "regime",
                "test_model",
                "final_cumulative_cost",
                "soc_lower_dwell_fraction",
                "infeasible_action_dwell_fraction",
                "train_model",
            ],
            kind="stable",
        )
        .groupby(selectors, dropna=False)
        .head(1)
        .index
    )
    best_df = paper_df.loc[best_idx].copy()
    return best_df.sort_values(by=selectors, kind="stable").reset_index(drop=True)


def main() -> int:
    args = build_parser().parse_args()
    input_dirs = _parse_csv_arg(args.input_dirs)
    summary_df = _annotate_experiment_metadata(_load_summary_dirs(input_dirs))
    aggregate_df = _aggregate_summary(summary_df)
    gap_df = _gap_table(summary_df)
    paper_df = _paper_key_metrics_table(summary_df)
    best_df = _best_train_family_table(paper_df)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_csv = output_dir / "combined_detail.csv"
    aggregate_csv = output_dir / "aggregate_summary.csv"
    gap_csv = output_dir / "test_model_gap_summary.csv"
    paper_csv = output_dir / "paper_key_metrics.csv"
    best_csv = output_dir / "best_train_family_by_test_env.csv"
    summary_json = output_dir / "summary.json"

    summary_df.to_csv(detail_csv, index=False)
    aggregate_df.to_csv(aggregate_csv, index=False)
    gap_df.to_csv(gap_csv, index=False)
    paper_df.to_csv(paper_csv, index=False)
    best_df.to_csv(best_csv, index=False)
    summary_payload = {
        "input_dirs": input_dirs,
        "combined_rows": int(len(summary_df)),
        "aggregate_rows": int(len(aggregate_df)),
        "gap_rows": int(len(gap_df)),
        "paper_rows": int(len(paper_df)),
        "best_rows": int(len(best_df)),
    }
    summary_json.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print("\n=== Fidelity Summary Tables ===")
    if not aggregate_df.empty:
        print(aggregate_df.to_string(index=False))
    print(f"\nSaved combined detail CSV: {detail_csv}")
    print(f"Saved aggregate summary CSV: {aggregate_csv}")
    print(f"Saved test-model gap CSV: {gap_csv}")
    print(f"Saved paper key-metrics CSV: {paper_csv}")
    print(f"Saved best-train-family CSV: {best_csv}")
    print(f"Saved summary JSON: {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
