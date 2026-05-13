#!/usr/bin/env python3
"""Posthoc enrich/regroup existing safe-warmstart summaries with oracle-normalized value recovery."""
# Ref: docs/spec/task.md (Task-ID: SPEC-SAFE-OFFLINE-RL-001)

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.compare_safe_warmstart_sac import (  # noqa: E402
    _aggregate_summary,
    _attach_oracle_value_recovery,
    _parse_csv_arg,
    ensure_eval_window_columns,
    reorder_safe_warmstart_grouped,
    reorder_safe_warmstart_summary,
    write_storyline_audit,
    write_paper_main_tables,
    write_reviewer_grouped_tables,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Posthoc enrich existing summary.csv files with oracle-normalized value recovery "
            "and regenerate grouped reviewer-facing summaries."
        )
    )
    parser.add_argument(
        "result_dirs",
        nargs="+",
        help="One or more result directories containing summary.csv produced by compare_safe_warmstart_sac.py.",
    )
    parser.add_argument(
        "--oracle-reference-csv",
        type=str,
        default="",
        help="Optional oracle/none reference CSV used to compute value-recovery normalization.",
    )
    parser.add_argument(
        "--groupby-columns",
        type=str,
        default="case,regime,controller_variant,train_model,test_model",
        help="Grouping columns used when regenerating summary_grouped.csv.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Overwrite summary.csv / summary_grouped.csv in place. Default writes *_enriched files.",
    )
    return parser


def _target_paths(result_dir: Path, *, overwrite_existing: bool) -> dict[str, Path]:
    if overwrite_existing:
        return {
            "summary_csv": result_dir / "summary.csv",
            "summary_json": result_dir / "summary.json",
            "grouped_csv": result_dir / "summary_grouped.csv",
            "grouped_json": result_dir / "summary_grouped.json",
        }
    return {
        "summary_csv": result_dir / "summary_enriched.csv",
        "summary_json": result_dir / "summary_enriched.json",
        "grouped_csv": result_dir / "summary_grouped_enriched.csv",
        "grouped_json": result_dir / "summary_grouped_enriched.json",
    }


def enrich_result_dir(
    result_dir: Path,
    *,
    oracle_reference_csv: str,
    groupby_columns: list[str],
    overwrite_existing: bool,
) -> dict[str, str]:
    summary_csv = result_dir / "summary.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"summary.csv not found under {result_dir}")

    summary_df = pd.read_csv(summary_csv)
    summary_df = ensure_eval_window_columns(summary_df)
    summary_df = _attach_oracle_value_recovery(summary_df, reference_csv=oracle_reference_csv)
    summary_df = reorder_safe_warmstart_summary(summary_df)
    grouped_df = _aggregate_summary(summary_df, groupby_columns=groupby_columns)
    grouped_df = reorder_safe_warmstart_grouped(grouped_df)

    targets = _target_paths(result_dir, overwrite_existing=overwrite_existing)
    summary_df.to_csv(targets["summary_csv"], index=False)
    targets["summary_json"].write_text(
        json.dumps(summary_df.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )
    grouped_df.to_csv(targets["grouped_csv"], index=False)
    targets["grouped_json"].write_text(
        json.dumps(grouped_df.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )
    write_reviewer_grouped_tables(result_dir, grouped_df)
    write_paper_main_tables(result_dir, summary_df)
    write_storyline_audit(result_dir, summary_df)
    return {key: str(value) for key, value in targets.items()}


def main() -> int:
    args = build_parser().parse_args(sys.argv[1:])
    groupby_columns = _parse_csv_arg(args.groupby_columns)
    if not groupby_columns:
        raise ValueError("At least one groupby column must be provided.")

    for raw_dir in args.result_dirs:
        result_dir = Path(raw_dir)
        outputs = enrich_result_dir(
            result_dir,
            oracle_reference_csv=str(args.oracle_reference_csv),
            groupby_columns=groupby_columns,
            overwrite_existing=bool(args.overwrite_existing),
        )
        print(f"[enrich] result_dir={result_dir}")
        print(f"  summary_csv={outputs['summary_csv']}")
        print(f"  grouped_csv={outputs['grouped_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
