#!/usr/bin/env python3
"""Generate regime-level fidelity gap summaries for network microgrid cases."""
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

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from short_cross_fidelity_probe import (  # noqa: E402
    _parse_csv_arg,
    _parse_seed_list,
    action_regularization_config,
    evaluate_agent,
    resolve_training_schedule,
    train_short_agent,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Regime-level fidelity gap mapper for network microgrids.")
    parser.add_argument("--cases", type=str, default="cigre,ieee33", help="Comma-separated case keys: cigre, ieee33")
    parser.add_argument(
        "--regimes",
        type=str,
        default="base,high_load,high_pv,network_stress,tight_soc",
        help="Comma-separated operating regimes",
    )
    parser.add_argument("--train-models", type=str, default="simple,thevenin", help="Comma-separated battery models for training")
    parser.add_argument("--test-models", type=str, default="simple,thevenin", help="Comma-separated battery models for evaluation")
    parser.add_argument("--reward-profile", type=str, default="network", help="Reward profile: network, paper_aligned, or paper_balanced")
    parser.add_argument("--agent", type=str, default="sac", help="SB3 agent name")
    parser.add_argument("--train-steps", type=int, default=500, help="Training horizon per agent")
    parser.add_argument("--eval-steps", type=int, default=96, help="Evaluation rollout steps")
    parser.add_argument("--days", type=int, default=2, help="Environment simulation days")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--seeds", type=str, default="", help="Optional comma-separated seed list overriding --seed")
    parser.add_argument("--device", type=str, default="cpu", help="Training device")
    parser.add_argument("--action-smoothing-coef", type=float, default=0.0, help="Exponential smoothing coefficient for continuous actions")
    parser.add_argument("--action-max-delta", type=float, default=0.0, help="Per-step maximum action delta before clipping")
    parser.add_argument("--action-rate-penalty", type=float, default=0.0, help="Penalty weight for applied action-rate changes")
    parser.add_argument("--battery-feasibility-aware", action="store_true", help="Clip battery actions to the current SOC-feasible range before env.step")
    parser.add_argument("--battery-infeasible-penalty", type=float, default=0.0, help="Penalty weight for requesting battery actions outside the current SOC-feasible range")
    parser.add_argument(
        "--symmetric-battery-action",
        action="store_true",
        help="Scale positive battery actions to enforce symmetric usable charge/discharge range",
    )
    parser.add_argument("--output-dir", type=str, default="results/fidelity_regime_map", help="Output directory")
    return parser


def _gap_summary_from_detail(detail_df: pd.DataFrame) -> pd.DataFrame:
    if detail_df.empty:
        return pd.DataFrame()
    numeric_gap_columns = [
        "final_cumulative_cost",
        "min_voltage_worst",
        "max_line_loading_peak",
        "max_line_current_peak_ka",
        "mean_grid_import_mw",
        "total_battery_loss_kwh",
        "total_battery_stress_kwh",
        "total_battery_throughput_kwh",
        "soc_upper_dwell_fraction",
        "soc_lower_dwell_fraction",
        "infeasible_action_dwell_fraction",
        "power_flow_failure_steps",
    ]
    group_cols = ["case", "regime", "agent", "seed", "train_model"]
    rows: list[dict[str, float | int | str]] = []
    for keys, group in detail_df.groupby(group_cols, dropna=False):
        by_test = {str(row["test_model"]): row for _, row in group.iterrows()}
        reference_model = None
        for candidate in ("thevenin", "thevenin_loss_only"):
            if candidate in by_test:
                reference_model = candidate
                break
        if "simple" not in by_test or reference_model is None:
            continue
        simple_row = by_test["simple"]
        reference_row = by_test[reference_model]
        row = {col: value for col, value in zip(group_cols, keys)}
        row["comparison"] = f"{reference_model}_minus_simple"
        for column in numeric_gap_columns:
            row[f"{column}_gap"] = float(reference_row[column]) - float(simple_row[column])
        rows.append(row)
    gap_df = pd.DataFrame(rows)
    if gap_df.empty:
        return gap_df
    agg_cols = [col for col in gap_df.columns if col.endswith("_gap")]
    summary = gap_df.groupby(["case", "regime", "train_model"], as_index=False)[agg_cols].mean(numeric_only=True)
    return summary


def main() -> int:
    args = build_parser().parse_args()
    case_keys = _parse_csv_arg(args.cases)
    regimes = _parse_csv_arg(args.regimes)
    train_models = _parse_csv_arg(args.train_models)
    test_models = _parse_csv_arg(args.test_models)
    seeds = _parse_seed_list(args.seeds, args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detail_rows: list[dict[str, float | int | str]] = []
    for seed in seeds:
        run_args = argparse.Namespace(**{**vars(args), "seed": int(seed)})
        for case_key in case_keys:
            for regime in regimes:
                for train_model in train_models:
                    print(f"[train] case={case_key} regime={regime} model={train_model} seed={seed} steps={args.train_steps}")
                    agent, train_schedule = train_short_agent(case_key=case_key, train_model=train_model, regime=regime, args=run_args)
                    for test_model in test_models:
                        print(f"[eval] case={case_key} regime={regime} train={train_model} test={test_model} seed={seed}")
                        summary, _ = evaluate_agent(agent, case_key=case_key, test_model=test_model, regime=regime, args=run_args)
                        detail_rows.append(
                            {
                                "case": case_key,
                                "regime": regime,
                                "agent": str(args.agent),
                                "seed": int(seed),
                                "train_model": train_model,
                                "test_model": test_model,
                                "train_steps": int(args.train_steps),
                                "eval_steps": int(summary["steps"]),
                                "action_smoothing_coef": float(args.action_smoothing_coef),
                                "action_max_delta": float(args.action_max_delta),
                                "action_rate_penalty": float(args.action_rate_penalty),
                                "battery_feasibility_aware": int(bool(args.battery_feasibility_aware)),
                                "battery_infeasible_penalty": float(args.battery_infeasible_penalty),
                                "symmetric_battery_action": int(bool(args.symmetric_battery_action)),
                                "resolved_train_stages": ",".join(str(stage) for stage in train_schedule["stages"]),
                                "resolved_train_stage_count": int(train_schedule["stage_count"]),
                                "resolved_train_stage_fractions": ",".join(f"{float(value):.6f}" for value in train_schedule["stage_fractions"]),
                                "resolved_train_stage_steps": ",".join(str(int(value)) for value in train_schedule["stage_steps"]),
                                "resolved_train_stage_learning_rates": ",".join(f"{float(value):.8g}" for value in train_schedule["stage_learning_rates"]),
                                **summary,
                            }
                        )

    detail_df = pd.DataFrame(detail_rows)
    gap_df = _gap_summary_from_detail(detail_df)

    detail_csv = output_dir / "detail.csv"
    summary_csv = output_dir / "summary.csv"
    summary_json = output_dir / "summary.json"
    detail_df.to_csv(detail_csv, index=False)
    gap_df.to_csv(summary_csv, index=False)
    summary_payload = {
        "action_regularization": action_regularization_config(args),
        "summary": gap_df.to_dict(orient="records"),
    }
    summary_json.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print("\n=== Fidelity Regime Gap Summary ===")
    print(gap_df.to_string(index=False))
    print(f"\nSaved detail CSV: {detail_csv}")
    print(f"Saved summary CSV: {summary_csv}")
    print(f"Saved summary JSON: {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
