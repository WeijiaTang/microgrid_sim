#!/usr/bin/env python3
"""Compare plain SAC, BC-warmstart SAC, and shielded SAC on the network-first microgrid cases."""
# Ref: docs/spec/task.md (Task-ID: SPEC-SAFE-OFFLINE-RL-001)

from __future__ import annotations

import argparse
import json
import re
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

import scripts.analysis.short_cross_fidelity_probe as probe


SUPPORTED_CONTROLLER_VARIANTS = (
    "plain_sac",
    "replay_warmstart_sac",
    "bc_warmstart_sac",
    "shielded_sac",
    "shielded_replay_warmstart_sac",
    "shielded_bc_warmstart_sac",
)

REVIEWER_VALUE_RECOVERY_PASS_THRESHOLD = 0.0
REVIEWER_MORPHOLOGY_MIDBAND_DWELL_THRESHOLD = 0.05
REVIEWER_MORPHOLOGY_PEAK_DISCHARGE_THRESHOLD = 0.20
REVIEWER_MORPHOLOGY_VALLEY_CHARGE_THRESHOLD = 0.20
REVIEWER_SHIELD_MEAN_DELTA_THRESHOLD = 0.05
REVIEWER_SHIELD_MATERIAL_DWELL_THRESHOLD = 0.60
REVIEWER_SHIELD_STRONG_DWELL_THRESHOLD = 0.20

CASE_ALIASES = {
    "ieee33": {"ieee33", "ieee33network", "ieee33bus", "ieee33-bus", "ieee 33-bus"},
    "cigre": {"cigre", "cigreeuropeanlv", "cigreeulvnetwork", "cigre_eu_lv_network", "cigre european lv"},
}

SAFE_WARMSTART_SUMMARY_PRIORITY_COLUMNS = [
    "case",
    "regime",
    "controller_variant",
    "train_model",
    "test_model",
    "seed",
    "eval_window_label",
    "eval_window_family",
    "eval_window_days",
    "eval_offset_days_within_year",
    "reward_profile",
    "agent",
    "train_steps",
    "eval_steps",
    "final_cumulative_objective_cost",
    "objective_none_cost",
    "objective_oracle_cost",
    "objective_savings_vs_none",
    "objective_oracle_savings_vs_none",
    "objective_recovery_fraction_vs_oracle_display",
    "oracle_normalized_objective_value_recovery_display",
    "objective_recovery_fraction_vs_oracle",
    "objective_recovery_fraction_vs_oracle_raw",
    "value_recovery_pass",
    "oracle_normalized_objective_value_recovery",
    "oracle_normalized_objective_value_recovery_raw",
    "objective_gap_to_oracle",
    "final_cumulative_cost",
    "raw_none_cost",
    "raw_oracle_cost",
    "raw_savings_vs_none",
    "raw_oracle_savings_vs_none",
    "raw_recovery_fraction_vs_oracle",
    "raw_gap_to_oracle",
    "final_soc",
    "total_terminal_soc_penalty",
    "final_terminal_soc_deviation",
    "soc_midband_dwell_fraction",
    "soc_target_tracking_mae",
    "soc_upper_dwell_fraction",
    "soc_lower_dwell_fraction",
    "soc_upper_parking_fraction",
    "soc_lower_parking_fraction",
    "peak_price_mean_discharge_limit_ratio",
    "valley_price_mean_charge_limit_ratio",
    "peak_price_discharge_action_fraction",
    "valley_price_charge_action_fraction",
    "morphology_midband_pass",
    "morphology_peak_pass",
    "morphology_valley_pass",
    "morphology_behavior_pass",
    "peak_price_low_discharge_limit_dwell_fraction",
    "infeasible_action_dwell_fraction",
    "total_battery_throughput_kwh",
    "total_battery_loss_kwh",
    "total_battery_stress_kwh",
    "mean_grid_import_mw",
    "shield_activation_fraction",
    "shield_boundary_activation_fraction",
    "shield_terminal_activation_fraction",
    "shield_reserve_activation_fraction",
    "mean_abs_shield_delta",
    "shield_material_activation_fraction",
    "shield_strong_activation_fraction",
    "shield_internalization_pass",
    "morphology_pass",
    "reviewer_ready_pass",
    "inventory_teacher_activation_fraction",
    "inventory_teacher_boundary_activation_fraction",
    "inventory_teacher_terminal_activation_fraction",
    "inventory_teacher_reserve_activation_fraction",
    "mean_abs_inventory_teacher_gap",
    "effective_online_safe_bc_gradient_steps",
    "last_validation_mean_soc_midband_dwell_fraction",
    "last_validation_mean_soc_target_tracking_mae",
    "last_validation_mean_peak_price_discharge_action_fraction",
    "last_validation_mean_valley_price_charge_action_fraction",
    "train_validation_metric",
    "validation_best_metric_value",
    "validation_best_total_reward",
    "validation_best_objective_cost",
    "validation_best_checkpoint_step",
    "shield_enabled",
    "oracle_reference_window_compatible",
    "oracle_reference_csv",
    "offline_dataset",
    "offline_dataset_controller_sources",
    "offline_dataset_max_transitions",
    "bc_pretrain_gradient_steps",
    "bc_pretrain_batch_size",
    "bc_pretrain_learning_rate",
    "shield_delta_penalty_coef",
    "online_safe_bc_gradient_steps",
    "online_safe_bc_batch_size",
    "online_safe_bc_max_samples",
    "online_safe_bc_learning_rate",
    "offline_bc_dataset_rows",
    "offline_bc_replay_seeded_transitions",
    "offline_bc_actor_gradient_steps",
    "offline_bc_actor_batch_size",
    "offline_bc_initial_actor_mse",
    "offline_bc_final_actor_mse",
    "learning_rate",
    "tensorboard_log_dir",
    "tensorboard_run_name",
    "action_smoothing_coef",
    "action_max_delta",
    "action_rate_penalty",
    "battery_feasibility_aware",
    "battery_infeasible_penalty",
    "symmetric_battery_action",
    "shield_soc_soft_buffer_fraction",
    "shield_soc_hard_buffer_fraction",
    "shield_peak_reserve_min_fraction",
    "shield_hard_pullback_action",
    "shield_terminal_closure_horizon_fraction",
    "shield_terminal_closure_urgency_soc",
    "rule_guidance_mix",
    "rule_guidance_decay_steps",
    "rule_guidance_policy",
    "protocol_profile",
    "train_window_start",
    "train_window_end",
    "eval_window_start",
    "eval_window_end",
]

SAFE_WARMSTART_GROUPED_PRIORITY_COLUMNS = [
    "case",
    "regime",
    "controller_variant",
    "train_model",
    "test_model",
    "eval_window_label",
    "eval_window_family",
    "eval_window_days",
    "eval_offset_days_within_year",
    "row_count",
    "mean_oracle_reference_window_compatible",
    "mean_final_cumulative_objective_cost",
    "mean_objective_none_cost",
    "mean_objective_oracle_cost",
    "mean_objective_savings_vs_none",
    "mean_objective_oracle_savings_vs_none",
    "mean_objective_recovery_fraction_vs_oracle_display",
    "mean_oracle_normalized_objective_value_recovery_display",
    "mean_objective_recovery_fraction_vs_oracle",
    "mean_objective_recovery_fraction_vs_oracle_raw",
    "mean_value_recovery_pass",
    "mean_oracle_normalized_objective_value_recovery",
    "mean_oracle_normalized_objective_value_recovery_raw",
    "mean_objective_gap_to_oracle",
    "mean_final_cumulative_cost",
    "mean_raw_savings_vs_none",
    "mean_raw_oracle_savings_vs_none",
    "mean_raw_recovery_fraction_vs_oracle",
    "mean_raw_gap_to_oracle",
    "mean_final_soc",
    "mean_total_terminal_soc_penalty",
    "mean_soc_midband_dwell_fraction",
    "mean_soc_target_tracking_mae",
    "mean_soc_upper_dwell_fraction",
    "mean_soc_lower_dwell_fraction",
    "mean_soc_upper_parking_fraction",
    "mean_soc_lower_parking_fraction",
    "mean_boundary_parking_fraction",
    "mean_peak_price_mean_discharge_limit_ratio",
    "mean_valley_price_mean_charge_limit_ratio",
    "mean_peak_price_discharge_action_fraction",
    "mean_valley_price_charge_action_fraction",
    "mean_morphology_midband_pass",
    "mean_morphology_peak_pass",
    "mean_morphology_valley_pass",
    "mean_morphology_behavior_pass",
    "mean_peak_price_low_discharge_limit_dwell_fraction",
    "mean_infeasible_action_dwell_fraction",
    "mean_shield_activation_fraction",
    "mean_abs_shield_delta",
    "mean_shield_material_activation_fraction",
    "mean_shield_strong_activation_fraction",
    "mean_shield_internalization_pass",
    "mean_morphology_pass",
    "mean_reviewer_ready_pass",
    "mean_inventory_teacher_activation_fraction",
    "mean_inventory_teacher_boundary_activation_fraction",
    "mean_inventory_teacher_terminal_activation_fraction",
    "mean_inventory_teacher_reserve_activation_fraction",
    "mean_abs_inventory_teacher_gap",
    "mean_effective_online_safe_bc_gradient_steps",
    "mean_last_validation_soc_midband_dwell_fraction",
    "mean_last_validation_soc_target_tracking_mae",
    "mean_last_validation_peak_price_discharge_action_fraction",
    "mean_last_validation_valley_price_charge_action_fraction",
]

ORACLE_VALUE_RECOVERY_COLUMNS = [
    "oracle_reference_csv",
    "oracle_reference_window_compatible",
    "objective_none_cost",
    "objective_oracle_cost",
    "objective_savings_vs_none",
    "objective_oracle_savings_vs_none",
    "objective_recovery_fraction_vs_oracle_display",
    "oracle_normalized_objective_value_recovery_display",
    "objective_recovery_fraction_vs_oracle",
    "objective_recovery_fraction_vs_oracle_raw",
    "oracle_normalized_objective_value_recovery",
    "oracle_normalized_objective_value_recovery_raw",
    "objective_gap_to_oracle",
    "raw_none_cost",
    "raw_oracle_cost",
    "raw_savings_vs_none",
    "raw_oracle_savings_vs_none",
    "raw_recovery_fraction_vs_oracle",
    "raw_gap_to_oracle",
]

REVIEWER_VALUE_RECOVERY_GROUPED_COLUMNS = [
    "case",
    "regime",
    "controller_variant",
    "train_model",
    "test_model",
    "eval_window_label",
    "eval_window_family",
    "eval_window_days",
    "eval_offset_days_within_year",
    "row_count",
    "mean_oracle_reference_window_compatible",
    "mean_final_cumulative_objective_cost",
    "mean_objective_none_cost",
    "mean_objective_oracle_cost",
    "mean_objective_savings_vs_none",
    "mean_objective_oracle_savings_vs_none",
    "mean_objective_recovery_fraction_vs_oracle",
    "mean_value_recovery_pass",
    "mean_oracle_normalized_objective_value_recovery",
    "mean_objective_gap_to_oracle",
    "mean_final_cumulative_cost",
    "mean_raw_savings_vs_none",
    "mean_raw_oracle_savings_vs_none",
    "mean_raw_recovery_fraction_vs_oracle",
    "mean_raw_gap_to_oracle",
]

REVIEWER_BATTERY_BEHAVIOR_GROUPED_COLUMNS = [
    "case",
    "regime",
    "controller_variant",
    "train_model",
    "test_model",
    "eval_window_label",
    "eval_window_family",
    "eval_window_days",
    "eval_offset_days_within_year",
    "row_count",
    "mean_final_soc",
    "mean_total_terminal_soc_penalty",
    "mean_soc_midband_dwell_fraction",
    "mean_soc_target_tracking_mae",
    "mean_soc_upper_dwell_fraction",
    "mean_soc_lower_dwell_fraction",
    "mean_soc_upper_parking_fraction",
    "mean_soc_lower_parking_fraction",
    "mean_boundary_parking_fraction",
    "mean_peak_price_mean_discharge_limit_ratio",
    "mean_valley_price_mean_charge_limit_ratio",
    "mean_peak_price_discharge_action_fraction",
    "mean_valley_price_charge_action_fraction",
    "mean_morphology_behavior_pass",
    "mean_peak_price_low_discharge_limit_dwell_fraction",
    "mean_infeasible_action_dwell_fraction",
    "mean_shield_activation_fraction",
    "mean_abs_shield_delta",
    "mean_shield_material_activation_fraction",
    "mean_shield_strong_activation_fraction",
    "mean_shield_internalization_pass",
    "mean_morphology_pass",
    "mean_reviewer_ready_pass",
    "mean_inventory_teacher_activation_fraction",
    "mean_inventory_teacher_boundary_activation_fraction",
    "mean_inventory_teacher_terminal_activation_fraction",
    "mean_inventory_teacher_reserve_activation_fraction",
    "mean_abs_inventory_teacher_gap",
    "mean_effective_online_safe_bc_gradient_steps",
    "mean_last_validation_soc_midband_dwell_fraction",
    "mean_last_validation_soc_target_tracking_mae",
    "mean_last_validation_peak_price_discharge_action_fraction",
    "mean_last_validation_valley_price_charge_action_fraction",
]

PAPER_MAIN_VALUE_PRIORITY_COLUMNS = [
    "case",
    "regime",
    "controller_variant",
    "train_model",
    "test_model",
    "eval_window_family",
    "window_count",
    "row_count",
    "mean_oracle_reference_window_compatible",
    "mean_final_cumulative_objective_cost",
    "mean_objective_none_cost",
    "mean_objective_oracle_cost",
    "mean_objective_savings_vs_none",
    "mean_objective_oracle_savings_vs_none",
    "mean_objective_recovery_fraction_vs_oracle_display",
    "mean_oracle_normalized_objective_value_recovery_display",
    "mean_objective_recovery_fraction_vs_oracle",
    "mean_objective_recovery_fraction_vs_oracle_raw",
    "mean_value_recovery_pass",
    "mean_oracle_normalized_objective_value_recovery",
    "mean_oracle_normalized_objective_value_recovery_raw",
    "mean_objective_gap_to_oracle",
    "mean_final_cumulative_cost",
]

PAPER_MAIN_BEHAVIOR_PRIORITY_COLUMNS = [
    "case",
    "regime",
    "controller_variant",
    "train_model",
    "test_model",
    "eval_window_family",
    "window_count",
    "row_count",
    "mean_final_soc",
    "mean_total_terminal_soc_penalty",
    "mean_soc_midband_dwell_fraction",
    "mean_soc_target_tracking_mae",
    "mean_soc_upper_dwell_fraction",
    "mean_soc_lower_dwell_fraction",
    "mean_soc_upper_parking_fraction",
    "mean_soc_lower_parking_fraction",
    "mean_boundary_parking_fraction",
    "mean_peak_price_mean_discharge_limit_ratio",
    "mean_valley_price_mean_charge_limit_ratio",
    "mean_peak_price_discharge_action_fraction",
    "mean_valley_price_charge_action_fraction",
    "mean_morphology_behavior_pass",
    "mean_peak_price_low_discharge_limit_dwell_fraction",
    "mean_infeasible_action_dwell_fraction",
    "mean_shield_activation_fraction",
    "mean_abs_shield_delta",
    "mean_shield_material_activation_fraction",
    "mean_shield_strong_activation_fraction",
    "mean_shield_internalization_pass",
    "mean_morphology_pass",
    "mean_reviewer_ready_pass",
]

STORYLINE_AUDIT_PRIORITY_COLUMNS = [
    "case",
    "regime",
    "controller_variant",
    "train_model",
    "test_model",
    "eval_window_family",
    "window_count",
    "row_count",
    "window_family_count",
    "total_eval_windows",
    "checkpoint_metric_set",
    "inventory_first_checkpoint",
    "window_coverage_status",
    "mean_oracle_normalized_objective_value_recovery",
    "value_recovery_status",
    "mean_objective_gap_to_oracle",
    "mean_final_soc",
    "inventory_health_status",
    "mean_boundary_parking_fraction",
    "mean_infeasible_action_dwell_fraction",
    "mean_shield_material_activation_fraction",
    "mean_shield_strong_activation_fraction",
    "protocol_dependence_status",
    "main_story_verdict",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reviewer-oriented safe/offline RL comparison on the network-first microgrid cases.",
        parents=[probe.build_parser()],
        add_help=False,
    )
    parser.set_defaults(
        cases="ieee33",
        output_dir="results/safe_warmstart_compare",
    )
    parser.add_argument(
        "--controller-variants",
        type=str,
        default="plain_sac,bc_warmstart_sac,shielded_sac",
        help=(
            "Comma-separated controller variants: plain_sac, replay_warmstart_sac, bc_warmstart_sac, "
            "shielded_sac, shielded_replay_warmstart_sac, shielded_bc_warmstart_sac"
        ),
    )
    parser.add_argument(
        "--groupby-columns",
        type=str,
        default="case,regime,controller_variant,train_model,test_model,eval_window_label",
        help="Comma-separated grouping columns used for aggregated comparison summaries.",
    )
    parser.add_argument(
        "--eval-window-days-list",
        type=str,
        default="",
        help="Optional comma-separated evaluation window lengths in days. Example: 30,365",
    )
    parser.add_argument(
        "--eval-offset-days-list",
        type=str,
        default="",
        help="Optional comma-separated evaluation start offsets within --eval-year. Example: 0,91,182,273",
    )
    parser.add_argument(
        "--oracle-reference-csv",
        type=str,
        default="",
        help=(
            "Optional CSV with none-vs-oracle reference values used to compute oracle-normalized value recovery. "
            "Supports full_year_oracle_compare protocol_summary.csv and metric-style protocol summaries."
        ),
    )
    parser.add_argument(
        "--network-oracle-reference-csv",
        type=str,
        default="",
        help=(
            "Optional network-replayed oracle reference CSV. When provided, or when a sibling "
            "network_replayed_oracle_reference_windows.csv exists next to --oracle-reference-csv, "
            "this same-objective reference is preferred over the lossless LP reference."
        ),
    )
    return parser


def _parse_csv_arg(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _preferred_oracle_reference_csv(reference_csv: str | Path, network_reference_csv: str | Path = "") -> str:
    """Prefer same-objective network-replayed references when available."""
    explicit = str(network_reference_csv).strip()
    if explicit:
        explicit_path = Path(explicit)
        if explicit_path.exists():
            return str(explicit_path)
        raise FileNotFoundError(f"Network oracle reference CSV not found: {explicit_path}")

    raw = str(reference_csv).strip()
    if not raw:
        return ""
    reference_path = Path(raw)
    candidates = [
        reference_path.with_name("network_replayed_oracle_reference_windows.csv"),
        reference_path.with_name(f"{reference_path.stem}_network_replayed{reference_path.suffix}"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(reference_path)


def _parse_controller_variants(raw: str) -> list[str]:
    variants = [item.strip().lower() for item in str(raw).split(",") if item.strip()]
    if not variants:
        raise ValueError("At least one controller variant must be provided.")
    unsupported = [variant for variant in variants if variant not in SUPPORTED_CONTROLLER_VARIANTS]
    if unsupported:
        raise ValueError(f"Unsupported controller variants {unsupported}; expected one of {SUPPORTED_CONTROLLER_VARIANTS}.")
    return variants


def _parse_int_csv_arg(raw: str) -> list[int]:
    values = [int(item.strip()) for item in str(raw).split(",") if item.strip()]
    return values


def _build_eval_window_specs(
    args: argparse.Namespace,
    *,
    case_key: str,
    regime: str,
) -> list[dict[str, Any]]:
    eval_year = int(getattr(args, "eval_year", 0))
    if eval_year <= 0:
        return [{"label": "default", "family": "default", "days": int(getattr(args, "days", 0)), "offset_days": 0, "window": None}]

    window_days_list = _parse_int_csv_arg(getattr(args, "eval_window_days_list", ""))
    offset_days_list = _parse_int_csv_arg(getattr(args, "eval_offset_days_list", ""))
    if not window_days_list:
        window_days = int(getattr(args, "eval_days", 0))
        if window_days <= 0:
            resolved = probe.resolve_eval_window(case_key=case_key, regime=regime, args=args)
            window_days = int(resolved["days"]) if resolved is not None else int(getattr(args, "days", 0))
        window_days_list = [int(window_days)]
    if not offset_days_list:
        offset_days_list = [int(getattr(args, "eval_offset_days_within_year", 0))]

    specs: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for window_days in window_days_list:
        candidate_offsets = [0] if int(window_days) >= 365 else offset_days_list
        for offset_days in candidate_offsets:
            key = (int(window_days), int(offset_days))
            if key in seen:
                continue
            seen.add(key)
            window_metadata = probe.resolve_window_metadata(
                case_key=case_key,
                regime=regime,
                reward_profile=str(args.reward_profile),
                seed=int(args.seed),
                year=int(eval_year),
                episode_days=int(window_days),
                random_start_within_year=False,
                stride_hours=int(getattr(args, "year_start_stride_hours", 24)),
                start_offset_days_within_year=int(offset_days),
            )
            family = f"{int(window_days)}d"
            label = f"{family}_off{int(offset_days)}"
            specs.append(
                {
                    "label": label,
                    "family": family,
                    "days": int(window_days),
                    "offset_days": int(offset_days),
                    "window": window_metadata,
                }
            )
    return specs


def _reorder_columns(df: pd.DataFrame, priority_columns: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    front = [column for column in priority_columns if column in df.columns]
    remainder = [column for column in df.columns if column not in front]
    return df[front + remainder]


def reorder_safe_warmstart_summary(df: pd.DataFrame) -> pd.DataFrame:
    return _reorder_columns(df, SAFE_WARMSTART_SUMMARY_PRIORITY_COLUMNS)


def reorder_safe_warmstart_grouped(df: pd.DataFrame) -> pd.DataFrame:
    return _reorder_columns(df, SAFE_WARMSTART_GROUPED_PRIORITY_COLUMNS)


def _artifact_stem(
    *,
    case_key: str,
    regime: str,
    agent: str,
    train_model: str,
    test_model: str,
    eval_window_label: str,
    seed: int,
) -> str:
    """Compact per-window artifact stem.

    The controller variant is intentionally omitted because artifacts already live under
    variant-specific subdirectories. Keeping it in both the directory and filename can
    exceed Windows MAX_PATH during reviewer-facing 20k runs.
    """
    return (
        f"{case_key}_{regime}_{agent}_"
        f"tr-{train_model}_te-{test_model}_{eval_window_label}_s{int(seed)}"
    )


def ensure_eval_window_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    work = df.copy()
    if "eval_window_days" not in work.columns:
        if "eval_config_days" in work.columns:
            work["eval_window_days"] = pd.to_numeric(work["eval_config_days"], errors="coerce")
        else:
            work["eval_window_days"] = np.nan
    if work["eval_window_days"].isna().any() and {"eval_window_start", "eval_window_end"}.issubset(work.columns):
        start_ts = pd.to_datetime(work["eval_window_start"], errors="coerce")
        end_ts = pd.to_datetime(work["eval_window_end"], errors="coerce")
        comparable = start_ts.notna() & end_ts.notna()
        inferred_days = (((end_ts - start_ts).dt.total_seconds() + 900.0) / 86400.0).round()
        work.loc[comparable & work["eval_window_days"].isna(), "eval_window_days"] = inferred_days.loc[
            comparable & work["eval_window_days"].isna()
        ]
    if "eval_offset_days_within_year" not in work.columns:
        offsets = pd.Series(np.nan, index=work.index, dtype=float)
        if {"eval_window_start", "eval_year"}.issubset(work.columns):
            start_ts = pd.to_datetime(work["eval_window_start"], errors="coerce")
            base_ts = pd.to_datetime(work["eval_year"].astype("Int64").astype(str) + "-01-01", errors="coerce")
            comparable = start_ts.notna() & base_ts.notna()
            offsets.loc[comparable] = (start_ts.loc[comparable] - base_ts.loc[comparable]).dt.days.astype(float)
        work["eval_offset_days_within_year"] = offsets
    if "eval_window_family" not in work.columns:
        days = pd.to_numeric(work["eval_window_days"], errors="coerce")
        work["eval_window_family"] = days.map(lambda value: f"{int(value)}d" if pd.notna(value) else "default")
    else:
        days = pd.to_numeric(work["eval_window_days"], errors="coerce")
        replace_mask = work["eval_window_family"].fillna("").astype(str).str.lower().eq("default") & days.notna()
        work.loc[replace_mask, "eval_window_family"] = days.loc[replace_mask].map(lambda value: f"{int(value)}d")
    if "eval_window_label" not in work.columns:
        labels: list[str] = []
        for _, row in work.iterrows():
            family = str(row.get("eval_window_family", "default"))
            offset = row.get("eval_offset_days_within_year", np.nan)
            if pd.notna(offset):
                labels.append(f"{family}_off{int(offset)}")
            else:
                labels.append(family)
        work["eval_window_label"] = labels
    else:
        replace_mask = work["eval_window_label"].fillna("").astype(str).str.lower().eq("default")
        if bool(replace_mask.any()):
            labels: list[str] = []
            for _, row in work.loc[replace_mask].iterrows():
                family = str(row.get("eval_window_family", "default"))
                offset = row.get("eval_offset_days_within_year", np.nan)
                if pd.notna(offset):
                    labels.append(f"{family}_off{int(offset)}")
                else:
                    labels.append(family)
            work.loc[replace_mask, "eval_window_label"] = labels
    return work


def build_reviewer_value_recovery_grouped(df: pd.DataFrame) -> pd.DataFrame:
    return _reorder_columns(df[[column for column in REVIEWER_VALUE_RECOVERY_GROUPED_COLUMNS if column in df.columns]], REVIEWER_VALUE_RECOVERY_GROUPED_COLUMNS)


def build_reviewer_battery_behavior_grouped(df: pd.DataFrame) -> pd.DataFrame:
    return _reorder_columns(df[[column for column in REVIEWER_BATTERY_BEHAVIOR_GROUPED_COLUMNS if column in df.columns]], REVIEWER_BATTERY_BEHAVIOR_GROUPED_COLUMNS)


def _aggregate_paper_main(summary_df: pd.DataFrame, *, metrics: list[str]) -> pd.DataFrame:
    if summary_df.empty or "eval_window_family" not in summary_df.columns:
        return pd.DataFrame()
    group_columns = ["case", "regime", "controller_variant", "train_model", "test_model", "eval_window_family"]
    agg_map = {metric: "mean" for metric in metrics if metric in summary_df.columns}
    if not agg_map:
        return pd.DataFrame()
    aggregated = summary_df.groupby(group_columns, dropna=False).agg(agg_map).reset_index()
    rename_map = {metric: (metric if metric.startswith("mean_") else f"mean_{metric}") for metric in agg_map}
    aggregated = aggregated.rename(columns=rename_map)
    row_counts = summary_df.groupby(group_columns, dropna=False).size().reset_index(name="row_count")
    window_counts = (
        summary_df.groupby(group_columns, dropna=False)["eval_window_label"]
        .nunique()
        .reset_index(name="window_count")
    )
    aggregated = aggregated.merge(row_counts, on=group_columns, how="left").merge(window_counts, on=group_columns, how="left")
    return aggregated


def build_paper_main_value_recovery(summary_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "oracle_reference_window_compatible",
        "final_cumulative_objective_cost",
        "objective_none_cost",
        "objective_oracle_cost",
        "objective_savings_vs_none",
        "objective_oracle_savings_vs_none",
        "objective_recovery_fraction_vs_oracle_display",
        "oracle_normalized_objective_value_recovery_display",
        "objective_recovery_fraction_vs_oracle",
        "objective_recovery_fraction_vs_oracle_raw",
        "value_recovery_pass",
        "oracle_normalized_objective_value_recovery",
        "oracle_normalized_objective_value_recovery_raw",
        "objective_gap_to_oracle",
        "final_cumulative_cost",
    ]
    aggregated = _aggregate_paper_main(summary_df, metrics=metrics)
    return _reorder_columns(aggregated, PAPER_MAIN_VALUE_PRIORITY_COLUMNS)


def build_paper_main_battery_behavior(summary_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "final_soc",
        "total_terminal_soc_penalty",
        "soc_midband_dwell_fraction",
        "soc_target_tracking_mae",
        "soc_upper_dwell_fraction",
        "soc_lower_dwell_fraction",
        "soc_upper_parking_fraction",
        "soc_lower_parking_fraction",
        "peak_price_mean_discharge_limit_ratio",
        "valley_price_mean_charge_limit_ratio",
        "peak_price_discharge_action_fraction",
        "valley_price_charge_action_fraction",
        "morphology_behavior_pass",
        "peak_price_low_discharge_limit_dwell_fraction",
        "infeasible_action_dwell_fraction",
        "shield_activation_fraction",
        "mean_abs_shield_delta",
        "shield_material_activation_fraction",
        "shield_strong_activation_fraction",
        "shield_internalization_pass",
        "morphology_pass",
        "reviewer_ready_pass",
    ]
    aggregated = _aggregate_paper_main(summary_df, metrics=metrics)
    if {"mean_soc_upper_parking_fraction", "mean_soc_lower_parking_fraction"}.issubset(aggregated.columns):
        aggregated["mean_boundary_parking_fraction"] = (
            aggregated["mean_soc_upper_parking_fraction"] + aggregated["mean_soc_lower_parking_fraction"]
        )
    return _reorder_columns(aggregated, PAPER_MAIN_BEHAVIOR_PRIORITY_COLUMNS)


def write_reviewer_grouped_tables(output_dir: Path, grouped_df: pd.DataFrame) -> dict[str, Path]:
    value_df = build_reviewer_value_recovery_grouped(grouped_df)
    behavior_df = build_reviewer_battery_behavior_grouped(grouped_df)
    outputs = {
        "value_csv": output_dir / "reviewer_value_recovery_grouped.csv",
        "value_json": output_dir / "reviewer_value_recovery_grouped.json",
        "behavior_csv": output_dir / "reviewer_battery_behavior_grouped.csv",
        "behavior_json": output_dir / "reviewer_battery_behavior_grouped.json",
    }
    value_df.to_csv(outputs["value_csv"], index=False)
    outputs["value_json"].write_text(json.dumps(value_df.to_dict(orient="records"), indent=2), encoding="utf-8")
    behavior_df.to_csv(outputs["behavior_csv"], index=False)
    outputs["behavior_json"].write_text(
        json.dumps(behavior_df.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )
    return outputs


def write_paper_main_tables(output_dir: Path, summary_df: pd.DataFrame) -> dict[str, Path]:
    value_df = build_paper_main_value_recovery(summary_df)
    behavior_df = build_paper_main_battery_behavior(summary_df)
    outputs = {
        "value_csv": output_dir / "paper_main_value_recovery.csv",
        "value_json": output_dir / "paper_main_value_recovery.json",
        "behavior_csv": output_dir / "paper_main_battery_behavior.csv",
        "behavior_json": output_dir / "paper_main_battery_behavior.json",
    }
    value_df.to_csv(outputs["value_csv"], index=False)
    outputs["value_json"].write_text(json.dumps(value_df.to_dict(orient="records"), indent=2), encoding="utf-8")
    behavior_df.to_csv(outputs["behavior_csv"], index=False)
    outputs["behavior_json"].write_text(json.dumps(behavior_df.to_dict(orient="records"), indent=2), encoding="utf-8")
    return outputs


def _classify_value_recovery(value: Any) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "unavailable"
    if float(numeric) >= 0.8:
        return "near_oracle"
    if float(numeric) >= 0.25:
        return "partial_positive"
    if float(numeric) >= 0.0:
        return "weak_positive"
    return "value_destructive"


def _classify_inventory_health(row: pd.Series) -> str:
    terminal_penalty = float(pd.to_numeric(pd.Series([row.get("mean_total_terminal_soc_penalty", np.nan)]), errors="coerce").iloc[0] or 0.0)
    boundary_parking = pd.to_numeric(pd.Series([row.get("mean_boundary_parking_fraction", np.nan)]), errors="coerce").iloc[0]
    if pd.isna(boundary_parking):
        upper_parking = pd.to_numeric(pd.Series([row.get("mean_soc_upper_parking_fraction", np.nan)]), errors="coerce").iloc[0]
        lower_parking = pd.to_numeric(pd.Series([row.get("mean_soc_lower_parking_fraction", np.nan)]), errors="coerce").iloc[0]
        if pd.notna(upper_parking) or pd.notna(lower_parking):
            boundary_parking = float(0.0 if pd.isna(upper_parking) else upper_parking) + float(
                0.0 if pd.isna(lower_parking) else lower_parking
            )
        else:
            upper_dwell = pd.to_numeric(pd.Series([row.get("mean_soc_upper_dwell_fraction", np.nan)]), errors="coerce").iloc[0]
            lower_dwell = pd.to_numeric(pd.Series([row.get("mean_soc_lower_dwell_fraction", np.nan)]), errors="coerce").iloc[0]
            if pd.notna(upper_dwell) or pd.notna(lower_dwell):
                boundary_parking = float(0.0 if pd.isna(upper_dwell) else upper_dwell) + float(
                    0.0 if pd.isna(lower_dwell) else lower_dwell
                )
    infeasible_dwell = pd.to_numeric(pd.Series([row.get("mean_infeasible_action_dwell_fraction", np.nan)]), errors="coerce").iloc[0]
    midband_dwell = pd.to_numeric(pd.Series([row.get("mean_soc_midband_dwell_fraction", np.nan)]), errors="coerce").iloc[0]
    headroom = pd.to_numeric(pd.Series([row.get("mean_peak_price_mean_discharge_limit_ratio", np.nan)]), errors="coerce").iloc[0]
    behavior_pass = pd.to_numeric(pd.Series([row.get("mean_morphology_behavior_pass", np.nan)]), errors="coerce").iloc[0]
    soc_tracking = pd.to_numeric(pd.Series([row.get("mean_soc_target_tracking_mae", np.nan)]), errors="coerce").iloc[0]

    boundary_clean = pd.notna(boundary_parking) and float(boundary_parking) <= 1e-9
    infeasible_clean = pd.notna(infeasible_dwell) and float(infeasible_dwell) <= 1e-9
    midband_good = pd.notna(midband_dwell) and float(midband_dwell) >= 0.40
    headroom_good = pd.notna(headroom) and float(headroom) >= 0.80
    behavior_mostly_present = pd.notna(behavior_pass) and float(behavior_pass) >= 0.75
    midband_improved = pd.notna(midband_dwell) and float(midband_dwell) >= 0.20
    tracking_acceptable = pd.isna(soc_tracking) or float(soc_tracking) <= 0.25

    if terminal_penalty <= 1e-9 and boundary_clean and infeasible_clean and midband_good and headroom_good:
        return "healthy"
    if boundary_clean and infeasible_clean and tracking_acceptable and (behavior_mostly_present or midband_improved):
        return "improved_but_fragile"
    return "pathological"


def _classify_protocol_dependence(row: pd.Series) -> str:
    material = pd.to_numeric(pd.Series([row.get("mean_shield_material_activation_fraction", np.nan)]), errors="coerce").iloc[0]
    strong = pd.to_numeric(pd.Series([row.get("mean_shield_strong_activation_fraction", np.nan)]), errors="coerce").iloc[0]
    mean_delta = pd.to_numeric(pd.Series([row.get("mean_abs_shield_delta", np.nan)]), errors="coerce").iloc[0]
    controller_variant = str(row.get("controller_variant", "")).strip().lower()
    if controller_variant == "plain_sac" and pd.isna(material) and pd.isna(strong) and pd.isna(mean_delta):
        return "not_applicable"
    if all(pd.isna(value) for value in (material, strong, mean_delta)):
        return "unavailable"
    material_value = float(material) if pd.notna(material) else 0.0
    strong_value = float(strong) if pd.notna(strong) else 0.0
    mean_delta_value = float(mean_delta) if pd.notna(mean_delta) else 0.0
    if material_value >= 0.50 or strong_value >= 0.10 or mean_delta_value >= 0.03:
        return "high_dependence"
    if material_value >= 0.20 or strong_value >= 0.02 or mean_delta_value >= 0.01:
        return "moderate_dependence"
    return "low_dependence"


def _classify_main_story_verdict(row: pd.Series) -> str:
    value_status = str(row.get("value_recovery_status", "unavailable"))
    inventory_status = str(row.get("inventory_health_status", "pathological"))
    story_supported_inventory = {"healthy", "improved_but_fragile", "healthy_clean", "constraint_clean"}
    behavior_clean_inventory = {"healthy", "healthy_clean", "constraint_clean"}
    if value_status in {"near_oracle", "partial_positive", "weak_positive"}:
        if inventory_status in story_supported_inventory:
            return "battery_story_supported"
        return "value_present_but_behavior_fragile"
    if value_status == "value_destructive":
        if inventory_status in behavior_clean_inventory:
            return "behavior_clean_but_value_unrecovered"
        return "not_paper_ready_value"
    return "evidence_incomplete"


def build_storyline_audit(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame(columns=STORYLINE_AUDIT_PRIORITY_COLUMNS)

    value_df = build_paper_main_value_recovery(summary_df)
    behavior_df = build_paper_main_battery_behavior(summary_df)
    group_columns = ["case", "regime", "controller_variant", "train_model", "test_model", "eval_window_family"]
    if value_df.empty and behavior_df.empty:
        return pd.DataFrame(columns=STORYLINE_AUDIT_PRIORITY_COLUMNS)
    if value_df.empty:
        audit_df = behavior_df.copy()
    elif behavior_df.empty:
        audit_df = value_df.copy()
    else:
        audit_df = value_df.merge(behavior_df, on=group_columns + ["window_count", "row_count"], how="outer")

    controller_group_columns = ["case", "regime", "controller_variant", "train_model", "test_model"]
    agg_map: dict[str, Any] = {
        "window_family_count": ("eval_window_family", "nunique"),
        "total_eval_windows": ("eval_window_label", "nunique"),
    }
    if "train_validation_metric" in summary_df.columns:
        agg_map["checkpoint_metric_set"] = (
            "train_validation_metric",
            lambda series: "|".join(sorted({str(value) for value in series if str(value).strip()})),
        )
    controller_meta = summary_df.groupby(controller_group_columns, dropna=False).agg(**agg_map).reset_index()
    if "checkpoint_metric_set" not in controller_meta.columns:
        controller_meta["checkpoint_metric_set"] = ""
    inventory_checkpoint_metrics = {"inventory_value_gate", "inventory_value_gate_shield"}
    controller_meta["inventory_first_checkpoint"] = controller_meta["checkpoint_metric_set"].apply(
        lambda value: int(
            bool({item for item in str(value).split("|") if item.strip()} & inventory_checkpoint_metrics)
        )
    )
    controller_meta["window_coverage_status"] = controller_meta.apply(
        lambda row: (
            "seasonal_plus_annual"
            if int(row.get("window_family_count", 0)) >= 2 and int(row.get("total_eval_windows", 0)) >= 5
            else "multiwindow_partial"
            if int(row.get("window_family_count", 0)) >= 2 or int(row.get("total_eval_windows", 0)) >= 2
            else "single_window"
        ),
        axis=1,
    )
    audit_df = audit_df.merge(controller_meta, on=controller_group_columns, how="left")
    audit_df["checkpoint_metric_set"] = audit_df["checkpoint_metric_set"].fillna("").astype(str)
    audit_df.loc[audit_df["checkpoint_metric_set"].str.len() == 0, "checkpoint_metric_set"] = "unknown"
    if "mean_boundary_parking_fraction" not in audit_df.columns and {
        "mean_soc_upper_parking_fraction",
        "mean_soc_lower_parking_fraction",
    }.issubset(audit_df.columns):
        audit_df["mean_boundary_parking_fraction"] = (
            audit_df["mean_soc_upper_parking_fraction"] + audit_df["mean_soc_lower_parking_fraction"]
        )
    audit_df["value_recovery_status"] = audit_df["mean_oracle_normalized_objective_value_recovery"].map(_classify_value_recovery)
    audit_df["inventory_health_status"] = audit_df.apply(_classify_inventory_health, axis=1)
    audit_df["protocol_dependence_status"] = audit_df.apply(_classify_protocol_dependence, axis=1)
    audit_df["main_story_verdict"] = audit_df.apply(_classify_main_story_verdict, axis=1)
    return _reorder_columns(audit_df, STORYLINE_AUDIT_PRIORITY_COLUMNS)


def write_storyline_audit(output_dir: Path, summary_df: pd.DataFrame) -> dict[str, Path]:
    audit_df = build_storyline_audit(summary_df)
    outputs = {
        "audit_csv": output_dir / "storyline_audit.csv",
        "audit_json": output_dir / "storyline_audit.json",
    }
    audit_df.to_csv(outputs["audit_csv"], index=False)
    outputs["audit_json"].write_text(json.dumps(audit_df.to_dict(orient="records"), indent=2), encoding="utf-8")
    return outputs


def _basic_normalize_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _normalize_token(value: Any) -> str:
    token = _basic_normalize_token(value)
    if not token:
        return ""
    for canonical, aliases in CASE_ALIASES.items():
        normalized_aliases = {_basic_normalize_token(alias) for alias in aliases}
        if token in normalized_aliases or token == canonical:
            return canonical
    return token


def _normalize_timestamp_token(value: Any) -> str:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    timestamp = pd.to_datetime(text, errors="coerce")
    if pd.isna(timestamp):
        return text
    return timestamp.isoformat()


def _coalesce_normalized_columns(
    df: pd.DataFrame,
    *,
    candidate_columns: list[str],
    normalizer,
) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=object)
    normalized_columns: list[pd.Series] = []
    for column in candidate_columns:
        if column not in df.columns:
            continue
        series = df[column].fillna("").astype(str).map(normalizer)
        series = series.where(series != "", pd.NA)
        normalized_columns.append(series.rename(column))
    if not normalized_columns:
        return pd.Series([""] * len(df), index=df.index, dtype=object)
    coalesced = pd.concat(normalized_columns, axis=1).bfill(axis=1).iloc[:, 0]
    return coalesced.fillna("").astype(str)


def _first_present(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def _load_oracle_reference(reference_csv: str | Path) -> pd.DataFrame:
    path = Path(reference_csv)
    if not path.exists():
        raise FileNotFoundError(f"Oracle reference CSV not found: {path}")
    reference_df = pd.read_csv(path)
    if reference_df.empty:
        return pd.DataFrame()

    base_join_columns = ["_case_norm", "_regime_norm", "_model_norm"]
    window_join_columns = ["_reference_window_start", "_reference_window_end"]
    work = reference_df.copy()
    work["_case_norm"] = _coalesce_normalized_columns(
        work,
        candidate_columns=["case_key", "case"],
        normalizer=_normalize_token,
    )
    work["_regime_norm"] = _coalesce_normalized_columns(
        work,
        candidate_columns=["regime"],
        normalizer=_normalize_token,
    )
    work["_model_norm"] = _coalesce_normalized_columns(
        work,
        candidate_columns=["battery_model", "test_model", "train_model"],
        normalizer=_normalize_token,
    )
    work["_reference_window_start"] = _coalesce_normalized_columns(
        work,
        candidate_columns=["profile_start_timestamp", "start_timestamp", "eval_window_start", "window_start_timestamp"],
        normalizer=_normalize_timestamp_token,
    )
    work["_reference_window_end"] = _coalesce_normalized_columns(
        work,
        candidate_columns=["profile_end_timestamp", "end_timestamp", "eval_window_end", "window_end_timestamp"],
        normalizer=_normalize_timestamp_token,
    )

    frames: list[pd.DataFrame] = []
    metadata_df = work[base_join_columns + window_join_columns].copy()
    frames.append(metadata_df)

    direct_objective_none = _first_present(work, ("none_objective", "none_objective_cost"))
    direct_objective_oracle = _first_present(work, ("oracle_objective", "oracle_objective_cost"))
    if direct_objective_none and direct_objective_oracle:
        direct_df = work[base_join_columns + window_join_columns + [direct_objective_none, direct_objective_oracle]].copy()
        direct_df = direct_df.rename(
            columns={
                direct_objective_none: "objective_none_cost",
                direct_objective_oracle: "objective_oracle_cost",
            }
        )
        frames.append(direct_df)

    direct_raw_none = _first_present(work, ("none_cost",))
    direct_raw_oracle = _first_present(work, ("oracle_cost",))
    if direct_raw_none and direct_raw_oracle and "metric" not in work.columns:
        direct_raw_df = work[base_join_columns + window_join_columns + [direct_raw_none, direct_raw_oracle]].copy()
        direct_raw_df = direct_raw_df.rename(
            columns={
                direct_raw_none: "raw_none_cost",
                direct_raw_oracle: "raw_oracle_cost",
            }
        )
        frames.append(direct_raw_df)

    if {"metric", "none_cost", "oracle_cost"}.issubset(work.columns):
        metric_series = work["metric"].fillna("").astype(str).str.strip().str.lower()
        objective_metric_df = work.loc[
            metric_series == "final_cumulative_objective_cost",
            base_join_columns + window_join_columns + ["none_cost", "oracle_cost"],
        ].copy()
        if not objective_metric_df.empty:
            objective_metric_df = objective_metric_df.rename(
                columns={
                    "none_cost": "objective_none_cost",
                    "oracle_cost": "objective_oracle_cost",
                }
            )
            frames.append(objective_metric_df)

        raw_metric_df = work.loc[
            metric_series == "final_cumulative_cost",
            base_join_columns + window_join_columns + ["none_cost", "oracle_cost"],
        ].copy()
        if not raw_metric_df.empty:
            raw_metric_df = raw_metric_df.rename(
                columns={
                    "none_cost": "raw_none_cost",
                    "oracle_cost": "raw_oracle_cost",
                }
            )
            frames.append(raw_metric_df)

    if not frames:
        raise ValueError(
            "Oracle reference CSV does not contain supported none/oracle columns. "
            "Expected none_objective/oracle_objective or metric+none_cost/oracle_cost."
        )

    merged_reference = frames[0]
    for frame in frames[1:]:
        merged_reference = merged_reference.merge(frame, on=base_join_columns + window_join_columns, how="outer")

    full_join_columns = base_join_columns + window_join_columns
    value_columns = [column for column in merged_reference.columns if column not in full_join_columns]
    metadata_columns = [column for column in value_columns if column.startswith("_reference_window_")]
    numeric_columns = [column for column in value_columns if column not in metadata_columns]
    for column in numeric_columns:
        merged_reference[column] = pd.to_numeric(merged_reference[column], errors="coerce")
    agg_map: dict[str, str] = {column: "mean" for column in numeric_columns}
    agg_map.update({column: "first" for column in metadata_columns})
    merged_reference = merged_reference.groupby(full_join_columns, dropna=False).agg(agg_map).reset_index()
    return merged_reference


def _attach_oracle_value_recovery(summary_df: pd.DataFrame, *, reference_csv: str) -> pd.DataFrame:
    if summary_df.empty or not str(reference_csv).strip():
        return summary_df

    reference_df = _load_oracle_reference(reference_csv)
    if reference_df.empty:
        return summary_df

    enriched = summary_df.copy()
    stale_oracle_columns = set(ORACLE_VALUE_RECOVERY_COLUMNS)
    for base_column in ORACLE_VALUE_RECOVERY_COLUMNS:
        stale_oracle_columns.add(f"{base_column}_x")
        stale_oracle_columns.add(f"{base_column}_y")
    enriched = enriched.drop(columns=list(stale_oracle_columns), errors="ignore")
    enriched["_case_norm"] = _coalesce_normalized_columns(
        enriched,
        candidate_columns=["case"],
        normalizer=_normalize_token,
    )
    enriched["_regime_norm"] = _coalesce_normalized_columns(
        enriched,
        candidate_columns=["regime"],
        normalizer=_normalize_token,
    )
    enriched["_model_norm"] = _coalesce_normalized_columns(
        enriched,
        candidate_columns=["test_model", "train_model"],
        normalizer=_normalize_token,
    )
    enriched["_window_start_norm"] = _coalesce_normalized_columns(
        enriched,
        candidate_columns=["eval_window_start"],
        normalizer=_normalize_timestamp_token,
    )
    enriched["_window_end_norm"] = _coalesce_normalized_columns(
        enriched,
        candidate_columns=["eval_window_end"],
        normalizer=_normalize_timestamp_token,
    )

    base_join_columns = ["_case_norm", "_regime_norm"]
    if bool(reference_df["_model_norm"].fillna("").astype(str).str.len().gt(0).any()):
        base_join_columns.append("_model_norm")

    value_columns = [
        column
        for column in (
            "objective_none_cost",
            "objective_oracle_cost",
            "raw_none_cost",
            "raw_oracle_cost",
        )
        if column in reference_df.columns
    ]
    reference_has_window = bool(
        reference_df["_reference_window_start"].fillna("").astype(str).str.len().gt(0).any()
        or reference_df["_reference_window_end"].fillna("").astype(str).str.len().gt(0).any()
    )
    enriched["oracle_reference_csv"] = str(Path(reference_csv))

    if reference_has_window:
        exact_reference = reference_df.rename(
            columns={
                "_reference_window_start": "_window_start_norm",
                "_reference_window_end": "_window_end_norm",
            }
        )
        exact_merge_columns = base_join_columns + ["_window_start_norm", "_window_end_norm"] + value_columns
        enriched = enriched.merge(exact_reference[exact_merge_columns], on=base_join_columns + ["_window_start_norm", "_window_end_norm"], how="left")
        base_presence = reference_df[base_join_columns].drop_duplicates().copy()
        base_presence["_oracle_reference_base_present"] = 1
        enriched = enriched.merge(base_presence, on=base_join_columns, how="left")
        matched_mask = pd.Series(False, index=enriched.index)
        for column in value_columns:
            matched_mask = matched_mask | enriched[column].notna()
        enriched["oracle_reference_window_compatible"] = np.where(
            matched_mask,
            1,
            np.where(enriched["_oracle_reference_base_present"].fillna(0).astype(int) == 1, 0, 0),
        )
        enriched = enriched.drop(columns=["_oracle_reference_base_present"], errors="ignore")
    else:
        merge_columns = base_join_columns + value_columns
        enriched = enriched.merge(reference_df[merge_columns], on=base_join_columns, how="left")
        matched_mask = pd.Series(False, index=enriched.index)
        for column in value_columns:
            matched_mask = matched_mask | enriched[column].notna()
        enriched["oracle_reference_window_compatible"] = matched_mask.astype(int)

    if {"objective_none_cost", "objective_oracle_cost", "final_cumulative_objective_cost"}.issubset(enriched.columns):
        enriched["objective_savings_vs_none"] = (
            enriched["objective_none_cost"] - enriched["final_cumulative_objective_cost"]
        )
        enriched["objective_oracle_savings_vs_none"] = (
            enriched["objective_none_cost"] - enriched["objective_oracle_cost"]
        )
        objective_denominator = enriched["objective_oracle_savings_vs_none"].astype(float)
        enriched["objective_recovery_fraction_vs_oracle"] = np.where(
            objective_denominator.abs() > 1e-9,
            enriched["objective_savings_vs_none"] / objective_denominator,
            np.nan,
        )
        enriched["objective_recovery_fraction_vs_oracle_raw"] = enriched["objective_recovery_fraction_vs_oracle"]
        enriched["objective_recovery_fraction_vs_oracle_display"] = np.clip(
            enriched["objective_recovery_fraction_vs_oracle"],
            0.0,
            1.0,
        )
        enriched["oracle_normalized_objective_value_recovery"] = enriched["objective_recovery_fraction_vs_oracle"]
        enriched["oracle_normalized_objective_value_recovery_raw"] = enriched["oracle_normalized_objective_value_recovery"]
        enriched["oracle_normalized_objective_value_recovery_display"] = np.clip(
            enriched["oracle_normalized_objective_value_recovery"],
            0.0,
            1.0,
        )
        enriched["objective_gap_to_oracle"] = (
            enriched["final_cumulative_objective_cost"] - enriched["objective_oracle_cost"]
        )

    if {"raw_none_cost", "raw_oracle_cost", "final_cumulative_cost"}.issubset(enriched.columns):
        enriched["raw_savings_vs_none"] = enriched["raw_none_cost"] - enriched["final_cumulative_cost"]
        enriched["raw_oracle_savings_vs_none"] = enriched["raw_none_cost"] - enriched["raw_oracle_cost"]
        raw_denominator = enriched["raw_oracle_savings_vs_none"].astype(float)
        enriched["raw_recovery_fraction_vs_oracle"] = np.where(
            raw_denominator.abs() > 1e-9,
            enriched["raw_savings_vs_none"] / raw_denominator,
            np.nan,
        )
        enriched["raw_gap_to_oracle"] = enriched["final_cumulative_cost"] - enriched["raw_oracle_cost"]

    return enriched.drop(
        columns=[
            "_case_norm",
            "_regime_norm",
            "_model_norm",
            "_window_start_norm",
            "_window_end_norm",
        ],
        errors="ignore",
    )


def _pass_fail_series(series: pd.Series, *, threshold: float, direction: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if direction == "ge":
        return numeric.ge(float(threshold)).fillna(False).astype(int)
    if direction == "gt":
        return numeric.gt(float(threshold)).fillna(False).astype(int)
    if direction == "le":
        return numeric.le(float(threshold)).fillna(False).astype(int)
    raise ValueError(f"Unsupported pass/fail direction '{direction}'.")


def attach_reviewer_pass_fail_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Attach reviewer-facing value and battery morphology pass/fail flags.

    These flags are intentionally stricter than "lowest objective wins": a row must recover
    positive oracle-normalized value and show healthy inventory morphology with low shield
    dependence before it is marked reviewer-ready.
    """
    if summary_df.empty:
        return summary_df
    work = summary_df.copy()

    zero = pd.Series(0, index=work.index)
    work["value_recovery_pass"] = _pass_fail_series(
        work.get("objective_recovery_fraction_vs_oracle", zero),
        threshold=REVIEWER_VALUE_RECOVERY_PASS_THRESHOLD,
        direction="gt",
    )
    work["morphology_midband_pass"] = _pass_fail_series(
        work.get("soc_midband_dwell_fraction", zero),
        threshold=REVIEWER_MORPHOLOGY_MIDBAND_DWELL_THRESHOLD,
        direction="ge",
    )
    work["morphology_peak_pass"] = _pass_fail_series(
        work.get("peak_price_discharge_action_fraction", zero),
        threshold=REVIEWER_MORPHOLOGY_PEAK_DISCHARGE_THRESHOLD,
        direction="ge",
    )
    work["morphology_valley_pass"] = _pass_fail_series(
        work.get("valley_price_charge_action_fraction", zero),
        threshold=REVIEWER_MORPHOLOGY_VALLEY_CHARGE_THRESHOLD,
        direction="ge",
    )
    work["morphology_behavior_pass"] = (
        work["morphology_midband_pass"].astype(bool)
        & work["morphology_peak_pass"].astype(bool)
        & work["morphology_valley_pass"].astype(bool)
    ).astype(int)
    work["shield_internalization_pass"] = (
        _pass_fail_series(
            work.get("mean_abs_shield_delta", zero),
            threshold=REVIEWER_SHIELD_MEAN_DELTA_THRESHOLD,
            direction="le",
        ).astype(bool)
        & _pass_fail_series(
            work.get("shield_material_activation_fraction", zero),
            threshold=REVIEWER_SHIELD_MATERIAL_DWELL_THRESHOLD,
            direction="le",
        ).astype(bool)
        & _pass_fail_series(
            work.get("shield_strong_activation_fraction", zero),
            threshold=REVIEWER_SHIELD_STRONG_DWELL_THRESHOLD,
            direction="le",
        ).astype(bool)
    ).astype(int)
    work["morphology_pass"] = (
        work["morphology_behavior_pass"].astype(bool) & work["shield_internalization_pass"].astype(bool)
    ).astype(int)
    work["reviewer_ready_pass"] = (
        work["value_recovery_pass"].astype(bool) & work["morphology_pass"].astype(bool)
    ).astype(int)
    return work


def _variant_args(base_args: argparse.Namespace, *, controller_variant: str, variant_output_dir: Path) -> argparse.Namespace:
    args = argparse.Namespace(**vars(base_args))
    args.output_dir = str(variant_output_dir)
    explicit_tb_name = str(getattr(base_args, "tb_log_name", "")).strip()
    if explicit_tb_name:
        args.tb_log_name = f"{explicit_tb_name}_{controller_variant}"

    def disable_replay_warmstart() -> None:
        args.offline_dataset = ""
        args.offline_dataset_controller_sources = ""
        args.offline_dataset_max_transitions = 0

    def disable_offline_actor_bc() -> None:
        args.bc_pretrain_gradient_steps = 0

    def disable_online_safe_bc() -> None:
        args.online_safe_bc_gradient_steps = 0
        args.online_safe_bc_max_samples = 0

    if controller_variant == "plain_sac":
        disable_replay_warmstart()
        disable_offline_actor_bc()
        disable_online_safe_bc()
        args.shield_enabled = False
    elif controller_variant == "replay_warmstart_sac":
        dataset_path = str(getattr(base_args, "offline_dataset", "")).strip()
        if not dataset_path:
            raise ValueError("replay_warmstart_sac requires --offline-dataset to be provided.")
        disable_offline_actor_bc()
        disable_online_safe_bc()
        args.shield_enabled = False
    elif controller_variant == "bc_warmstart_sac":
        dataset_path = str(getattr(base_args, "offline_dataset", "")).strip()
        if not dataset_path:
            raise ValueError("bc_warmstart_sac requires --offline-dataset to be provided.")
        disable_online_safe_bc()
        args.shield_enabled = False
    elif controller_variant == "shielded_sac":
        disable_replay_warmstart()
        disable_offline_actor_bc()
        disable_online_safe_bc()
        args.shield_enabled = True
    elif controller_variant == "shielded_replay_warmstart_sac":
        dataset_path = str(getattr(base_args, "offline_dataset", "")).strip()
        if not dataset_path:
            raise ValueError("shielded_replay_warmstart_sac requires --offline-dataset to be provided.")
        disable_offline_actor_bc()
        args.shield_enabled = True
    elif controller_variant == "shielded_bc_warmstart_sac":
        dataset_path = str(getattr(base_args, "offline_dataset", "")).strip()
        if not dataset_path:
            raise ValueError("shielded_bc_warmstart_sac requires --offline-dataset to be provided.")
        disable_online_safe_bc()
        args.shield_enabled = True
    else:
        raise ValueError(f"Unsupported controller variant '{controller_variant}'.")
    return args


def _aggregate_summary(summary_df: pd.DataFrame, *, groupby_columns: list[str]) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()
    metrics = [
        "oracle_reference_window_compatible",
        "final_cumulative_cost",
        "final_cumulative_objective_cost",
        "raw_savings_vs_none",
        "raw_oracle_savings_vs_none",
        "raw_recovery_fraction_vs_oracle",
        "raw_gap_to_oracle",
        "objective_none_cost",
        "objective_oracle_cost",
        "objective_savings_vs_none",
        "objective_oracle_savings_vs_none",
        "objective_recovery_fraction_vs_oracle",
        "value_recovery_pass",
        "oracle_normalized_objective_value_recovery",
        "objective_gap_to_oracle",
        "total_terminal_soc_penalty",
        "final_soc",
        "soc_upper_dwell_fraction",
        "soc_lower_dwell_fraction",
        "soc_midband_dwell_fraction",
        "soc_target_tracking_mae",
        "soc_upper_parking_fraction",
        "soc_lower_parking_fraction",
        "infeasible_action_dwell_fraction",
        "peak_price_discharge_action_fraction",
        "valley_price_charge_action_fraction",
        "morphology_midband_pass",
        "morphology_peak_pass",
        "morphology_valley_pass",
        "morphology_behavior_pass",
        "peak_price_mean_discharge_limit_ratio",
        "valley_price_mean_charge_limit_ratio",
        "peak_price_low_discharge_limit_dwell_fraction",
        "shield_activation_fraction",
        "mean_abs_shield_delta",
        "shield_material_activation_fraction",
        "shield_strong_activation_fraction",
        "shield_internalization_pass",
        "morphology_pass",
        "reviewer_ready_pass",
        "inventory_teacher_activation_fraction",
        "inventory_teacher_boundary_activation_fraction",
        "inventory_teacher_terminal_activation_fraction",
        "inventory_teacher_reserve_activation_fraction",
        "mean_abs_inventory_teacher_gap",
        "effective_online_safe_bc_gradient_steps",
        "last_validation_mean_soc_midband_dwell_fraction",
        "last_validation_mean_soc_target_tracking_mae",
        "last_validation_mean_peak_price_discharge_action_fraction",
        "last_validation_mean_valley_price_charge_action_fraction",
    ]
    agg_map = {metric: "mean" for metric in metrics if metric in summary_df.columns}
    aggregated = summary_df.groupby(groupby_columns, dropna=False).agg(agg_map).reset_index()
    aggregated = aggregated.rename(
        columns={
            "oracle_reference_window_compatible": "mean_oracle_reference_window_compatible",
            "final_cumulative_cost": "mean_final_cumulative_cost",
            "final_cumulative_objective_cost": "mean_final_cumulative_objective_cost",
            "raw_savings_vs_none": "mean_raw_savings_vs_none",
            "raw_oracle_savings_vs_none": "mean_raw_oracle_savings_vs_none",
            "raw_recovery_fraction_vs_oracle": "mean_raw_recovery_fraction_vs_oracle",
            "raw_gap_to_oracle": "mean_raw_gap_to_oracle",
            "objective_none_cost": "mean_objective_none_cost",
            "objective_oracle_cost": "mean_objective_oracle_cost",
            "objective_savings_vs_none": "mean_objective_savings_vs_none",
            "objective_oracle_savings_vs_none": "mean_objective_oracle_savings_vs_none",
            "objective_recovery_fraction_vs_oracle": "mean_objective_recovery_fraction_vs_oracle",
            "value_recovery_pass": "mean_value_recovery_pass",
            "oracle_normalized_objective_value_recovery": "mean_oracle_normalized_objective_value_recovery",
            "objective_gap_to_oracle": "mean_objective_gap_to_oracle",
            "total_terminal_soc_penalty": "mean_total_terminal_soc_penalty",
            "final_soc": "mean_final_soc",
            "soc_upper_dwell_fraction": "mean_soc_upper_dwell_fraction",
            "soc_lower_dwell_fraction": "mean_soc_lower_dwell_fraction",
            "soc_midband_dwell_fraction": "mean_soc_midband_dwell_fraction",
            "soc_target_tracking_mae": "mean_soc_target_tracking_mae",
            "soc_upper_parking_fraction": "mean_soc_upper_parking_fraction",
            "soc_lower_parking_fraction": "mean_soc_lower_parking_fraction",
            "infeasible_action_dwell_fraction": "mean_infeasible_action_dwell_fraction",
            "peak_price_discharge_action_fraction": "mean_peak_price_discharge_action_fraction",
            "valley_price_charge_action_fraction": "mean_valley_price_charge_action_fraction",
            "morphology_midband_pass": "mean_morphology_midband_pass",
            "morphology_peak_pass": "mean_morphology_peak_pass",
            "morphology_valley_pass": "mean_morphology_valley_pass",
            "morphology_behavior_pass": "mean_morphology_behavior_pass",
            "peak_price_mean_discharge_limit_ratio": "mean_peak_price_mean_discharge_limit_ratio",
            "valley_price_mean_charge_limit_ratio": "mean_valley_price_mean_charge_limit_ratio",
            "peak_price_low_discharge_limit_dwell_fraction": "mean_peak_price_low_discharge_limit_dwell_fraction",
            "shield_activation_fraction": "mean_shield_activation_fraction",
            "mean_abs_shield_delta": "mean_abs_shield_delta",
            "shield_material_activation_fraction": "mean_shield_material_activation_fraction",
            "shield_strong_activation_fraction": "mean_shield_strong_activation_fraction",
            "shield_internalization_pass": "mean_shield_internalization_pass",
            "morphology_pass": "mean_morphology_pass",
            "reviewer_ready_pass": "mean_reviewer_ready_pass",
            "inventory_teacher_activation_fraction": "mean_inventory_teacher_activation_fraction",
            "inventory_teacher_boundary_activation_fraction": "mean_inventory_teacher_boundary_activation_fraction",
            "inventory_teacher_terminal_activation_fraction": "mean_inventory_teacher_terminal_activation_fraction",
            "inventory_teacher_reserve_activation_fraction": "mean_inventory_teacher_reserve_activation_fraction",
            "mean_abs_inventory_teacher_gap": "mean_abs_inventory_teacher_gap",
            "effective_online_safe_bc_gradient_steps": "mean_effective_online_safe_bc_gradient_steps",
            "last_validation_mean_soc_midband_dwell_fraction": "mean_last_validation_soc_midband_dwell_fraction",
            "last_validation_mean_soc_target_tracking_mae": "mean_last_validation_soc_target_tracking_mae",
            "last_validation_mean_peak_price_discharge_action_fraction": "mean_last_validation_peak_price_discharge_action_fraction",
            "last_validation_mean_valley_price_charge_action_fraction": "mean_last_validation_valley_price_charge_action_fraction",
        }
    )
    if {
        "mean_soc_upper_parking_fraction",
        "mean_soc_lower_parking_fraction",
    }.issubset(aggregated.columns):
        aggregated["mean_boundary_parking_fraction"] = (
            aggregated["mean_soc_upper_parking_fraction"] + aggregated["mean_soc_lower_parking_fraction"]
        )
    count_df = summary_df.groupby(groupby_columns, dropna=False).size().reset_index(name="row_count")
    aggregated = aggregated.merge(count_df, on=groupby_columns, how="left")
    return reorder_safe_warmstart_grouped(aggregated)


def main() -> int:
    raw_argv = sys.argv[1:]
    args = build_parser().parse_args(raw_argv)
    controller_variants = _parse_controller_variants(args.controller_variants)
    case_keys = probe._parse_csv_arg(args.cases)
    regimes = probe._parse_csv_arg(args.regimes)
    train_models = probe._parse_csv_arg(args.train_models)
    test_models = probe._parse_csv_arg(args.test_models)
    seeds = probe._parse_seed_list(args.seeds, args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectories_root = output_dir / "trajectories"
    trajectories_root.mkdir(parents=True, exist_ok=True)
    validation_history_root = output_dir / "validation_history"
    validation_history_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    for seed in seeds:
        for case_key in case_keys:
            for regime in regimes:
                for train_model in train_models:
                    for controller_variant in controller_variants:
                        variant_output_dir = output_dir / controller_variant
                        variant_output_dir.mkdir(parents=True, exist_ok=True)
                        run_args = argparse.Namespace(**{**vars(args), "seed": int(seed)})
                        run_args = probe.apply_ieee33_sac_default_protocol(run_args, case_key=case_key, raw_argv=raw_argv)
                        run_args = probe.apply_ieee33_full_fair_protocol(
                            run_args,
                            case_key=case_key,
                            train_model=train_model,
                            raw_argv=raw_argv,
                        )
                        run_args = _variant_args(run_args, controller_variant=controller_variant, variant_output_dir=variant_output_dir)

                        print(
                            f"[compare-train] variant={controller_variant} case={case_key} regime={regime} "
                            f"train_model={train_model} seed={seed} steps={int(run_args.train_steps)}"
                        )
                        regularization_cfg = probe.action_regularization_config(run_args)
                        rule_cfg = probe.rule_guidance_config(run_args)
                        shield_cfg = probe.shield_config(run_args)
                        agent, train_schedule, train_window, tb_metadata = probe.train_short_agent(
                            case_key=case_key,
                            train_model=train_model,
                            regime=regime,
                            args=run_args,
                        )
                        validation_state = dict(tb_metadata.get("validation", {}))
                        validation_history = list(tb_metadata.get("validation_history", []))
                        eval_window_specs = _build_eval_window_specs(run_args, case_key=case_key, regime=regime)

                        for test_model in test_models:
                            for eval_spec in eval_window_specs:
                                print(
                                    f"[compare-eval] variant={controller_variant} case={case_key} regime={regime} "
                                    f"train={train_model} test={test_model} seed={seed} window={eval_spec['label']}"
                                )
                                summary, trajectory, eval_window = probe.evaluate_agent(
                                    agent,
                                    case_key=case_key,
                                    test_model=test_model,
                                    regime=regime,
                                    args=run_args,
                                    eval_window_override=eval_spec["window"],
                                    eval_steps_override=0,
                                    eval_full_horizon_override=True,
                                )
                                row = {
                                    "case": case_key,
                                    "regime": regime,
                                    "controller_variant": controller_variant,
                                    "reward_profile": str(run_args.reward_profile),
                                    "agent": str(run_args.agent),
                                    "seed": int(seed),
                                    "train_model": train_model,
                                    "test_model": test_model,
                                    "eval_window_label": str(eval_spec["label"]),
                                    "eval_window_family": str(eval_spec["family"]),
                                    "eval_window_days": int(eval_spec["days"]),
                                    "eval_offset_days_within_year": int(eval_spec["offset_days"]),
                                    "train_steps": int(run_args.train_steps),
                                    "eval_steps": int(summary["steps"]),
                                    "learning_rate": float(run_args.learning_rate),
                                    "tensorboard_log_dir": str(tb_metadata["tensorboard_log_dir"]),
                                    "tensorboard_run_name": str(tb_metadata["tensorboard_run_name"]),
                                    "action_smoothing_coef": float(regularization_cfg["smoothing_coef"]),
                                    "action_max_delta": float(regularization_cfg["max_delta"]),
                                    "action_rate_penalty": float(regularization_cfg["rate_penalty"]),
                                    "battery_feasibility_aware": int(bool(regularization_cfg["battery_feasibility_aware"])),
                                    "battery_infeasible_penalty": float(regularization_cfg["battery_infeasible_penalty"]),
                                    "symmetric_battery_action": int(bool(regularization_cfg["symmetric_battery_action"])),
                                    "shield_enabled": int(bool(getattr(run_args, "shield_enabled", False))),
                                    "shield_soc_soft_buffer_fraction": float(shield_cfg["soc_soft_buffer_fraction"]),
                                    "shield_soc_hard_buffer_fraction": float(shield_cfg["soc_hard_buffer_fraction"]),
                                    "shield_peak_reserve_min_fraction": float(shield_cfg["reserve_discharge_min_fraction"]),
                                    "shield_hard_pullback_action": float(shield_cfg["hard_pullback_action"]),
                                    "shield_terminal_closure_horizon_fraction": float(shield_cfg["terminal_closure_horizon_fraction"]),
                                    "shield_terminal_closure_urgency_soc": float(shield_cfg["terminal_closure_urgency_soc"]),
                                    "rule_guidance_mix": float(rule_cfg["guidance_mix"]),
                                    "rule_guidance_decay_steps": int(rule_cfg["guidance_decay_steps"]),
                                    "rule_guidance_policy": str(rule_cfg["guidance_policy"]),
                                    "protocol_profile": str(probe.protocol_profile(run_args)),
                                    "train_validation_metric": str(getattr(run_args, "train_validation_metric", "objective_cost")),
                                    "offline_dataset": str(getattr(run_args, "offline_dataset", "")),
                                    "offline_dataset_controller_sources": str(getattr(run_args, "offline_dataset_controller_sources", "")),
                                    "offline_dataset_max_transitions": int(getattr(run_args, "offline_dataset_max_transitions", 0)),
                                    "bc_pretrain_gradient_steps": int(getattr(run_args, "bc_pretrain_gradient_steps", 0)),
                                    "bc_pretrain_batch_size": int(getattr(run_args, "bc_pretrain_batch_size", 256)),
                                    "bc_pretrain_learning_rate": float(getattr(run_args, "bc_pretrain_learning_rate", 0.0)),
                                    "shield_delta_penalty_coef": float(getattr(run_args, "shield_delta_penalty_coef", 0.0)),
                                    "online_safe_bc_gradient_steps": int(getattr(run_args, "online_safe_bc_gradient_steps", 0)),
                                    "online_safe_bc_batch_size": int(getattr(run_args, "online_safe_bc_batch_size", 256)),
                                    "online_safe_bc_max_samples": int(getattr(run_args, "online_safe_bc_max_samples", 0)),
                                    "online_safe_bc_learning_rate": float(getattr(run_args, "online_safe_bc_learning_rate", 0.0)),
                                    "online_safe_bc_intervention_priority_coef": float(
                                        getattr(run_args, "online_safe_bc_intervention_priority_coef", 4.0)
                                    ),
                                    "online_safe_bc_boundary_priority_coef": float(
                                        getattr(run_args, "online_safe_bc_boundary_priority_coef", 2.0)
                                    ),
                                    "online_safe_bc_terminal_priority_coef": float(
                                        getattr(run_args, "online_safe_bc_terminal_priority_coef", 2.0)
                                    ),
                                    "online_safe_bc_teacher_priority_coef": float(
                                        getattr(run_args, "online_safe_bc_teacher_priority_coef", 2.0)
                                    ),
                                    "train_window_start": str(train_window["window_start_timestamp"]) if train_window is not None else "",
                                    "train_window_end": str(train_window["window_end_timestamp"]) if train_window is not None else "",
                                    "eval_window_start": str(eval_window["window_start_timestamp"]) if eval_window is not None else "",
                                    "eval_window_end": str(eval_window["window_end_timestamp"]) if eval_window is not None else "",
                                    "validation_best_metric_value": float(validation_state.get("best_metric_value", float("nan"))),
                                    "validation_best_total_reward": float(validation_state.get("best_total_reward", float("nan"))),
                                    "validation_best_objective_cost": float(validation_state.get("best_objective_cost", float("nan"))),
                                    "validation_best_checkpoint_step": int(validation_state.get("best_checkpoint_step", int(run_args.train_steps))),
                                    "offline_bc_dataset_rows": int(validation_state.get("offline_bc_dataset_rows", 0)),
                                    "offline_bc_replay_seeded_transitions": int(validation_state.get("offline_bc_replay_seeded_transitions", 0)),
                                    "offline_bc_actor_gradient_steps": int(validation_state.get("offline_bc_actor_gradient_steps", 0)),
                                    "offline_bc_actor_batch_size": int(validation_state.get("offline_bc_actor_batch_size", 0)),
                                    "offline_bc_initial_actor_mse": float(validation_state.get("offline_bc_initial_actor_mse", float("nan"))),
                                    "offline_bc_final_actor_mse": float(validation_state.get("offline_bc_final_actor_mse", float("nan"))),
                                    "online_safe_bc_replay_rows": int(validation_state.get("online_safe_bc_replay_rows", 0)),
                                    "online_safe_bc_actor_gradient_steps_applied": int(
                                        validation_state.get("online_safe_bc_actor_gradient_steps", 0)
                                    ),
                                    "online_safe_bc_initial_actor_mse": float(
                                        validation_state.get("online_safe_bc_initial_actor_mse", float("nan"))
                                    ),
                                    "online_safe_bc_final_actor_mse": float(
                                        validation_state.get("online_safe_bc_final_actor_mse", float("nan"))
                                    ),
                                    "online_safe_bc_intervention_rows": int(validation_state.get("online_safe_bc_intervention_rows", 0)),
                                    "online_safe_bc_inventory_teacher_rows": int(
                                        validation_state.get("online_safe_bc_inventory_teacher_rows", 0)
                                    ),
                                    "online_safe_bc_mean_sample_weight": float(
                                        validation_state.get("online_safe_bc_mean_sample_weight", 1.0)
                                    ),
                                    "effective_online_safe_bc_gradient_steps": int(
                                        validation_state.get("effective_online_safe_bc_gradient_steps", 0)
                                    ),
                                    "last_validation_mean_shield_material_activation_fraction": float(
                                        validation_state.get("last_validation_mean_shield_material_activation_fraction", float("nan"))
                                    ),
                                    "last_validation_mean_abs_shield_delta": float(
                                        validation_state.get("last_validation_mean_abs_shield_delta", float("nan"))
                                    ),
                                    "last_validation_mean_soc_midband_dwell_fraction": float(
                                        validation_state.get("last_validation_mean_soc_midband_dwell_fraction", float("nan"))
                                    ),
                                    "last_validation_mean_soc_target_tracking_mae": float(
                                        validation_state.get("last_validation_mean_soc_target_tracking_mae", float("nan"))
                                    ),
                                    "last_validation_mean_peak_price_discharge_action_fraction": float(
                                        validation_state.get(
                                            "last_validation_mean_peak_price_discharge_action_fraction", float("nan")
                                        )
                                    ),
                                    "last_validation_mean_valley_price_charge_action_fraction": float(
                                        validation_state.get(
                                            "last_validation_mean_valley_price_charge_action_fraction", float("nan")
                                        )
                                    ),
                                    **summary,
                                }
                                summary_rows.append(row)

                                stem = _artifact_stem(
                                    case_key=case_key,
                                    regime=regime,
                                    agent=str(run_args.agent),
                                    train_model=train_model,
                                    test_model=test_model,
                                    eval_window_label=str(eval_spec["label"]),
                                    seed=int(seed),
                                )
                                trajectory = trajectory.copy()
                                trajectory.insert(0, "controller_variant", controller_variant)
                                trajectory.insert(1, "train_model", train_model)
                                trajectory.insert(2, "test_model", test_model)
                                trajectory.insert(3, "eval_window_label", str(eval_spec["label"]))
                                trajectories_dir = trajectories_root / controller_variant
                                trajectories_dir.mkdir(parents=True, exist_ok=True)
                                trajectory.to_csv(trajectories_dir / f"{stem}.csv", index=False)
                                if validation_history:
                                    validation_dir = validation_history_root / controller_variant
                                    validation_dir.mkdir(parents=True, exist_ok=True)
                                    pd.DataFrame(validation_history).to_csv(validation_dir / f"{stem}.csv", index=False)
                        vec_env = getattr(agent, "get_env", lambda: None)()
                        if vec_env is not None and hasattr(vec_env, "close"):
                            vec_env.close()

    summary_df = pd.DataFrame(summary_rows)
    summary_df = ensure_eval_window_columns(summary_df)
    oracle_reference_csv = _preferred_oracle_reference_csv(
        getattr(args, "oracle_reference_csv", ""),
        getattr(args, "network_oracle_reference_csv", ""),
    )
    if oracle_reference_csv and oracle_reference_csv != str(getattr(args, "oracle_reference_csv", "")):
        print(f"[oracle-reference] using network-replayed reference: {oracle_reference_csv}")
    summary_df = _attach_oracle_value_recovery(summary_df, reference_csv=oracle_reference_csv)
    summary_df = attach_reviewer_pass_fail_summary(summary_df)
    if not summary_df.empty:
        summary_df = reorder_safe_warmstart_summary(summary_df)

    groupby_columns = _parse_csv_arg(args.groupby_columns)
    grouped_df = _aggregate_summary(summary_df, groupby_columns=groupby_columns)

    summary_csv = output_dir / "summary.csv"
    summary_json = output_dir / "summary.json"
    grouped_csv = output_dir / "summary_grouped.csv"
    grouped_json = output_dir / "summary_grouped.json"
    summary_df.to_csv(summary_csv, index=False)
    summary_json.write_text(json.dumps(summary_df.to_dict(orient="records"), indent=2), encoding="utf-8")
    grouped_df.to_csv(grouped_csv, index=False)
    grouped_json.write_text(json.dumps(grouped_df.to_dict(orient="records"), indent=2), encoding="utf-8")
    reviewer_outputs = write_reviewer_grouped_tables(output_dir, grouped_df)
    paper_outputs = write_paper_main_tables(output_dir, summary_df)
    audit_outputs = write_storyline_audit(output_dir, summary_df)

    print("\n=== Safe / Warmstart Comparison Summary ===")
    print(summary_df.to_string(index=False))
    if not grouped_df.empty:
        print("\n=== Grouped Comparison Summary ===")
        print(grouped_df.to_string(index=False))
    print(f"\nSaved summary CSV: {summary_csv}")
    print(f"Saved grouped summary CSV: {grouped_csv}")
    print(f"Saved reviewer value-recovery table: {reviewer_outputs['value_csv']}")
    print(f"Saved reviewer battery-behavior table: {reviewer_outputs['behavior_csv']}")
    print(f"Saved paper main value-recovery table: {paper_outputs['value_csv']}")
    print(f"Saved paper main battery-behavior table: {paper_outputs['behavior_csv']}")
    print(f"Saved storyline audit: {audit_outputs['audit_csv']}")
    print(f"Saved trajectories root: {trajectories_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
