#!/usr/bin/env python3
"""Rebuild consolidated summary tables from existing trajectory CSV files across all seeds."""

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

import scripts.analysis.short_cross_fidelity_probe as probe
from scripts.analysis.compare_safe_warmstart_sac import (
    _aggregate_summary,
    _attach_oracle_value_recovery,
    attach_reviewer_pass_fail_summary,
    ensure_eval_window_columns,
    reorder_safe_warmstart_grouped,
    reorder_safe_warmstart_summary,
    write_storyline_audit,
    write_paper_main_tables,
    write_reviewer_grouped_tables,
)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rebuild summary tables from trajectory files.")
    parser.add_argument(
        "--result-dir",
        type=str,
        default="results/sc33_20k_prio_sailsac_balancedgate",
        help="Directory containing trajectories/ and summary.csv template.",
    )
    parser.add_argument(
        "--oracle-reference-csv",
        type=str,
        default="results/oracle_reference_suite_ieee33_simple_2024/network_replayed_oracle_reference_windows.csv",
        help="Oracle/None reference CSV for normalization.",
    )
    return parser

def _dwell_fraction(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    return float(series.astype(bool).sum() / len(series))

def rebuild_metrics(trajectory: pd.DataFrame, args: argparse.Namespace) -> dict[str, Any]:
    # Hardcoded thresholds for IEEE33 as used in the project
    peak_price_threshold = 0.51373
    valley_price_threshold = 0.39073
    soc_min = 0.1
    soc_max = 0.9
    soc_init = 0.5
    p_charge_max = 1000000.0  # 1MW
    p_discharge_max = 1000000.0

    peak_price_metrics = probe._peak_price_reserve_metrics(
        trajectory,
        peak_price_threshold=peak_price_threshold,
        discharge_limit_scale_w=p_discharge_max,
        low_discharge_limit_threshold=0.25,
    )
    inventory_behavior_metrics = probe._inventory_behavior_metrics(
        trajectory,
        target_soc=soc_init,
        target_tolerance=0.0,
        soc_min=soc_min,
        soc_max=soc_max,
        valley_price_threshold=valley_price_threshold,
        peak_price_threshold=peak_price_threshold,
        charge_limit_scale_w=p_charge_max,
        discharge_limit_scale_w=p_discharge_max,
    )
    
    total_reward = float(trajectory["reward"].sum()) if "reward" in trajectory.columns else 0.0
    
    metrics = {
        "steps": int(len(trajectory)),
        "total_reward": total_reward,
        "final_soc": float(trajectory["soc"].iloc[-1]) if not trajectory.empty else 0.0,
        "final_cumulative_cost": float(trajectory["cumulative_cost"].iloc[-1]) if "cumulative_cost" in trajectory.columns and not trajectory.empty else 0.0,
        "final_cumulative_objective_cost": float(trajectory["cumulative_objective_cost"].iloc[-1]) if "cumulative_objective_cost" in trajectory.columns and not trajectory.empty else 0.0,
        "min_voltage_worst": float(trajectory["min_bus_voltage_pu"].min()) if "min_bus_voltage_pu" in trajectory.columns else 1.0,
        "max_line_loading_peak": float(trajectory["max_line_loading_pct"].max()) if "max_line_loading_pct" in trajectory.columns else 0.0,
        "max_line_current_peak_ka": float(trajectory["max_line_current_ka"].max()) if "max_line_current_ka" in trajectory.columns else 0.0,
        "mean_grid_import_mw": float(trajectory["grid_import_mw"].mean()) if "grid_import_mw" in trajectory.columns else 0.0,
        "final_temperature_c": float(trajectory["temperature_c"].iloc[-1]) if "temperature_c" in trajectory.columns and not trajectory.empty else 0.0,
        "final_terminal_soc_deviation": float(trajectory["terminal_soc_deviation"].iloc[-1]) if "terminal_soc_deviation" in trajectory.columns and not trajectory.empty else 0.0,
        "total_terminal_soc_penalty": float(trajectory["terminal_soc_penalty"].sum()) if "terminal_soc_penalty" in trajectory.columns else 0.0,
        "total_battery_loss_kwh": float(trajectory["battery_loss_kwh"].sum()) if "battery_loss_kwh" in trajectory.columns else 0.0,
        "total_battery_stress_kwh": float(trajectory["battery_stress_kwh"].sum()) if "battery_stress_kwh" in trajectory.columns else 0.0,
        "total_battery_throughput_kwh": float(trajectory["battery_throughput_kwh"].sum()) if "battery_throughput_kwh" in trajectory.columns else 0.0,
        "mean_abs_battery_action_delta": float(trajectory["battery_action_delta"].abs().mean()) if "battery_action_delta" in trajectory.columns else 0.0,
        "total_action_rate_penalty": float(trajectory["action_rate_penalty"].sum()) if "action_rate_penalty" in trajectory.columns else 0.0,
        "shield_activation_fraction": _dwell_fraction(trajectory["shield_applied"]) if "shield_applied" in trajectory.columns else 0.0,
        "shield_reserve_activation_fraction": _dwell_fraction(trajectory["shield_reserve_active"]) if "shield_reserve_active" in trajectory.columns else 0.0,
        "shield_boundary_activation_fraction": _dwell_fraction(trajectory["shield_boundary_active"]) if "shield_boundary_active" in trajectory.columns else 0.0,
        "shield_terminal_activation_fraction": _dwell_fraction(trajectory["shield_terminal_active"]) if "shield_terminal_active" in trajectory.columns else 0.0,
        "mean_abs_shield_delta": float(trajectory["shield_delta"].abs().mean()) if "shield_delta" in trajectory.columns else 0.0,
        "shield_material_activation_fraction": _dwell_fraction(trajectory["shield_delta"].abs() > 0.01) if "shield_delta" in trajectory.columns else 0.0,
        "shield_strong_activation_fraction": _dwell_fraction(trajectory["shield_delta"].abs() > 0.05) if "shield_delta" in trajectory.columns else 0.0,
        "inventory_teacher_activation_fraction": _dwell_fraction(trajectory["inventory_teacher_active"]) if "inventory_teacher_active" in trajectory.columns else 0.0,
        "inventory_teacher_boundary_activation_fraction": _dwell_fraction(trajectory["inventory_teacher_boundary_active"]) if "inventory_teacher_boundary_active" in trajectory.columns else 0.0,
        "inventory_teacher_terminal_activation_fraction": _dwell_fraction(trajectory["inventory_teacher_terminal_active"]) if "inventory_teacher_terminal_active" in trajectory.columns else 0.0,
        "inventory_teacher_reserve_activation_fraction": _dwell_fraction(trajectory["inventory_teacher_reserve_active"]) if "inventory_teacher_reserve_active" in trajectory.columns else 0.0,
        "mean_abs_inventory_teacher_gap": float(
            (
                trajectory["inventory_teacher_action"].astype(float)
                - trajectory["battery_action_applied_pre_shield"].astype(float)
            ).abs().mean()
        ) if "inventory_teacher_action" in trajectory.columns and "battery_action_applied_pre_shield" in trajectory.columns else 0.0,
        **peak_price_metrics,
        **inventory_behavior_metrics,
        "mean_battery_action_infeasible_gap": float(trajectory["battery_action_infeasible_gap"].mean()) if "battery_action_infeasible_gap" in trajectory.columns else 0.0,
        "mean_battery_internal_clip_gap_w": float(trajectory["battery_internal_clip_gap_w"].mean()) if "battery_internal_clip_gap_w" in trajectory.columns else 0.0,
        "total_battery_action_infeasible_penalty": float(trajectory["battery_action_infeasible_penalty"].sum()) if "battery_action_infeasible_penalty" in trajectory.columns else 0.0,
        "soc_upper_dwell_fraction": _dwell_fraction(trajectory["soc_upper_bound_hit"]) if "soc_upper_bound_hit" in trajectory.columns else 0.0,
        "soc_lower_dwell_fraction": _dwell_fraction(trajectory["soc_lower_bound_hit"]) if "soc_lower_bound_hit" in trajectory.columns else 0.0,
        "infeasible_action_dwell_fraction": _dwell_fraction(trajectory["battery_action_infeasible_flag"]) if "battery_action_infeasible_flag" in trajectory.columns else 0.0,
        "internal_clip_dwell_fraction": _dwell_fraction(trajectory["battery_internal_clip_flag"]) if "battery_internal_clip_flag" in trajectory.columns else 0.0,
        "power_flow_failure_steps": int(trajectory["power_flow_failed"].sum()) if "power_flow_failed" in trajectory.columns else 0,
    }
    return metrics

def parse_filename(filename: str) -> dict[str, Any]:
    # Example: ieee33_network_stress_sac_tr-simple_te-simple_30d_off0_s42.csv
    pattern = r"ieee33_network_stress_sac_tr-(?P<train_model>[^_]+)_te-(?P<test_model>[^_]+)_(?P<eval_window_label>[^_]+_off[^_]+)_s(?P<seed>\d+)\.csv"
    match = re.match(pattern, filename)
    if not match:
        # Try 365d: ieee33_network_stress_sac_tr-simple_te-simple_365d_off0_s42.csv
        pattern = r"ieee33_network_stress_sac_tr-(?P<train_model>[^_]+)_te-(?P<test_model>[^_]+)_(?P<eval_window_label>365d_off\d+)_s(?P<seed>\d+)\.csv"
        match = re.match(pattern, filename)
    
    if match:
        d = match.groupdict()
        d["seed"] = int(d["seed"])
        # Split eval_window_label into days and offset
        win_match = re.match(r"(?P<days>\d+)d_off(?P<offset>\d+)", d["eval_window_label"])
        if win_match:
            d["eval_window_days"] = int(win_match.group("days"))
            d["eval_offset_days_within_year"] = int(win_match.group("offset"))
            d["eval_window_family"] = f"{d['eval_window_days']}d"
        return d
    return {}

def main() -> int:
    args = build_parser().parse_args()
    result_dir = Path(args.result_dir)
    trajectories_root = result_dir / "trajectories" / "shielded_replay_warmstart_sac"
    
    if not trajectories_root.exists():
        print(f"Error: {trajectories_root} does not exist.")
        return 1
    
    template_df = pd.read_csv(result_dir / "summary.csv")
    if template_df.empty:
        print("Error: summary.csv is empty.")
        return 1
    template_row = template_df.iloc[0].to_dict()
    
    # Static metadata keys to keep from template
    metadata_keys = [
        "case", "regime", "controller_variant", "reward_profile", "agent",
        "train_steps", "learning_rate", "tensorboard_log_dir", "tensorboard_run_name",
        "action_smoothing_coef", "action_max_delta", "action_rate_penalty",
        "battery_feasibility_aware", "battery_infeasible_penalty", "symmetric_battery_action",
        "shield_enabled", "shield_soc_soft_buffer_fraction", "shield_soc_hard_buffer_fraction",
        "shield_peak_reserve_min_fraction", "shield_hard_pullback_action",
        "shield_terminal_closure_horizon_fraction", "shield_terminal_closure_urgency_soc",
        "rule_guidance_mix", "rule_guidance_decay_steps", "rule_guidance_policy",
        "protocol_profile", "train_validation_metric", "offline_dataset",
        "offline_dataset_controller_sources", "offline_dataset_max_transitions",
        "bc_pretrain_gradient_steps", "bc_pretrain_batch_size", "bc_pretrain_learning_rate",
        "shield_delta_penalty_coef", "online_safe_bc_gradient_steps", "online_safe_bc_batch_size",
        "online_safe_bc_max_samples", "online_safe_bc_learning_rate",
        "online_safe_bc_intervention_priority_coef", "online_safe_bc_boundary_priority_coef",
        "online_safe_bc_terminal_priority_coef", "online_safe_bc_teacher_priority_coef",
        "validation_best_metric_value", "validation_best_total_reward", "validation_best_objective_cost",
        "validation_best_checkpoint_step", "offline_bc_dataset_rows", "offline_bc_replay_seeded_transitions",
        "offline_bc_actor_gradient_steps", "offline_bc_actor_batch_size",
        "offline_bc_initial_actor_mse", "offline_bc_final_actor_mse",
        "online_safe_bc_replay_rows", "online_safe_bc_actor_gradient_steps_applied",
        "online_safe_bc_initial_actor_mse", "online_safe_bc_final_actor_mse",
        "online_safe_bc_intervention_rows", "online_safe_bc_inventory_teacher_rows",
        "online_safe_bc_mean_sample_weight", "effective_online_safe_bc_gradient_steps",
        "last_validation_mean_shield_material_activation_fraction", "last_validation_mean_abs_shield_delta",
        "last_validation_mean_soc_midband_dwell_fraction", "last_validation_mean_soc_target_tracking_mae",
        "last_validation_mean_peak_price_discharge_action_fraction", "last_validation_mean_valley_price_charge_action_fraction",
        "train_window_start", "train_window_end", "eval_window_start", "eval_window_end"
    ]
    base_metadata = {k: template_row.get(k) for k in metadata_keys if k in template_row}
    
    summary_rows = []
    for csv_path in trajectories_root.glob("*.csv"):
        print(f"Processing {csv_path.name}...")
        file_info = parse_filename(csv_path.name)
        if not file_info:
            print(f"  Skipping {csv_path.name} (failed to parse filename)")
            continue
            
        trajectory = pd.read_csv(csv_path)
        metrics = rebuild_metrics(trajectory, argparse.Namespace())
        
        row = {**base_metadata, **file_info, **metrics}
        
        # Correct eval_window_start/end for normalization alignment
        eval_year = 2024  # Standard for this project
        if "eval_offset_days_within_year" in file_info and "eval_window_days" in file_info:
            offset = file_info["eval_offset_days_within_year"]
            win_days = file_info["eval_window_days"]
            start_ts = pd.to_datetime(f"{eval_year}-01-01") + pd.Timedelta(days=offset)
            end_ts = start_ts + pd.Timedelta(days=win_days) - pd.Timedelta(minutes=15)
            row["eval_window_start"] = start_ts.strftime("%Y-%m-%d %H:%M:%S")
            row["eval_window_end"] = end_ts.strftime("%Y-%m-%d %H:%M:%S")

        summary_rows.append(row)
        
    if not summary_rows:
        print("Error: No summary rows generated.")
        return 1
        
    summary_df = pd.DataFrame(summary_rows)
    summary_df = ensure_eval_window_columns(summary_df)
    summary_df = _attach_oracle_value_recovery(summary_df, reference_csv=args.oracle_reference_csv)
    summary_df = attach_reviewer_pass_fail_summary(summary_df)
    summary_df = reorder_safe_warmstart_summary(summary_df)
    
    grouped_df = _aggregate_summary(summary_df, groupby_columns=["case", "regime", "controller_variant", "train_model", "test_model"])
    grouped_df = reorder_safe_warmstart_grouped(grouped_df)
    
    # Save files
    summary_df.to_csv(result_dir / "summary_3seed.csv", index=False)
    grouped_df.to_csv(result_dir / "summary_grouped_3seed.csv", index=False)
    
    write_reviewer_grouped_tables(result_dir, grouped_df, suffix="_3seed")
    write_paper_main_tables(result_dir, summary_df, suffix="_3seed")
    write_storyline_audit(result_dir, summary_df, suffix="_3seed")
    
    print(f"\nRebuild complete! Outputs in {result_dir}:")
    print("  - summary_3seed.csv (15 rows expected)")
    print("  - summary_grouped_3seed.csv")
    print("  - paper_main_value_recovery_3seed.csv")
    print("  - paper_main_battery_behavior_3seed.csv")
    print("  - storyline_audit_3seed.csv")
    
    return 0

# Need to use the actual function names from compare_safe_warmstart_sac
def write_reviewer_grouped_tables(result_dir: Path, grouped_df: pd.DataFrame, suffix: str = ""):
    from scripts.analysis.compare_safe_warmstart_sac import (
        build_reviewer_value_recovery_grouped, build_reviewer_battery_behavior_grouped
    )
    value_df = build_reviewer_value_recovery_grouped(grouped_df)
    behavior_df = build_reviewer_battery_behavior_grouped(grouped_df)
    value_df.to_csv(result_dir / f"reviewer_value_recovery_grouped{suffix}.csv", index=False)
    behavior_df.to_csv(result_dir / f"reviewer_battery_behavior_grouped{suffix}.csv", index=False)

def write_paper_main_tables(result_dir: Path, summary_df: pd.DataFrame, suffix: str = ""):
    from scripts.analysis.compare_safe_warmstart_sac import (
        build_paper_main_value_recovery, build_paper_main_battery_behavior
    )
    paper_value = build_paper_main_value_recovery(summary_df)
    paper_behavior = build_paper_main_battery_behavior(summary_df)
    paper_value.to_csv(result_dir / f"paper_main_value_recovery{suffix}.csv", index=False)
    paper_behavior.to_csv(result_dir / f"paper_main_battery_behavior{suffix}.csv", index=False)

def write_storyline_audit(result_dir: Path, summary_df: pd.DataFrame, suffix: str = ""):
    from scripts.analysis.compare_safe_warmstart_sac import (
        build_storyline_audit
    )
    audit = build_storyline_audit(summary_df)
    audit.to_csv(result_dir / f"storyline_audit{suffix}.csv", index=False)

if __name__ == "__main__":
    sys.exit(main())
