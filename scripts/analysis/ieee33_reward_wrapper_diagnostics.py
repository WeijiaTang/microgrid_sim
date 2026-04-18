from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.short_cross_fidelity_probe import build_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate reward and wrapper diagnostics from probe trajectories.")
    parser.add_argument("--summary-csv", type=Path, required=True, help="Path to short_cross_fidelity_probe summary.csv")
    parser.add_argument(
        "--trajectories-dir",
        type=Path,
        default=None,
        help="Directory containing trajectory CSVs. Defaults to <summary parent>/trajectories",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional output CSV. Defaults to <summary stem>_reward_diagnostics.csv",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional output JSON. Defaults to <summary stem>_reward_diagnostics.json",
    )
    return parser


def _safe_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.zeros(len(frame), dtype=float), index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def _trajectory_stem(row: pd.Series) -> str:
    return (
        f"{str(row['case'])}_{str(row['regime'])}_{str(row['agent'])}"
        f"_train-{str(row['train_model'])}_test-{str(row['test_model'])}_seed{int(row['seed'])}"
    )


def _aggregate_reward_terms(row: pd.Series, trajectory: pd.DataFrame) -> dict[str, float | int | str]:
    cfg = build_config(
        case_key=str(row["case"]),
        battery_model=str(row["test_model"]),
        days=max(int(row.get("eval_config_days", 1)), 1),
        seed=int(row["seed"]),
        regime=str(row["regime"]),
        reward_profile=str(row.get("reward_profile", "network")),
    )
    reward_cfg = cfg.reward

    import_cost = _safe_series(trajectory, "import_cost")
    soc_violation = _safe_series(trajectory, "soc_violation")
    undervoltage = _safe_series(trajectory, "undervoltage")
    overvoltage = _safe_series(trajectory, "overvoltage")
    line_overload_pct = _safe_series(trajectory, "line_overload_pct")
    transformer_overload_pct = _safe_series(trajectory, "transformer_overload_pct")
    battery_throughput_kwh = _safe_series(trajectory, "battery_throughput_kwh")
    battery_loss_kwh = _safe_series(trajectory, "battery_loss_kwh")
    battery_stress_kwh = _safe_series(trajectory, "battery_stress_kwh")
    power_flow_failure_penalty = _safe_series(trajectory, "power_flow_failure_penalty")
    battery_shaping_penalty = _safe_series(trajectory, "battery_shaping_penalty")
    boundary_dwell_penalty = _safe_series(trajectory, "boundary_dwell_penalty")
    terminal_soc_penalty = _safe_series(trajectory, "terminal_soc_penalty")
    reward_wrapper_adjustment = _safe_series(trajectory, "reward_wrapper_adjustment")
    step_reward_before_clip = _safe_series(trajectory, "step_reward_before_clip")
    step_reward_after_clip = _safe_series(trajectory, "step_reward_after_clip")
    infeasible_penalty = _safe_series(trajectory, "battery_action_infeasible_penalty")
    action_rate_penalty = _safe_series(trajectory, "action_rate_penalty")

    weighted_cost_term = float(reward_cfg.w_cost) * float(import_cost.sum())
    weighted_soc_violation_term = float(reward_cfg.w_soc_violation) * float(soc_violation.sum())
    weighted_voltage_term = float(reward_cfg.w_voltage_violation) * float((undervoltage + overvoltage).sum())
    weighted_line_term = float(reward_cfg.w_line_overload) * float((line_overload_pct / 100.0).sum())
    weighted_transformer_term = float(reward_cfg.w_transformer_overload) * float((transformer_overload_pct / 100.0).sum())
    weighted_throughput_term = float(getattr(cfg, "battery_throughput_penalty_per_kwh", 0.0)) * float(battery_throughput_kwh.sum())
    weighted_loss_term = float(getattr(cfg, "battery_loss_penalty_per_kwh", 0.0)) * float(battery_loss_kwh.sum())
    weighted_stress_term = float(getattr(cfg, "battery_stress_penalty_per_kwh", 0.0)) * float(battery_stress_kwh.sum())
    power_flow_failure_term = float(power_flow_failure_penalty.sum())
    wrapper_penalty_magnitude = float(-reward_wrapper_adjustment.sum())

    dominance_blocks = {
        "energy_cost_term": weighted_cost_term,
        "grid_constraint_term": weighted_voltage_term + weighted_line_term + weighted_transformer_term + power_flow_failure_term,
        "battery_shaping_term": float(battery_shaping_penalty.sum()),
        "boundary_dwell_term": float(boundary_dwell_penalty.sum()),
        "terminal_soc_term": float(terminal_soc_penalty.sum()),
        "wrapper_term": wrapper_penalty_magnitude,
    }
    dominant_block = max(dominance_blocks.items(), key=lambda item: abs(float(item[1])))[0]

    clipped_fraction = float((step_reward_before_clip.ne(step_reward_after_clip)).mean()) if len(trajectory) else 0.0
    terminal_fraction = float((terminal_soc_penalty.abs() > 1e-9).mean()) if len(trajectory) else 0.0

    return {
        "trajectory_path": "",
        "steps": int(len(trajectory)),
        "weighted_cost_term": weighted_cost_term,
        "weighted_soc_violation_term": weighted_soc_violation_term,
        "weighted_voltage_term": weighted_voltage_term,
        "weighted_line_overload_term": weighted_line_term,
        "weighted_transformer_overload_term": weighted_transformer_term,
        "weighted_throughput_term": weighted_throughput_term,
        "weighted_loss_term": weighted_loss_term,
        "weighted_stress_term": weighted_stress_term,
        "power_flow_failure_term": power_flow_failure_term,
        "battery_shaping_term": float(battery_shaping_penalty.sum()),
        "terminal_soc_term": float(terminal_soc_penalty.sum()),
        "wrapper_penalty_term": wrapper_penalty_magnitude,
        "action_rate_penalty_term": float(action_rate_penalty.sum()),
        "battery_infeasible_penalty_term": float(-infeasible_penalty.sum()),
        "clipped_step_fraction": clipped_fraction,
        "terminal_penalty_step_fraction": terminal_fraction,
        "soc_upper_dwell_fraction": float(_safe_series(trajectory, "soc_upper_bound_hit").mean()) if len(trajectory) else 0.0,
        "soc_lower_dwell_fraction": float(_safe_series(trajectory, "soc_lower_bound_hit").mean()) if len(trajectory) else 0.0,
        "infeasible_action_dwell_fraction": float(_safe_series(trajectory, "battery_action_infeasible_flag").mean()) if len(trajectory) else 0.0,
        "internal_clip_dwell_fraction": float(_safe_series(trajectory, "battery_internal_clip_flag").mean()) if len(trajectory) else 0.0,
        "mean_discharge_limit_ratio": float(_safe_series(trajectory, "discharge_limit_ratio").mean()) if len(trajectory) else 0.0,
        "mean_import_cost": float(import_cost.mean()) if len(trajectory) else 0.0,
        "mean_reward_after_terminal_penalty": float(_safe_series(trajectory, "reward_after_terminal_penalty").mean()) if len(trajectory) else 0.0,
        "mean_final_reward": float(_safe_series(trajectory, "reward").mean()) if len(trajectory) else 0.0,
        "dominant_reward_block": dominant_block,
    }


def main() -> int:
    args = build_parser().parse_args()
    summary_csv = Path(args.summary_csv)
    trajectories_dir = Path(args.trajectories_dir) if args.trajectories_dir is not None else summary_csv.parent / "trajectories"
    output_csv = Path(args.output_csv) if args.output_csv is not None else summary_csv.with_name(f"{summary_csv.stem}_reward_diagnostics.csv")
    output_json = Path(args.output_json) if args.output_json is not None else summary_csv.with_name(f"{summary_csv.stem}_reward_diagnostics.json")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    summary_df = pd.read_csv(summary_csv)
    if summary_df.empty:
        raise ValueError(f"Summary CSV is empty: {summary_csv}")

    rows: list[dict[str, float | int | str]] = []
    for _, row in summary_df.iterrows():
        stem = _trajectory_stem(row)
        trajectory_path = trajectories_dir / f"{stem}.csv"
        if not trajectory_path.exists():
            raise FileNotFoundError(f"Trajectory CSV not found for {stem}: {trajectory_path}")
        trajectory = pd.read_csv(trajectory_path)
        diagnostic_row = {
            "case": str(row["case"]),
            "regime": str(row["regime"]),
            "reward_profile": str(row.get("reward_profile", "")),
            "agent": str(row["agent"]),
            "seed": int(row["seed"]),
            "train_model": str(row["train_model"]),
            "test_model": str(row["test_model"]),
            "train_steps": int(row["train_steps"]),
            "final_cumulative_objective_cost": float(row.get("final_cumulative_objective_cost", 0.0)),
            "final_soc": float(row.get("final_soc", 0.0)),
        }
        diagnostic_row.update(_aggregate_reward_terms(row, trajectory))
        diagnostic_row["trajectory_path"] = str(trajectory_path)
        rows.append(diagnostic_row)

    diagnostics_df = pd.DataFrame(rows)
    diagnostics_df.to_csv(output_csv, index=False)
    output_json.write_text(diagnostics_df.to_json(orient="records", indent=2), encoding="utf-8")

    display_columns = [
        "case",
        "train_model",
        "test_model",
        "seed",
        "final_cumulative_objective_cost",
        "dominant_reward_block",
        "weighted_cost_term",
        "battery_shaping_term",
        "terminal_soc_term",
        "wrapper_penalty_term",
        "clipped_step_fraction",
        "infeasible_action_dwell_fraction",
        "soc_upper_dwell_fraction",
        "soc_lower_dwell_fraction",
    ]
    print("Reward/Wrapper Diagnostics")
    print(diagnostics_df.loc[:, display_columns].to_string(index=False))
    print(f"Saved reward diagnostics CSV: {output_csv}")
    print(f"Saved reward diagnostics JSON: {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
