#!/usr/bin/env python3
"""Trajectory-level battery morphology analysis for gate-passing IEEE33 SAC policies."""

from __future__ import annotations

import argparse
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

from microgrid_sim.rl_utils import load_agent
from scripts.analysis.short_cross_fidelity_probe import evaluate_agent, resolve_window_metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate saved gate-passing IEEE33 SAC checkpoints across quarterly windows and export battery morphology metrics."
    )
    parser.add_argument("--case", type=str, default="ieee33")
    parser.add_argument("--regime", type=str, default="network_stress")
    parser.add_argument("--reward-profile", type=str, default="paper_balanced")
    parser.add_argument("--agent", type=str, default="sac")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-year", type=int, default=2024)
    parser.add_argument("--eval-days", type=int, default=30)
    parser.add_argument("--quarter-offsets", type=str, default="0,91,182,273")
    parser.add_argument("--train-models", type=str, default="thevenin_full,thevenin_rint_only")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="results/ieee33_reasonable_drl_gate_rintfull_5k_3seeds/checkpoints",
    )
    parser.add_argument(
        "--baseline-csv",
        type=str,
        default="results/diagnostics/ieee33_gatepass_quarterly_generalization_seed42.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/diagnostics/ieee33_gatepass_battery_morphology_seed42",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--year-start-stride-hours", type=int, default=24)
    parser.add_argument("--eval-steps", type=int, default=0)
    parser.add_argument("--eval-full-horizon", action="store_true", default=True)
    parser.add_argument("--action-smoothing-coef", type=float, default=0.5)
    parser.add_argument("--action-max-delta", type=float, default=0.1)
    parser.add_argument("--action-rate-penalty", type=float, default=0.05)
    parser.add_argument("--battery-feasibility-aware", action="store_true", default=True)
    parser.add_argument("--battery-infeasible-penalty", type=float, default=-1.0)
    parser.add_argument("--symmetric-battery-action", action="store_true", default=True)
    parser.add_argument("--train-validation-peak-discharge-limit-threshold", type=float, default=0.25)
    return parser


def _parse_csv_arg(raw: str) -> list[str]:
    return [item.strip().lower() for item in str(raw).split(",") if item.strip()]


def _parse_int_csv_arg(raw: str) -> list[int]:
    values: list[int] = []
    for item in str(raw).split(","):
        token = item.strip()
        if token:
            values.append(int(token))
    return values


def _quarter_label(offset_days: int) -> str:
    labels = {0: "jan", 91: "apr", 182: "jul", 273: "oct"}
    return labels.get(int(offset_days), f"offset_{int(offset_days)}")


def _build_eval_args(args: argparse.Namespace, *, offset_days: int) -> argparse.Namespace:
    return argparse.Namespace(
        reward_profile=str(args.reward_profile),
        agent=str(args.agent),
        days=int(args.eval_days),
        seed=int(args.seed),
        regime=str(args.regime),
        eval_year=int(args.eval_year),
        eval_days=int(args.eval_days),
        eval_offset_days_within_year=int(offset_days),
        year_start_stride_hours=int(args.year_start_stride_hours),
        eval_full_horizon=bool(args.eval_full_horizon),
        eval_steps=int(args.eval_steps),
        train_validation_peak_discharge_limit_threshold=float(args.train_validation_peak_discharge_limit_threshold),
        action_smoothing_coef=float(args.action_smoothing_coef),
        action_max_delta=float(args.action_max_delta),
        action_rate_penalty=float(args.action_rate_penalty),
        battery_feasibility_aware=bool(args.battery_feasibility_aware),
        battery_infeasible_penalty=float(args.battery_infeasible_penalty),
        symmetric_battery_action=bool(args.symmetric_battery_action),
        rule_guidance_mix=0.0,
        rule_guidance_decay_steps=0,
    )


def _compute_morphology_metrics(
    trajectory: pd.DataFrame,
    summary: dict[str, float | int | str],
    *,
    discharge_scale_w: float,
) -> dict[str, float | int | str]:
    traj = trajectory.copy()
    traj["timestamp"] = pd.to_datetime(traj["timestamp"])
    dt_hours = 0.25
    discharge_energy_kwh = traj["battery_power_mw"].clip(lower=0.0) * 1000.0 * dt_hours
    charge_energy_kwh = (-traj["battery_power_mw"].clip(upper=0.0)) * 1000.0 * dt_hours
    daily = traj.groupby(traj["timestamp"].dt.date).agg(
        soc_min=("soc", "min"),
        soc_max=("soc", "max"),
        charge_energy_kwh=("battery_power_mw", lambda s: float((-s.clip(upper=0.0)).sum() * 1000.0 * dt_hours)),
        discharge_energy_kwh=("battery_power_mw", lambda s: float((s.clip(lower=0.0)).sum() * 1000.0 * dt_hours)),
    )
    daily["soc_swing"] = daily["soc_max"] - daily["soc_min"]
    peak_threshold = float(traj["price"].quantile(0.75))
    valley_threshold = float(traj["price"].quantile(0.25))
    peak_mask = traj["price"] >= peak_threshold - 1e-9
    valley_mask = traj["price"] <= valley_threshold + 1e-9
    peak_limit_ratio = (
        traj.loc[peak_mask, "battery_discharge_power_limit_w"].clip(lower=0.0) / max(float(discharge_scale_w), 1e-9)
    ).clip(lower=0.0, upper=1.0)
    hint_sign = np.sign(pd.Series(traj["rule_based_action_hint"], dtype=float))
    action_sign = np.sign(pd.Series(traj["battery_action_applied"], dtype=float))
    return {
        "mean_soc": float(traj["soc"].mean()),
        "soc_std": float(traj["soc"].std(ddof=0)),
        "soc_min": float(traj["soc"].min()),
        "soc_max": float(traj["soc"].max()),
        "mean_daily_soc_swing": float(daily["soc_swing"].mean()),
        "max_daily_soc_swing": float(daily["soc_swing"].max()),
        "mean_daily_charge_energy_kwh": float(daily["charge_energy_kwh"].mean()),
        "mean_daily_discharge_energy_kwh": float(daily["discharge_energy_kwh"].mean()),
        "charge_energy_kwh": float(charge_energy_kwh.sum()),
        "discharge_energy_kwh": float(discharge_energy_kwh.sum()),
        "peak_price_discharge_kwh": float(discharge_energy_kwh[peak_mask].sum()),
        "peak_price_charge_kwh": float(charge_energy_kwh[peak_mask].sum()),
        "valley_price_charge_kwh": float(charge_energy_kwh[valley_mask].sum()),
        "valley_price_discharge_kwh": float(discharge_energy_kwh[valley_mask].sum()),
        "peak_price_discharge_share": float(discharge_energy_kwh[peak_mask].sum() / max(discharge_energy_kwh.sum(), 1e-9)),
        "valley_price_charge_share": float(charge_energy_kwh[valley_mask].sum() / max(charge_energy_kwh.sum(), 1e-9)),
        "mean_peak_soc": float(traj.loc[peak_mask, "soc"].mean()) if bool(peak_mask.any()) else 0.0,
        "mean_valley_soc": float(traj.loc[valley_mask, "soc"].mean()) if bool(valley_mask.any()) else 0.0,
        "mean_peak_discharge_limit_ratio": float(peak_limit_ratio.mean()) if not peak_limit_ratio.empty else 0.0,
        "low_peak_discharge_limit_dwell": float((peak_limit_ratio < 0.25).mean()) if not peak_limit_ratio.empty else 0.0,
        "mean_rule_hint": float(traj["rule_based_action_hint"].mean()),
        "rule_hint_alignment": float((action_sign == hint_sign).mean()),
        "mean_p_max_trend_w": float(traj["p_max_trend_w"].mean()),
        "mean_abs_p_max_trend_w": float(traj["p_max_trend_w"].abs().mean()),
        "terminal_soc_deviation": float(traj["terminal_soc_deviation"].iloc[-1]),
        "final_soc": float(traj["soc"].iloc[-1]),
        "throughput_kwh": float(summary["total_battery_throughput_kwh"]),
        "objective_cost": float(summary["final_cumulative_objective_cost"]),
        "cost": float(summary["final_cumulative_cost"]),
    }


def main() -> int:
    args = build_parser().parse_args()
    checkpoint_dir = REPO_ROOT / str(args.checkpoint_dir)
    output_dir = REPO_ROOT / str(args.output_dir)
    trajectories_dir = output_dir / "trajectories"
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectories_dir.mkdir(parents=True, exist_ok=True)

    baseline_df = pd.read_csv(REPO_ROOT / str(args.baseline_csv))
    baseline_df["train_model"] = baseline_df["train_model"].astype(str).str.lower()
    baseline_df["test_model"] = baseline_df["test_model"].astype(str).str.lower()

    train_models = _parse_csv_arg(args.train_models)
    quarter_offsets = _parse_int_csv_arg(args.quarter_offsets)

    rows: list[dict[str, float | int | str]] = []
    for train_model in train_models:
        checkpoint_path = checkpoint_dir / f"ieee33_network_stress_sac_{train_model}_seed{int(args.seed)}_best.zip"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        agent = load_agent(str(args.agent), str(checkpoint_path), env=None, device=str(args.device))
        print(f"[load] {train_model} <- {checkpoint_path}")
        for offset_days in quarter_offsets:
            eval_args = _build_eval_args(args, offset_days=int(offset_days))
            eval_window = resolve_window_metadata(
                case_key=str(args.case),
                regime=str(args.regime),
                reward_profile=str(args.reward_profile),
                seed=int(args.seed),
                year=int(args.eval_year),
                episode_days=int(args.eval_days),
                random_start_within_year=False,
                stride_hours=int(args.year_start_stride_hours),
                start_offset_days_within_year=int(offset_days),
            )
            summary, trajectory, resolved_window = evaluate_agent(
                agent,
                case_key=str(args.case),
                test_model=str(train_model),
                regime=str(args.regime),
                args=eval_args,
                eval_window_override=eval_window,
                eval_steps_override=0,
                eval_full_horizon_override=True,
            )
            trajectory_path = trajectories_dir / f"{train_model}_{_quarter_label(int(offset_days))}_matched.csv"
            trajectory.to_csv(trajectory_path, index=False)
            row = {
                "train_model": str(train_model),
                "test_model": str(train_model),
                "quarter_offset_days": int(offset_days),
                "quarter_label": _quarter_label(int(offset_days)),
                "window_start": str(resolved_window["window_start_timestamp"]),
                "window_end": str(resolved_window["window_end_timestamp"]),
                "trajectory_csv": str(trajectory_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            }
            row.update(
                _compute_morphology_metrics(
                    trajectory,
                    summary,
                    discharge_scale_w=500_000.0,
                )
            )
            baseline_match = baseline_df[
                (baseline_df["train_model"] == str(train_model))
                & (baseline_df["test_model"] == str(train_model))
                & (baseline_df["quarter_offset_days"] == int(offset_days))
            ]
            if baseline_match.empty:
                raise ValueError(
                    "Could not find matched baseline row for "
                    f"train_model={train_model} quarter_offset_days={offset_days} in {args.baseline_csv}"
                )
            baseline_row = baseline_match.iloc[0]
            row["baseline_objective_cost"] = float(baseline_row["baseline_objective_cost"])
            row["baseline_cost"] = float(baseline_row["baseline_cost"])
            row["objective_savings_vs_none"] = float(baseline_row["objective_savings_vs_none"])
            row["cost_savings_vs_none"] = float(baseline_row["cost_savings_vs_none"])
            row["objective_savings_per_throughput_kwh"] = float(row["objective_savings_vs_none"]) / max(
                float(row["throughput_kwh"]), 1e-9
            )
            row["cost_savings_per_throughput_kwh"] = float(row["cost_savings_vs_none"]) / max(
                float(row["throughput_kwh"]), 1e-9
            )
            rows.append(row)
            print(
                "[eval] "
                f"train={train_model} quarter={_quarter_label(int(offset_days))} "
                f"obj_saving={float(row['objective_savings_vs_none']):.2f} "
                f"mean_soc={float(row['mean_soc']):.3f} "
                f"peak_limit={float(row['mean_peak_discharge_limit_ratio']):.3f}"
            )

    morphology_df = pd.DataFrame(rows).sort_values(["quarter_offset_days", "train_model"]).reset_index(drop=True)
    summary_csv = output_dir / "quarterly_morphology.csv"
    morphology_df.to_csv(summary_csv, index=False)

    full_df = morphology_df[morphology_df["train_model"] == "thevenin_full"].set_index("quarter_offset_days")
    rint_df = morphology_df[morphology_df["train_model"] == "thevenin_rint_only"].set_index("quarter_offset_days")
    comparison_rows: list[dict[str, float | int | str]] = []
    shared_columns = [
        "objective_savings_vs_none",
        "cost_savings_vs_none",
        "mean_soc",
        "final_soc",
        "throughput_kwh",
        "mean_daily_soc_swing",
        "mean_peak_discharge_limit_ratio",
        "low_peak_discharge_limit_dwell",
        "peak_price_discharge_share",
        "valley_price_charge_share",
        "mean_peak_soc",
        "mean_valley_soc",
        "objective_savings_per_throughput_kwh",
    ]
    for offset_days in quarter_offsets:
        row = {
            "quarter_offset_days": int(offset_days),
            "quarter_label": _quarter_label(int(offset_days)),
        }
        if int(offset_days) not in full_df.index or int(offset_days) not in rint_df.index:
            continue
        for column in shared_columns:
            row[f"{column}_adv_full_minus_rint"] = float(full_df.loc[int(offset_days), column]) - float(
                rint_df.loc[int(offset_days), column]
            )
        comparison_rows.append(row)
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_csv = output_dir / "full_vs_rint_morphology_diff.csv"
    comparison_df.to_csv(comparison_csv, index=False)

    print(f"Saved morphology CSV: {summary_csv}")
    print(f"Saved comparison CSV: {comparison_csv}")
    print(f"Saved trajectories: {trajectories_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
