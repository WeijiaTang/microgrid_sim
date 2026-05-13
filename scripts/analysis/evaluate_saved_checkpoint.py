#!/usr/bin/env python3
"""Evaluate saved probe checkpoints and export a compact summary CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from microgrid_sim.rl_utils import load_agent
from scripts.analysis.short_cross_fidelity_probe import (
    apply_ieee33_full_fair_protocol,
    apply_ieee33_sac_default_protocol,
    evaluate_agent,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate saved best checkpoints from short_cross_fidelity_probe.py.")
    parser.add_argument("--case", type=str, required=True)
    parser.add_argument("--regime", type=str, default="network_stress")
    parser.add_argument("--reward-profile", type=str, default="paper_balanced")
    parser.add_argument("--agent", type=str, default="sac")
    parser.add_argument(
        "--protocol-profile",
        type=str,
        default="auto",
        choices=("auto", "none", "ieee33_sac_default", "ieee33_full_fair", "ieee33_full_fair_closure", "ieee33_full_fair_closure_gate", "ieee33_full_fair_staged_gate", "ieee33_full_fair_staged_gate_reserve"),
    )
    parser.add_argument("--train-model", type=str, required=True)
    parser.add_argument("--test-model", type=str, required=True)
    parser.add_argument("--seeds", type=str, required=True)
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--eval-year", type=int, default=2024)
    parser.add_argument("--eval-days", type=int, default=30)
    parser.add_argument("--eval-steps", type=int, default=2880)
    parser.add_argument("--eval-full-horizon", action="store_true")
    parser.add_argument("--year-start-stride-hours", type=int, default=24)
    parser.add_argument("--action-smoothing-coef", type=float, default=0.5)
    parser.add_argument("--action-max-delta", type=float, default=0.1)
    parser.add_argument("--action-rate-penalty", type=float, default=0.05)
    parser.add_argument("--battery-feasibility-aware", action="store_true", default=True)
    parser.add_argument("--battery-infeasible-penalty", type=float, default=-1.0)
    parser.add_argument("--symmetric-battery-action", action="store_true", default=True)
    parser.add_argument("--rule-guidance-mix", type=float, default=0.0)
    parser.add_argument("--rule-guidance-decay-steps", type=int, default=0)
    parser.add_argument("--rule-guidance-policy", type=str, default="rule")
    parser.add_argument("--train-validation-peak-discharge-limit-threshold", type=float, default=0.25)
    return parser


def _parse_seeds(raw: str) -> list[int]:
    seeds: list[int] = []
    for token in str(raw).split(","):
        token = token.strip()
        if token:
            seeds.append(int(token))
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def main() -> int:
    raw_argv = sys.argv[1:]
    args = build_parser().parse_args(raw_argv)
    args = apply_ieee33_sac_default_protocol(args, case_key=str(args.case), raw_argv=raw_argv)
    args = apply_ieee33_full_fair_protocol(
        args,
        case_key=str(args.case),
        train_model=str(args.train_model),
        raw_argv=raw_argv,
    )
    seeds = _parse_seeds(args.seeds)
    checkpoint_dir = REPO_ROOT / str(args.checkpoint_dir)
    output_dir = REPO_ROOT / str(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectories_dir = output_dir / "trajectories"
    trajectories_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for seed in seeds:
        checkpoint_name = f"{args.case}_{args.regime}_{args.agent}_{args.train_model.replace('+', '_to_')}_seed{seed}_best.zip"
        checkpoint_path = checkpoint_dir / checkpoint_name
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        eval_args = argparse.Namespace(
            reward_profile=str(args.reward_profile),
            agent=str(args.agent),
            days=int(args.days),
            seed=int(seed),
            regime=str(args.regime),
            eval_year=int(args.eval_year),
            eval_days=int(args.eval_days),
            eval_offset_days_within_year=0,
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
            rule_guidance_mix=float(args.rule_guidance_mix),
            rule_guidance_decay_steps=int(args.rule_guidance_decay_steps),
            rule_guidance_policy=str(args.rule_guidance_policy),
        )

        agent = load_agent(str(args.agent), str(checkpoint_path), env=None, device=str(args.device))
        summary, trajectory, _ = evaluate_agent(
            agent,
            case_key=str(args.case),
            test_model=str(args.test_model),
            regime=str(args.regime),
            args=eval_args,
        )
        trajectory_path = trajectories_dir / f"{args.case}_{args.regime}_{args.agent}_train-{args.train_model}_test-{args.test_model}_seed{seed}.csv"
        trajectory.to_csv(trajectory_path, index=False)
        rows.append(
            {
                "case": str(args.case),
                "regime": str(args.regime),
                "reward_profile": str(args.reward_profile),
                "agent": str(args.agent),
                "seed": int(seed),
                "train_model": str(args.train_model),
                "test_model": str(args.test_model),
                "source_checkpoint": str(checkpoint_path),
                "source_trajectory": str(trajectory_path),
                **summary,
            }
        )
        print(f"[eval] seed={seed} checkpoint={checkpoint_name}")

    summary_df = pd.DataFrame(rows)
    summary_csv = output_dir / "summary.csv"
    summary_json = output_dir / "summary.json"
    summary_df.to_csv(summary_csv, index=False)
    summary_json.write_text(summary_df.to_json(orient="records", indent=2), encoding="utf-8")
    print(f"Saved summary CSV: {summary_csv}")
    print(f"Saved summary JSON: {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
