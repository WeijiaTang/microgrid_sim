#!/usr/bin/env python3
"""Run short train-test mismatch probes on the network-first microgrid cases."""
# Ref: docs/spec/task.md (Task-ID: SPEC-SAFE-OFFLINE-RL-001)

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import replace
from functools import lru_cache
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

from microgrid_sim.cases import CIGREEuropeanLVConfig, IEEE33Config
from microgrid_sim.data.network_profiles import load_network_profiles
from microgrid_sim.envs.network_microgrid import NetworkMicrogridEnv
from microgrid_sim.envs.wrappers import (
    ContinuousActionRegularizationWrapper,
    RuleGuidedActionWrapper,
    ShieldedActionWrapper,
    compute_rule_guidance_action,
)
from microgrid_sim.rl_utils import create_agent
from microgrid_sim.time_utils import steps_per_day, steps_per_hour
from microgrid_sim.training.bc_warmstart import apply_bc_warmstart, distill_sac_actor_from_replay_buffer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Short cross-fidelity train-test probe for network microgrids.")
    parser.add_argument("--cases", type=str, default="cigre,ieee33", help="Comma-separated case keys: cigre, ieee33")
    parser.add_argument("--regimes", type=str, default="base", help="Comma-separated operating regimes: base, high_load, high_pv, network_stress, tight_soc")
    parser.add_argument(
        "--protocol-profile",
        type=str,
        default="auto",
        choices=("auto", "none", "ieee33_sac_default", "ieee33_full_fair", "ieee33_full_fair_closure", "ieee33_full_fair_closure_gate", "ieee33_full_fair_staged_gate", "ieee33_full_fair_staged_gate_reserve"),
        help=(
            "Training protocol preset. 'auto' preserves the existing IEEE33 SAC default auto-protocol for research-scale runs. "
            "'ieee33_full_fair' additionally applies a longer-budget, lower-LR, rule-guided fair protocol for final-stage thevenin_full schedules. "
            "'ieee33_full_fair_closure' keeps the same full-fidelity budget while adding terminal-aware warmstart and guidance. "
            "'ieee33_full_fair_closure_gate' additionally hardens checkpoint selection around the documented reviewer-safe dwell gate. "
            "'ieee33_full_fair_staged_gate' keeps the gate-aware checkpointing but removes training-time direct guidance and relies on a blended warmstart only. "
            "'ieee33_full_fair_staged_gate_reserve' additionally hardens peak-reserve checkpoint gating and increases training-time reserve shaping."
        ),
    )
    parser.add_argument(
        "--train-models",
        type=str,
        default="simple,thevenin",
        help="Comma-separated battery-model training specs. Supports single-stage specs like none/simple/thevenin/thevenin_loss_only/thevenin_rint_only/thevenin_rint_thermal_stress/thevenin_full and mixed specs like simple+thevenin.",
    )
    parser.add_argument("--test-models", type=str, default="simple,thevenin", help="Comma-separated battery models for evaluation")
    parser.add_argument("--reward-profile", type=str, default="network", help="Reward profile: network, paper_aligned, or paper_balanced")
    parser.add_argument(
        "--train-disable-explicit-battery-degradation-penalties",
        action="store_true",
        help="Disable explicit throughput/loss/stress penalties during training only while keeping the same penalties in validation/evaluation diagnostics.",
    )
    parser.add_argument(
        "--train-keep-explicit-battery-degradation-penalties",
        dest="train_disable_explicit_battery_degradation_penalties",
        action="store_false",
        help="Keep explicit throughput/loss/stress penalties active during training even when a protocol preset would disable them by default.",
    )
    parser.add_argument(
        "--train-peak-reserve-weight-scale",
        type=float,
        default=1.0,
        help="Optional multiplicative scale applied to reward.w_peak_reserve during training only.",
    )
    parser.add_argument(
        "--train-peak-reserve-power-floor",
        type=float,
        default=0.0,
        help="Optional override for reward.peak_reserve_power_floor during training only; <= 0 keeps the case default.",
    )
    parser.add_argument("--agent", type=str, default="sac", help="SB3 agent name")
    parser.add_argument("--train-steps", type=int, default=2000, help="Short training horizon per agent")
    parser.add_argument("--eval-steps", type=int, default=96, help="Evaluation rollout steps")
    parser.add_argument("--days", type=int, default=3, help="Environment simulation days")
    parser.add_argument("--train-year", type=int, default=0, help="Optional calendar year restriction for training data, e.g. 2023")
    parser.add_argument("--eval-year", type=int, default=0, help="Optional calendar year restriction for evaluation data, e.g. 2024")
    parser.add_argument("--train-episode-days", type=int, default=0, help="Optional training episode length when --train-year is used; defaults to --days")
    parser.add_argument("--eval-days", type=int, default=0, help="Optional evaluation window length when --eval-year is used; defaults to the full year")
    parser.add_argument("--eval-offset-days-within-year", type=int, default=0, help="Optional start-day offset inside --eval-year, useful for quarterly or monthly held-out windows")
    parser.add_argument("--train-random-start-within-year", action="store_true", help="When --train-year is set, sample training episodes from random starts within that year only")
    parser.add_argument("--year-start-stride-hours", type=int, default=24, help="Stride between admissible start times inside a yearly training window")
    parser.add_argument("--eval-full-horizon", action="store_true", help="Ignore --eval-steps and evaluate over the full configured evaluation horizon")
    parser.add_argument(
        "--train-validation-days",
        type=int,
        default=0,
        help="Optional held-out validation window length inside the training year. Uses the tail of train-year and excludes it from random training starts.",
    )
    parser.add_argument(
        "--train-validation-offset-days-within-year",
        type=str,
        default="",
        help="Optional comma-separated validation window start offsets inside the training year. Example: 0,91,182,273",
    )
    parser.add_argument(
        "--train-validation-checkpoint-every",
        type=int,
        default=0,
        help="If > 0, run held-out train-year validation every N training timesteps and keep the best checkpoint.",
    )
    parser.add_argument(
        "--train-validation-metric",
        type=str,
        default="health_objective",
        choices=(
            "objective_cost",
            "reward",
            "health_objective",
            "health_objective_gate",
            "health_objective_gate_shield",
            "inventory_value",
            # Continuous objective + inventory morphology + optional shield penalties.
            # Unlike *_gate variants, this does not add hard pass/fail penalties.
            "inventory_value_balanced",
            "inventory_value_gate",
            "inventory_value_gate_shield",
        ),
        help="Metric for train-year checkpoint selection: minimize objective_cost or maximize reward.",
    )
    parser.add_argument(
        "--train-validation-terminal-penalty-weight",
        type=float,
        default=1.0,
        help="Weight on terminal SOC penalty when using health-aware validation checkpoint selection.",
    )
    parser.add_argument(
        "--train-validation-boundary-dwell-weight",
        type=float,
        default=20000.0,
        help="Cost-scale weight on SOC boundary dwell fraction when using health-aware validation checkpoint selection.",
    )
    parser.add_argument(
        "--train-validation-infeasible-dwell-weight",
        type=float,
        default=20000.0,
        help="Cost-scale weight on infeasible-action dwell fraction when using health-aware validation checkpoint selection.",
    )
    parser.add_argument(
        "--train-validation-peak-reserve-weight",
        type=float,
        default=0.0,
        help="Additional cost-scale weight on high-price steps where the available discharge headroom falls below the configured reserve floor.",
    )
    parser.add_argument(
        "--train-validation-peak-discharge-limit-threshold",
        type=float,
        default=0.25,
        help="Normalized discharge-limit floor used by the peak-reserve validation penalty.",
    )
    parser.add_argument(
        "--train-validation-gate-dwell-threshold",
        type=float,
        default=0.05,
        help="Reviewer-safe strict upper bound on each dwell fraction when using gate-aware validation checkpoint selection.",
    )
    parser.add_argument(
        "--train-validation-gate-violation-weight",
        type=float,
        default=1_000_000.0,
        help="Large additive penalty applied per dwell-gate violation when using gate-aware validation checkpoint selection.",
    )
    parser.add_argument(
        "--train-validation-gate-peak-reserve-dwell-threshold",
        type=float,
        default=-1.0,
        help="Optional hard gate on peak-price low-discharge-limit dwell fraction; negative disables this extra gate.",
    )
    parser.add_argument(
        "--train-validation-shield-mean-delta-weight",
        type=float,
        default=0.0,
        help="Additional cost-scale weight on mean_abs_shield_delta during validation checkpoint selection.",
    )
    parser.add_argument(
        "--train-validation-shield-material-dwell-weight",
        type=float,
        default=0.0,
        help="Additional cost-scale weight on shield_material_activation_fraction during validation checkpoint selection.",
    )
    parser.add_argument(
        "--train-validation-shield-strong-dwell-weight",
        type=float,
        default=0.0,
        help="Additional cost-scale weight on shield_strong_activation_fraction during validation checkpoint selection.",
    )
    parser.add_argument(
        "--train-validation-final-soc-deviation-weight",
        type=float,
        default=0.0,
        help="Additional cost-scale weight on absolute final terminal SOC deviation during validation checkpoint selection.",
    )
    parser.add_argument(
        "--train-validation-midband-dwell-weight",
        type=float,
        default=10000.0,
        help="Additional cost-scale weight on low SOC midband dwell fraction for inventory-first validation checkpoint selection.",
    )
    parser.add_argument(
        "--train-validation-soc-target-tracking-weight",
        type=float,
        default=5000.0,
        help="Additional cost-scale weight on mean SOC deviation from terminal/target inventory for inventory-first validation checkpoint selection.",
    )
    parser.add_argument(
        "--train-validation-peak-discharge-headroom-weight",
        type=float,
        default=10000.0,
        help="Additional cost-scale weight on inadequate peak-price discharge headroom for inventory-first validation checkpoint selection.",
    )
    parser.add_argument(
        "--train-validation-valley-charge-weight",
        type=float,
        default=5000.0,
        help="Additional cost-scale weight when valley-price charging behavior undershoots available charge headroom.",
    )
    parser.add_argument(
        "--train-validation-peak-discharge-weight",
        type=float,
        default=5000.0,
        help="Additional cost-scale weight when peak-price discharge behavior undershoots available discharge headroom.",
    )
    parser.add_argument(
        "--train-validation-shield-mean-delta-threshold",
        type=float,
        default=-1.0,
        help="Optional hard gate on mean_abs_shield_delta for health_objective_gate_shield; negative disables this shield gate.",
    )
    parser.add_argument(
        "--train-validation-shield-material-dwell-threshold",
        type=float,
        default=-1.0,
        help="Optional hard gate on shield_material_activation_fraction for health_objective_gate_shield; negative falls back to --train-validation-gate-dwell-threshold.",
    )
    parser.add_argument(
        "--train-validation-shield-strong-dwell-threshold",
        type=float,
        default=-1.0,
        help="Optional hard gate on shield_strong_activation_fraction for health_objective_gate_shield; negative falls back to --train-validation-gate-dwell-threshold.",
    )
    parser.add_argument(
        "--causal-heuristic-warmstart-steps",
        type=int,
        default=0,
        help="Replay warm-start steps collected from a causal heuristic policy before SAC learning begins.",
    )
    parser.add_argument(
        "--causal-heuristic-warmstart-policy",
        type=str,
        default="blended",
        choices=("rule", "blended", "terminal_balanced"),
        help="Causal heuristic used for replay warm start when --causal-heuristic-warmstart-steps > 0.",
    )
    parser.add_argument(
        "--offline-dataset",
        type=str,
        default="",
        help="Optional offline dataset CSV used for BC-guided SAC warmstart.",
    )
    parser.add_argument(
        "--offline-dataset-controller-sources",
        type=str,
        default="",
        help="Optional comma-separated controller_source filter applied to --offline-dataset, e.g. oracle,heuristic.",
    )
    parser.add_argument(
        "--offline-dataset-max-transitions",
        type=int,
        default=0,
        help="Optional cap on replay transitions seeded from --offline-dataset; <= 0 uses all filtered rows.",
    )
    parser.add_argument(
        "--bc-pretrain-gradient-steps",
        type=int,
        default=0,
        help="Optional number of supervised actor BC updates before online SAC learning.",
    )
    parser.add_argument(
        "--bc-pretrain-batch-size",
        type=int,
        default=256,
        help="Mini-batch size for --bc-pretrain-gradient-steps.",
    )
    parser.add_argument(
        "--bc-pretrain-learning-rate",
        type=float,
        default=0.0,
        help="Optional temporary actor learning rate for BC prefit; <= 0 keeps the current optimizer LR.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--seeds", type=str, default="", help="Optional comma-separated seed list overriding --seed")
    parser.add_argument("--device", type=str, default="cpu", help="Training device")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Base learning rate for single-stage training and the default mixed-fidelity stage learning rate",
    )
    parser.add_argument("--action-smoothing-coef", type=float, default=0.0, help="Exponential smoothing coefficient for continuous actions")
    parser.add_argument("--action-max-delta", type=float, default=0.0, help="Per-step maximum action delta before clipping")
    parser.add_argument("--action-rate-penalty", type=float, default=0.0, help="Penalty weight for applied action-rate changes")
    parser.add_argument("--battery-feasibility-aware", action="store_true", help="Clip battery actions to the current SOC-feasible range before env.step")
    parser.add_argument("--battery-infeasible-penalty", type=float, default=-1.0, help="Reward adjustment per unit infeasible battery-action gap when SOC-feasible clipping is enabled")
    parser.add_argument(
        "--symmetric-battery-action",
        action="store_true",
        help="Scale positive battery actions to enforce symmetric usable charge/discharge range",
    )
    parser.add_argument(
        "--rule-guidance-mix",
        type=float,
        default=0.0,
        help="Initial blend weight for rule-based action guidance during training only; 0 disables rule guidance.",
    )
    parser.add_argument(
        "--rule-guidance-decay-steps",
        type=int,
        default=0,
        help="Environment-step horizon over which rule guidance decays to zero during training; 0 keeps a constant mix.",
    )
    parser.add_argument(
        "--rule-guidance-policy",
        type=str,
        default="rule",
        choices=("rule", "blended", "terminal_balanced"),
        help="Rule-guidance heuristic blended into the policy during training when --rule-guidance-mix > 0.",
    )
    parser.add_argument("--shield-enabled", action="store_true", help="Enable the battery action shield during training and evaluation.")
    parser.add_argument(
        "--shield-soc-soft-buffer-fraction",
        type=float,
        default=0.18,
        help="SOC soft buffer fraction used by the shield before allowing same-direction boundary actions.",
    )
    parser.add_argument(
        "--shield-soc-hard-buffer-fraction",
        type=float,
        default=0.10,
        help="SOC hard buffer fraction used by the shield for active boundary pullback.",
    )
    parser.add_argument(
        "--shield-peak-reserve-min-fraction",
        type=float,
        default=0.25,
        help="Minimum discharge-limit ratio preserved by the shield during peak-price steps.",
    )
    parser.add_argument(
        "--shield-hard-pullback-action",
        type=float,
        default=0.25,
        help="Minimum opposite-direction normalized action enforced by the shield in hard boundary zones.",
    )
    parser.add_argument(
        "--shield-terminal-closure-horizon-fraction",
        type=float,
        default=0.35,
        help="Tail-horizon fraction inside which the shield can assist terminal SOC closure.",
    )
    parser.add_argument(
        "--shield-terminal-closure-urgency-soc",
        type=float,
        default=0.20,
        help="SOC deviation scale used to convert terminal mismatch into shield closure urgency.",
    )
    parser.add_argument(
        "--shield-delta-penalty-coef",
        type=float,
        default=0.0,
        help="Optional reward penalty coefficient on |shield_post_action - shield_pre_action| to encourage policy-shield alignment.",
    )
    parser.add_argument(
        "--shield-delta-penalty-start",
        type=float,
        default=-1.0,
        help=(
            "Initial curriculum coefficient for |shield_post_action - shield_pre_action|. "
            "Negative means use --shield-delta-penalty-coef for backward-compatible static behavior."
        ),
    )
    parser.add_argument(
        "--shield-delta-penalty-end",
        type=float,
        default=-1.0,
        help=(
            "Final curriculum coefficient for |shield_post_action - shield_pre_action|. "
            "Negative means use --shield-delta-penalty-coef for backward-compatible static behavior."
        ),
    )
    parser.add_argument(
        "--shield-delta-penalty-warmup-steps",
        type=int,
        default=0,
        help="Number of environment steps over which the shield-delta penalty ramps from start to end.",
    )
    parser.add_argument(
        "--online-safe-bc-gradient-steps",
        type=int,
        default=0,
        help="Optional actor distillation gradient steps after each training chunk using replay-buffer executed actions.",
    )
    parser.add_argument(
        "--online-safe-bc-batch-size",
        type=int,
        default=256,
        help="Batch size for online replay-buffer safe behavior cloning.",
    )
    parser.add_argument(
        "--online-safe-bc-max-samples",
        type=int,
        default=0,
        help="If > 0, limit replay-buffer safe BC to the most recent N flattened transitions.",
    )
    parser.add_argument(
        "--online-safe-bc-learning-rate",
        type=float,
        default=0.0,
        help="Optional learning rate override for online replay-buffer safe behavior cloning.",
    )
    parser.add_argument(
        "--online-safe-bc-intervention-priority-coef",
        type=float,
        default=4.0,
        help="Priority weight bonus for replay samples where the shield actively corrected the action.",
    )
    parser.add_argument(
        "--online-safe-bc-boundary-priority-coef",
        type=float,
        default=2.0,
        help="Priority weight bonus for boundary-risk replay samples during safe distillation.",
    )
    parser.add_argument(
        "--online-safe-bc-terminal-priority-coef",
        type=float,
        default=2.0,
        help="Priority weight bonus for terminal-closure replay samples during safe distillation.",
    )
    parser.add_argument(
        "--online-safe-bc-reserve-priority-coef",
        type=float,
        default=1.0,
        help="Priority weight bonus for reserve-risk replay samples during safe distillation.",
    )
    parser.add_argument(
        "--online-safe-bc-teacher-priority-coef",
        type=float,
        default=2.0,
        help="Priority weight bonus for inventory-teacher replay samples during safe distillation.",
    )
    parser.add_argument(
        "--online-safe-bc-peak-value-priority-coef",
        type=float,
        default=0.75,
        help="Priority weight bonus for pure peak-value replay teacher rows during safe distillation.",
    )
    parser.add_argument(
        "--online-safe-bc-valley-value-priority-coef",
        type=float,
        default=0.5,
        help="Priority weight bonus for pure valley-value replay teacher rows during safe distillation.",
    )
    parser.add_argument(
        "--online-safe-bc-delta-priority-coef",
        type=float,
        default=2.0,
        help="Priority weight scaling on absolute shield correction magnitude during safe distillation.",
    )
    parser.add_argument(
        "--online-safe-bc-terminal-deviation-priority-coef",
        type=float,
        default=1.0,
        help="Priority weight scaling on terminal SOC deviation during safe distillation.",
    )
    parser.add_argument(
        "--online-safe-bc-small-replay-priority-scale",
        type=float,
        default=0.5,
        help="Downscale factor applied to correction-driven online Safe BC priorities when recent online replay is still small.",
    )
    parser.add_argument(
        "--online-safe-bc-small-replay-min-rows-multiplier",
        type=float,
        default=4.0,
        help="Treat recent online replay as small until it reaches ceil(batch_size * multiplier) rows before priority ramp-up begins.",
    )
    parser.add_argument(
        "--online-safe-bc-small-replay-full-strength-rows-multiplier",
        type=float,
        default=16.0,
        help=(
            "Recover full-strength online Safe BC priorities at ceil(batch_size * multiplier) rows; "
            "between the reduced-scale and full-strength thresholds the priority scale ramps up smoothly."
        ),
    )
    parser.add_argument(
        "--online-safe-bc-adaptive-scale-factor",
        type=float,
        default=2.0,
        help="Multiplier applied to online Safe BC gradient steps when validation shows weak safety internalization.",
    )
    parser.add_argument(
        "--online-safe-bc-adaptive-shield-material-threshold",
        type=float,
        default=0.40,
        help="Validation threshold above which material shield dependence triggers stronger online Safe BC.",
    )
    parser.add_argument(
        "--online-safe-bc-adaptive-shield-delta-threshold",
        type=float,
        default=0.015,
        help="Validation threshold above which mean abs shield delta triggers stronger online Safe BC.",
    )
    parser.add_argument(
        "--online-safe-bc-adaptive-inventory-teacher-activation-threshold",
        type=float,
        default=0.50,
        help="Validation threshold above which inventory-teacher activation triggers more conservative online Safe BC.",
    )
    parser.add_argument(
        "--online-safe-bc-adaptive-inventory-teacher-gap-threshold",
        type=float,
        default=0.10,
        help="Validation threshold above which mean abs inventory-teacher gap triggers more conservative online Safe BC.",
    )
    parser.add_argument(
        "--online-safe-bc-adaptive-patience",
        type=int,
        default=2,
        help="Number of non-improving validation rounds tolerated before backing off online Safe BC gradient steps.",
    )
    parser.add_argument(
        "--online-safe-bc-adaptive-max-gradient-steps",
        type=int,
        default=128,
        help="Upper cap on adaptive online Safe BC gradient steps per training chunk.",
    )
    parser.add_argument(
        "--online-safe-bc-adaptive-midband-dwell-threshold",
        type=float,
        default=0.75,
        help="Validation threshold below which weak SOC midband occupancy triggers stronger online Safe BC.",
    )
    parser.add_argument(
        "--online-safe-bc-adaptive-soc-target-mae-threshold",
        type=float,
        default=0.08,
        help="Validation threshold above which poor SOC target tracking triggers stronger online Safe BC.",
    )
    parser.add_argument(
        "--online-safe-bc-adaptive-peak-discharge-action-threshold",
        type=float,
        default=0.20,
        help="Validation threshold below which weak peak-price discharge participation triggers stronger online Safe BC.",
    )
    parser.add_argument(
        "--online-safe-bc-adaptive-valley-charge-action-threshold",
        type=float,
        default=0.20,
        help="Validation threshold below which weak valley-price charging participation triggers stronger online Safe BC.",
    )
    parser.add_argument(
        "--mixed-fidelity-pretrain-fraction",
        type=float,
        default=0.5,
        help="Default first-stage fraction for mixed-fidelity specs when explicit stage fractions are not provided.",
    )
    parser.add_argument(
        "--mixed-fidelity-stage-fractions",
        type=str,
        default="",
        help="Optional comma-separated stage fractions for mixed-fidelity specs such as simple+thevenin_loss_only+thevenin.",
    )
    parser.add_argument(
        "--mixed-fidelity-stage-learning-rates",
        type=str,
        default="",
        help="Optional comma-separated learning rates aligned with mixed-fidelity stages. Example: 3e-4,5e-5",
    )
    parser.add_argument("--tensorboard-log", type=str, default="", help="Optional TensorBoard log root directory for SB3 agents")
    parser.add_argument("--tb-log-name", type=str, default="", help="Optional TensorBoard run name prefix; defaults to a case/regime/model-specific name")
    parser.add_argument("--output-dir", type=str, default="results/short_cross_fidelity", help="Output directory")
    return parser


def _parse_csv_arg(raw: str) -> list[str]:
    return [item.strip().lower() for item in str(raw).split(",") if item.strip()]


def _parse_seed_list(raw: str, fallback_seed: int) -> list[int]:
    if not str(raw).strip():
        return [int(fallback_seed)]
    seeds: list[int] = []
    for item in str(raw).split(","):
        token = item.strip()
        if token:
            seeds.append(int(token))
    return seeds or [int(fallback_seed)]


def _parse_int_csv_arg(raw: str) -> list[int]:
    values: list[int] = []
    for item in str(raw).split(","):
        token = item.strip()
        if token:
            values.append(int(token))
    return values


IEEE33_SAC_RESEARCH_PROTOCOL_MIN_TRAIN_STEPS = 5000
IEEE33_SAC_DEFAULT_YEAR = 2023
IEEE33_SAC_DEFAULT_EVAL_YEAR = 2024
IEEE33_SAC_DEFAULT_WINDOW_DAYS = 30
IEEE33_SAC_DEFAULT_VALIDATION_DAYS = 7
IEEE33_SAC_DEFAULT_VALIDATION_OFFSETS = "0,91,182,273"
IEEE33_SAC_DEFAULT_PEAK_RESERVE_WEIGHT = 10_000.0
IEEE33_SAC_DEFAULT_PEAK_DISCHARGE_LIMIT_THRESHOLD = 0.25
IEEE33_SAC_STRICT_MORPHOLOGY_VALIDATION_METRIC = "inventory_value_gate_shield"
IEEE33_SAC_STRICT_MORPHOLOGY_GATE_DWELL_THRESHOLD = 0.05
IEEE33_SAC_STRICT_MORPHOLOGY_SHIELD_MEAN_DELTA_THRESHOLD = 0.05
IEEE33_SAC_STRICT_MORPHOLOGY_SHIELD_MATERIAL_DWELL_THRESHOLD = 0.60
IEEE33_SAC_STRICT_MORPHOLOGY_SHIELD_STRONG_DWELL_THRESHOLD = 0.20
IEEE33_SAC_STRICT_MORPHOLOGY_SHIELD_MEAN_DELTA_WEIGHT = 100_000.0
IEEE33_SAC_STRICT_MORPHOLOGY_SHIELD_MATERIAL_DWELL_WEIGHT = 50_000.0
IEEE33_SAC_STRICT_MORPHOLOGY_SHIELD_STRONG_DWELL_WEIGHT = 100_000.0
IEEE33_SAC_STRICT_MORPHOLOGY_FINAL_SOC_DEVIATION_WEIGHT = 30_000.0
IEEE33_SAC_DEFAULT_ACTION_SMOOTHING_COEF = 0.5
IEEE33_SAC_DEFAULT_ACTION_MAX_DELTA = 0.1
IEEE33_SAC_DEFAULT_ACTION_RATE_PENALTY = 0.05
IEEE33_SAC_SHORT_RUN_FINE_VALIDATION_MAX_STEPS = 10_000
IEEE33_SAC_SHORT_RUN_FINE_VALIDATION_INTERVAL = 1_000
IEEE33_FULL_FAIR_MIN_TRAIN_STEPS = 50_000
IEEE33_FULL_FAIR_DEFAULT_LEARNING_RATE = 1e-4
IEEE33_FULL_FAIR_DEFAULT_RULE_GUIDANCE_MIX = 0.2
IEEE33_FULL_FAIR_CLOSURE_DEFAULT_RULE_GUIDANCE_POLICY = "terminal_balanced"
IEEE33_FULL_FAIR_CLOSURE_DEFAULT_WARMSTART_POLICY = "terminal_balanced"
IEEE33_FULL_FAIR_CLOSURE_GATE_DEFAULT_RULE_GUIDANCE_MIX = 0.1
IEEE33_FULL_FAIR_CLOSURE_GATE_DEFAULT_RULE_GUIDANCE_POLICY = "terminal_balanced"
IEEE33_FULL_FAIR_CLOSURE_GATE_DEFAULT_WARMSTART_POLICY = "blended"
IEEE33_FULL_FAIR_STAGED_GATE_DEFAULT_RULE_GUIDANCE_MIX = 0.0
IEEE33_FULL_FAIR_STAGED_GATE_DEFAULT_RULE_GUIDANCE_POLICY = "blended"
IEEE33_FULL_FAIR_STAGED_GATE_DEFAULT_WARMSTART_POLICY = "blended"
IEEE33_FULL_FAIR_STAGED_GATE_RESERVE_DEFAULT_PEAK_RESERVE_WEIGHT = 20_000.0
IEEE33_FULL_FAIR_STAGED_GATE_RESERVE_DEFAULT_PEAK_RESERVE_GATE_THRESHOLD = 0.5
IEEE33_FULL_FAIR_STAGED_GATE_RESERVE_DEFAULT_TRAIN_PEAK_RESERVE_WEIGHT_SCALE = 3.0
IEEE33_FULL_FAIR_STAGED_GATE_RESERVE_DEFAULT_TRAIN_PEAK_RESERVE_POWER_FLOOR = 0.35
IEEE33_INVENTORY_FIRST_REWARD_PROFILE = "paper_balanced"


def _default_ieee33_validation_checkpoint_every(train_steps: int) -> int:
    steps = max(int(train_steps), 0)
    if steps <= IEEE33_SAC_SHORT_RUN_FINE_VALIDATION_MAX_STEPS:
        return IEEE33_SAC_SHORT_RUN_FINE_VALIDATION_INTERVAL
    return max(2500, steps // 4)


def _default_ieee33_full_fair_validation_checkpoint_every(train_steps: int) -> int:
    steps = max(int(train_steps), 0)
    if steps <= IEEE33_SAC_SHORT_RUN_FINE_VALIDATION_MAX_STEPS:
        return IEEE33_SAC_SHORT_RUN_FINE_VALIDATION_INTERVAL
    return max(2500, steps // 20)


def _default_ieee33_full_fair_guidance_decay_steps(train_steps: int) -> int:
    steps = max(int(train_steps), 0)
    if steps <= 0:
        return 0
    return max(2000, min(10_000, steps // 5))


def _default_ieee33_full_fair_closure_guidance_decay_steps(train_steps: int) -> int:
    steps = max(int(train_steps), 0)
    if steps <= 0:
        return 0
    return max(10_000, min(20_000, steps // 2))


def _default_ieee33_full_fair_closure_warmstart_steps(train_steps: int) -> int:
    steps = max(int(train_steps), 0)
    if steps <= 0:
        return 0
    return min(5000, steps // 10)


def _default_ieee33_full_fair_closure_gate_guidance_decay_steps(train_steps: int) -> int:
    steps = max(int(train_steps), 0)
    if steps <= 0:
        return 0
    return max(5000, min(10_000, steps // 5))


def _default_ieee33_full_fair_closure_gate_warmstart_steps(train_steps: int) -> int:
    steps = max(int(train_steps), 0)
    if steps <= 0:
        return 0
    return min(2500, steps // 20)


def _default_ieee33_full_fair_staged_gate_warmstart_steps(train_steps: int) -> int:
    steps = max(int(train_steps), 0)
    if steps <= 0:
        return 0
    return min(5000, steps // 10)


def _cli_flag_present(raw_argv: list[str], flag: str) -> bool:
    normalized = str(flag).strip()
    if not normalized:
        return False
    for token in raw_argv:
        if token == normalized or token.startswith(f"{normalized}="):
            return True
    return False




def protocol_profile(args: argparse.Namespace) -> str:
    return str(getattr(args, "protocol_profile", "auto")).strip().lower() or "auto"


def ieee33_sac_research_protocol_enabled(args: argparse.Namespace, case_key: str) -> bool:
    if str(case_key).strip().lower() != "ieee33" or str(getattr(args, "agent", "")).strip().lower() != "sac":
        return False
    preset = protocol_profile(args)
    if preset == "none":
        return False
    if preset in {"ieee33_sac_default", "ieee33_full_fair", "ieee33_full_fair_closure", "ieee33_full_fair_closure_gate", "ieee33_full_fair_staged_gate", "ieee33_full_fair_staged_gate_reserve"}:
        return True
    return int(getattr(args, "train_steps", 0)) >= IEEE33_SAC_RESEARCH_PROTOCOL_MIN_TRAIN_STEPS


def apply_ieee33_sac_default_protocol(args: argparse.Namespace, *, case_key: str, raw_argv: list[str]) -> argparse.Namespace:
    run_args = argparse.Namespace(**vars(args))
    run_args.ieee33_sac_default_protocol_applied = False
    if not ieee33_sac_research_protocol_enabled(run_args, case_key=case_key):
        return run_args

    if protocol_profile(run_args) in {"ieee33_sac_default", "ieee33_full_fair", "ieee33_full_fair_closure", "ieee33_full_fair_closure_gate", "ieee33_full_fair_staged_gate", "ieee33_full_fair_staged_gate_reserve"} and not _cli_flag_present(raw_argv, "--train-steps"):
        run_args.train_steps = max(int(getattr(run_args, "train_steps", 0)), IEEE33_SAC_RESEARCH_PROTOCOL_MIN_TRAIN_STEPS)

    if not _cli_flag_present(raw_argv, "--days"):
        run_args.days = max(int(getattr(run_args, "days", 0)), IEEE33_SAC_DEFAULT_WINDOW_DAYS)
    if not _cli_flag_present(raw_argv, "--train-year"):
        run_args.train_year = IEEE33_SAC_DEFAULT_YEAR
    if not _cli_flag_present(raw_argv, "--eval-year"):
        run_args.eval_year = IEEE33_SAC_DEFAULT_EVAL_YEAR
    if not _cli_flag_present(raw_argv, "--reward-profile"):
        run_args.reward_profile = IEEE33_INVENTORY_FIRST_REWARD_PROFILE
    if not _cli_flag_present(raw_argv, "--train-episode-days"):
        run_args.train_episode_days = int(getattr(run_args, "train_episode_days", 0)) or int(getattr(run_args, "days", IEEE33_SAC_DEFAULT_WINDOW_DAYS))
    if not _cli_flag_present(raw_argv, "--eval-days"):
        run_args.eval_days = int(getattr(run_args, "eval_days", 0)) or int(getattr(run_args, "days", IEEE33_SAC_DEFAULT_WINDOW_DAYS))
    if not _cli_flag_present(raw_argv, "--train-random-start-within-year"):
        run_args.train_random_start_within_year = True
    if not _cli_flag_present(raw_argv, "--eval-full-horizon"):
        run_args.eval_full_horizon = True

    if not _cli_flag_present(raw_argv, "--train-validation-days"):
        run_args.train_validation_days = IEEE33_SAC_DEFAULT_VALIDATION_DAYS
    if not _cli_flag_present(raw_argv, "--train-validation-offset-days-within-year"):
        run_args.train_validation_offset_days_within_year = IEEE33_SAC_DEFAULT_VALIDATION_OFFSETS
    if not _cli_flag_present(raw_argv, "--train-validation-checkpoint-every"):
        run_args.train_validation_checkpoint_every = _default_ieee33_validation_checkpoint_every(
            int(getattr(run_args, "train_steps", 0))
        )
    if not _cli_flag_present(raw_argv, "--train-validation-metric"):
        # Strict validation-best checkpointing: preserve battery inventory
        # morphology while rejecting checkpoints that still rely heavily on
        # shield corrections. Final evaluation restores these best parameters.
        run_args.train_validation_metric = IEEE33_SAC_STRICT_MORPHOLOGY_VALIDATION_METRIC
    if not _cli_flag_present(raw_argv, "--train-validation-gate-dwell-threshold"):
        run_args.train_validation_gate_dwell_threshold = IEEE33_SAC_STRICT_MORPHOLOGY_GATE_DWELL_THRESHOLD
    if not _cli_flag_present(raw_argv, "--train-validation-shield-mean-delta-threshold"):
        run_args.train_validation_shield_mean_delta_threshold = (
            IEEE33_SAC_STRICT_MORPHOLOGY_SHIELD_MEAN_DELTA_THRESHOLD
        )
    if not _cli_flag_present(raw_argv, "--train-validation-shield-material-dwell-threshold"):
        run_args.train_validation_shield_material_dwell_threshold = (
            IEEE33_SAC_STRICT_MORPHOLOGY_SHIELD_MATERIAL_DWELL_THRESHOLD
        )
    if not _cli_flag_present(raw_argv, "--train-validation-shield-strong-dwell-threshold"):
        run_args.train_validation_shield_strong_dwell_threshold = (
            IEEE33_SAC_STRICT_MORPHOLOGY_SHIELD_STRONG_DWELL_THRESHOLD
        )
    if not _cli_flag_present(raw_argv, "--train-validation-shield-mean-delta-weight"):
        run_args.train_validation_shield_mean_delta_weight = (
            IEEE33_SAC_STRICT_MORPHOLOGY_SHIELD_MEAN_DELTA_WEIGHT
        )
    if not _cli_flag_present(raw_argv, "--train-validation-shield-material-dwell-weight"):
        run_args.train_validation_shield_material_dwell_weight = (
            IEEE33_SAC_STRICT_MORPHOLOGY_SHIELD_MATERIAL_DWELL_WEIGHT
        )
    if not _cli_flag_present(raw_argv, "--train-validation-shield-strong-dwell-weight"):
        run_args.train_validation_shield_strong_dwell_weight = (
            IEEE33_SAC_STRICT_MORPHOLOGY_SHIELD_STRONG_DWELL_WEIGHT
        )
    if not _cli_flag_present(raw_argv, "--train-validation-final-soc-deviation-weight"):
        run_args.train_validation_final_soc_deviation_weight = (
            IEEE33_SAC_STRICT_MORPHOLOGY_FINAL_SOC_DEVIATION_WEIGHT
        )
    if not _cli_flag_present(raw_argv, "--train-validation-peak-reserve-weight"):
        run_args.train_validation_peak_reserve_weight = IEEE33_SAC_DEFAULT_PEAK_RESERVE_WEIGHT
    if not _cli_flag_present(raw_argv, "--train-validation-peak-discharge-limit-threshold"):
        run_args.train_validation_peak_discharge_limit_threshold = IEEE33_SAC_DEFAULT_PEAK_DISCHARGE_LIMIT_THRESHOLD
    if not _cli_flag_present(raw_argv, "--action-smoothing-coef"):
        run_args.action_smoothing_coef = IEEE33_SAC_DEFAULT_ACTION_SMOOTHING_COEF
    if not _cli_flag_present(raw_argv, "--action-max-delta"):
        run_args.action_max_delta = IEEE33_SAC_DEFAULT_ACTION_MAX_DELTA
    if not _cli_flag_present(raw_argv, "--action-rate-penalty"):
        run_args.action_rate_penalty = IEEE33_SAC_DEFAULT_ACTION_RATE_PENALTY
    if not _cli_flag_present(raw_argv, "--battery-feasibility-aware"):
        run_args.battery_feasibility_aware = True
    if not _cli_flag_present(raw_argv, "--symmetric-battery-action"):
        run_args.symmetric_battery_action = True

    run_args.ieee33_sac_default_protocol_applied = True
    return run_args


def train_spec_targets_thevenin_full(train_spec: str) -> bool:
    return _parse_train_spec(train_spec)[-1] == "thevenin_full"


def ieee33_full_fair_protocol_enabled(args: argparse.Namespace, *, case_key: str, train_model: str) -> bool:
    return (
        protocol_profile(args) in {"ieee33_full_fair", "ieee33_full_fair_closure", "ieee33_full_fair_closure_gate", "ieee33_full_fair_staged_gate", "ieee33_full_fair_staged_gate_reserve"}
        and str(case_key).strip().lower() == "ieee33"
        and str(getattr(args, "agent", "")).strip().lower() == "sac"
        and train_spec_targets_thevenin_full(train_model)
    )


def apply_ieee33_full_fair_protocol(
    args: argparse.Namespace,
    *,
    case_key: str,
    train_model: str,
    raw_argv: list[str],
) -> argparse.Namespace:
    run_args = argparse.Namespace(**vars(args))
    run_args.ieee33_full_fair_protocol_applied = False
    run_args.ieee33_full_fair_closure_protocol_applied = False
    if not ieee33_full_fair_protocol_enabled(run_args, case_key=case_key, train_model=train_model):
        return run_args

    preset = protocol_profile(run_args)
    if not _cli_flag_present(raw_argv, "--reward-profile"):
        run_args.reward_profile = IEEE33_INVENTORY_FIRST_REWARD_PROFILE
    if not _cli_flag_present(raw_argv, "--train-steps"):
        run_args.train_steps = max(int(getattr(run_args, "train_steps", 0)), IEEE33_FULL_FAIR_MIN_TRAIN_STEPS)
    if not _cli_flag_present(raw_argv, "--learning-rate"):
        run_args.learning_rate = float(IEEE33_FULL_FAIR_DEFAULT_LEARNING_RATE)
    if not any(
        _cli_flag_present(raw_argv, flag)
        for flag in (
            "--train-disable-explicit-battery-degradation-penalties",
            "--train-keep-explicit-battery-degradation-penalties",
        )
    ):
        run_args.train_disable_explicit_battery_degradation_penalties = True
    if not _cli_flag_present(raw_argv, "--train-validation-checkpoint-every"):
        run_args.train_validation_checkpoint_every = _default_ieee33_full_fair_validation_checkpoint_every(
            int(getattr(run_args, "train_steps", 0))
        )
    if not _cli_flag_present(raw_argv, "--rule-guidance-mix"):
        if preset == "ieee33_full_fair_closure_gate":
            run_args.rule_guidance_mix = float(IEEE33_FULL_FAIR_CLOSURE_GATE_DEFAULT_RULE_GUIDANCE_MIX)
        elif preset in {"ieee33_full_fair_staged_gate", "ieee33_full_fair_staged_gate_reserve"}:
            run_args.rule_guidance_mix = float(IEEE33_FULL_FAIR_STAGED_GATE_DEFAULT_RULE_GUIDANCE_MIX)
        else:
            run_args.rule_guidance_mix = float(IEEE33_FULL_FAIR_DEFAULT_RULE_GUIDANCE_MIX)
    if not _cli_flag_present(raw_argv, "--rule-guidance-decay-steps"):
        if preset == "ieee33_full_fair_closure":
            run_args.rule_guidance_decay_steps = _default_ieee33_full_fair_closure_guidance_decay_steps(
                int(getattr(run_args, "train_steps", 0))
            )
        elif preset == "ieee33_full_fair_closure_gate":
            run_args.rule_guidance_decay_steps = _default_ieee33_full_fair_closure_gate_guidance_decay_steps(
                int(getattr(run_args, "train_steps", 0))
            )
        elif preset in {"ieee33_full_fair_staged_gate", "ieee33_full_fair_staged_gate_reserve"}:
            run_args.rule_guidance_decay_steps = 0
        else:
            run_args.rule_guidance_decay_steps = _default_ieee33_full_fair_guidance_decay_steps(
                int(getattr(run_args, "train_steps", 0))
            )
    if preset == "ieee33_full_fair_closure":
        if not _cli_flag_present(raw_argv, "--rule-guidance-policy"):
            run_args.rule_guidance_policy = IEEE33_FULL_FAIR_CLOSURE_DEFAULT_RULE_GUIDANCE_POLICY
        if not _cli_flag_present(raw_argv, "--causal-heuristic-warmstart-steps"):
            run_args.causal_heuristic_warmstart_steps = _default_ieee33_full_fair_closure_warmstart_steps(
                int(getattr(run_args, "train_steps", 0))
            )
        if not _cli_flag_present(raw_argv, "--causal-heuristic-warmstart-policy"):
            run_args.causal_heuristic_warmstart_policy = IEEE33_FULL_FAIR_CLOSURE_DEFAULT_WARMSTART_POLICY
        run_args.ieee33_full_fair_closure_protocol_applied = True
    if preset == "ieee33_full_fair_closure_gate":
        if not _cli_flag_present(raw_argv, "--rule-guidance-policy"):
            run_args.rule_guidance_policy = IEEE33_FULL_FAIR_CLOSURE_GATE_DEFAULT_RULE_GUIDANCE_POLICY
        if not _cli_flag_present(raw_argv, "--causal-heuristic-warmstart-steps"):
            run_args.causal_heuristic_warmstart_steps = _default_ieee33_full_fair_closure_gate_warmstart_steps(
                int(getattr(run_args, "train_steps", 0))
            )
        if not _cli_flag_present(raw_argv, "--causal-heuristic-warmstart-policy"):
            run_args.causal_heuristic_warmstart_policy = IEEE33_FULL_FAIR_CLOSURE_GATE_DEFAULT_WARMSTART_POLICY
        if not _cli_flag_present(raw_argv, "--train-validation-metric"):
            run_args.train_validation_metric = IEEE33_SAC_STRICT_MORPHOLOGY_VALIDATION_METRIC
        run_args.ieee33_full_fair_closure_protocol_applied = True
    if preset in {"ieee33_full_fair_staged_gate", "ieee33_full_fair_staged_gate_reserve"}:
        if not _cli_flag_present(raw_argv, "--rule-guidance-policy"):
            run_args.rule_guidance_policy = IEEE33_FULL_FAIR_STAGED_GATE_DEFAULT_RULE_GUIDANCE_POLICY
        if not _cli_flag_present(raw_argv, "--causal-heuristic-warmstart-steps"):
            run_args.causal_heuristic_warmstart_steps = _default_ieee33_full_fair_staged_gate_warmstart_steps(
                int(getattr(run_args, "train_steps", 0))
            )
        if not _cli_flag_present(raw_argv, "--causal-heuristic-warmstart-policy"):
            run_args.causal_heuristic_warmstart_policy = IEEE33_FULL_FAIR_STAGED_GATE_DEFAULT_WARMSTART_POLICY
        if not _cli_flag_present(raw_argv, "--train-validation-metric"):
            run_args.train_validation_metric = IEEE33_SAC_STRICT_MORPHOLOGY_VALIDATION_METRIC
        run_args.ieee33_full_fair_closure_protocol_applied = True
    if preset == "ieee33_full_fair_staged_gate_reserve":
        if not _cli_flag_present(raw_argv, "--train-validation-peak-reserve-weight"):
            run_args.train_validation_peak_reserve_weight = IEEE33_FULL_FAIR_STAGED_GATE_RESERVE_DEFAULT_PEAK_RESERVE_WEIGHT
        if not _cli_flag_present(raw_argv, "--train-validation-gate-peak-reserve-dwell-threshold"):
            run_args.train_validation_gate_peak_reserve_dwell_threshold = (
                IEEE33_FULL_FAIR_STAGED_GATE_RESERVE_DEFAULT_PEAK_RESERVE_GATE_THRESHOLD
            )
        if not _cli_flag_present(raw_argv, "--train-peak-reserve-weight-scale"):
            run_args.train_peak_reserve_weight_scale = IEEE33_FULL_FAIR_STAGED_GATE_RESERVE_DEFAULT_TRAIN_PEAK_RESERVE_WEIGHT_SCALE
        if not _cli_flag_present(raw_argv, "--train-peak-reserve-power-floor"):
            run_args.train_peak_reserve_power_floor = IEEE33_FULL_FAIR_STAGED_GATE_RESERVE_DEFAULT_TRAIN_PEAK_RESERVE_POWER_FLOOR
    run_args.ieee33_full_fair_protocol_applied = True
    return run_args


def _parse_train_spec(train_spec: str) -> list[str]:
    stages = [part.strip().lower() for part in str(train_spec).split("+") if part.strip()]
    if not stages:
        raise ValueError(f"Unsupported empty training spec '{train_spec}'.")
    for stage in stages:
        if stage not in {
            "none",
            "simple",
            "thevenin",
            "thevenin_loss_only",
            "thevenin_rint_only",
            "thevenin_rint_thermal_stress",
            "thevenin_full",
        }:
            raise ValueError(
                "Unsupported training stage "
                f"'{stage}' in '{train_spec}'. Expected none/simple/thevenin/thevenin_loss_only/"
                "thevenin_rint_only/thevenin_rint_thermal_stress/thevenin_full."
            )
    return stages


def _parse_stage_learning_rates(stage_count: int, raw_learning_rates: str = "", default_learning_rate: float = 3e-4) -> list[float]:
    if stage_count <= 0:
        raise ValueError("stage_count must be positive.")
    if not str(raw_learning_rates).strip():
        return [float(default_learning_rate)] * stage_count
    tokens = [token.strip() for token in str(raw_learning_rates).split(",") if token.strip()]
    if len(tokens) != stage_count:
        raise ValueError(
            f"Expected {stage_count} mixed-fidelity stage learning rates, got {len(tokens)} from '{raw_learning_rates}'."
        )
    learning_rates = [float(token) for token in tokens]
    if any(value <= 0.0 for value in learning_rates):
        raise ValueError(f"Mixed-fidelity stage learning rates must be positive, got '{raw_learning_rates}'.")
    return learning_rates


def _resolve_stage_fractions(stage_count: int, pretrain_fraction: float, raw_fractions: str = "") -> list[float]:
    if stage_count <= 0:
        raise ValueError("stage_count must be positive.")
    if stage_count == 1:
        return [1.0]

    raw_tokens = [token.strip() for token in str(raw_fractions).split(",") if token.strip()]
    if raw_tokens:
        if len(raw_tokens) != stage_count:
            raise ValueError(
                f"Expected {stage_count} mixed-fidelity stage fractions, got {len(raw_tokens)} from '{raw_fractions}'."
            )
        fractions = [float(token) for token in raw_tokens]
        if any(value < 0.0 for value in fractions):
            raise ValueError(f"Mixed-fidelity stage fractions must be non-negative, got '{raw_fractions}'.")
        total = float(sum(fractions))
        if total <= 0.0:
            raise ValueError(f"Mixed-fidelity stage fractions must sum to a positive value, got '{raw_fractions}'.")
        return [value / total for value in fractions]

    first_fraction = float(min(max(pretrain_fraction, 0.0), 1.0))
    remaining_fraction = max(0.0, 1.0 - first_fraction)
    if stage_count == 2:
        return [first_fraction, remaining_fraction]
    tail_fraction = remaining_fraction / float(stage_count - 1)
    return [first_fraction, *([tail_fraction] * (stage_count - 1))]


def _train_stage_steps(total_steps: int, stage_count: int, pretrain_fraction: float, raw_fractions: str = "") -> list[int]:
    total_steps = max(int(total_steps), 0)
    fractions = _resolve_stage_fractions(
        stage_count=stage_count,
        pretrain_fraction=pretrain_fraction,
        raw_fractions=raw_fractions,
    )
    raw_steps = [fraction * float(total_steps) for fraction in fractions]
    stage_steps = [int(value) for value in raw_steps]
    remainder = total_steps - sum(stage_steps)
    if remainder > 0:
        residual_order = sorted(
            range(stage_count),
            key=lambda index: (raw_steps[index] - stage_steps[index], -index),
            reverse=True,
        )
        for index in residual_order[:remainder]:
            stage_steps[index] += 1
    return stage_steps


def resolve_training_schedule(train_model: str, args: argparse.Namespace) -> dict[str, list[str] | list[int] | list[float] | int]:
    stages = _parse_train_spec(train_model)
    stage_fractions = _resolve_stage_fractions(
        stage_count=len(stages),
        pretrain_fraction=float(getattr(args, "mixed_fidelity_pretrain_fraction", 0.5)),
        raw_fractions=str(getattr(args, "mixed_fidelity_stage_fractions", "")),
    )
    stage_steps = _train_stage_steps(
        total_steps=int(args.train_steps),
        stage_count=len(stages),
        pretrain_fraction=float(getattr(args, "mixed_fidelity_pretrain_fraction", 0.5)),
        raw_fractions=str(getattr(args, "mixed_fidelity_stage_fractions", "")),
    )
    stage_learning_rates = _parse_stage_learning_rates(
        stage_count=len(stages),
        raw_learning_rates=str(getattr(args, "mixed_fidelity_stage_learning_rates", "")),
        default_learning_rate=float(getattr(args, "learning_rate", 3e-4)),
    )
    return {
        "stages": stages,
        "stage_count": len(stages),
        "stage_fractions": [float(value) for value in stage_fractions],
        "stage_steps": [int(value) for value in stage_steps],
        "stage_learning_rates": [float(value) for value in stage_learning_rates],
    }


def _set_optimizer_learning_rate(optimizer: Any, learning_rate: float) -> None:
    if optimizer is None:
        return
    for param_group in getattr(optimizer, "param_groups", []):
        param_group["lr"] = float(learning_rate)


def _set_agent_learning_rate(agent: Any, learning_rate: float) -> None:
    learning_rate = float(learning_rate)
    if hasattr(agent, "learning_rate"):
        agent.learning_rate = learning_rate

    lr_schedule = getattr(agent, "lr_schedule", None)
    if callable(lr_schedule):
        agent.lr_schedule = lambda _progress_remaining, lr=learning_rate: lr

    for attr_name in ("actor", "critic", "critic_target", "actor_target"):
        module = getattr(agent, attr_name, None)
        _set_optimizer_learning_rate(getattr(module, "optimizer", None), learning_rate)

    policy = getattr(agent, "policy", None)
    if policy is not None:
        _set_optimizer_learning_rate(getattr(policy, "optimizer", None), learning_rate)

    for optimizer_name in ("ent_coef_optimizer",):
        _set_optimizer_learning_rate(getattr(agent, optimizer_name, None), learning_rate)

    if hasattr(agent, "_policy"):
        policy_container = getattr(agent, "_policy")
        learn_mode = getattr(policy_container, "learn_mode", None)
        if hasattr(learn_mode, "_optimizer_actor"):
            _set_optimizer_learning_rate(getattr(learn_mode, "_optimizer_actor", None), learning_rate)
        if hasattr(learn_mode, "_optimizer_critic"):
            _set_optimizer_learning_rate(getattr(learn_mode, "_optimizer_critic", None), learning_rate)


def build_config(case_key: str, battery_model: str, days: int, seed: int, regime: str, reward_profile: str):
    if case_key == "ieee33":
        return IEEE33Config(
            simulation_days=days,
            seed=seed,
            battery_model=battery_model,
            regime=regime,
            reward_profile=reward_profile,
        )
    if case_key == "cigre":
        return CIGREEuropeanLVConfig(
            simulation_days=days,
            seed=seed,
            battery_model=battery_model,
            regime=regime,
            reward_profile=reward_profile,
        )
    raise ValueError(f"Unsupported case '{case_key}'")


def maybe_disable_explicit_battery_degradation_penalties(config, *, disable: bool):
    if not bool(disable):
        return config
    patched = replace(config)
    patched.battery_throughput_penalty_per_kwh = 0.0
    patched.battery_loss_penalty_per_kwh = 0.0
    patched.battery_stress_penalty_per_kwh = 0.0
    return patched


def maybe_patch_training_peak_reserve_reward(
    config,
    *,
    training: bool,
    weight_scale: float,
    power_floor: float,
):
    if not bool(training):
        return config
    scale = max(float(weight_scale), 0.0)
    floor = float(power_floor)
    if scale == 1.0 and floor <= 0.0:
        return config
    patched = replace(config)
    reward_cfg = getattr(patched, "reward", None)
    if reward_cfg is None:
        return patched
    patched.reward = replace(
        reward_cfg,
        w_peak_reserve=max(float(getattr(reward_cfg, "w_peak_reserve", 0.0)) * scale, 0.0),
        peak_reserve_power_floor=max(float(floor), float(getattr(reward_cfg, "peak_reserve_power_floor", 0.0)))
        if floor > 0.0
        else float(getattr(reward_cfg, "peak_reserve_power_floor", 0.0)),
    )
    return patched


@lru_cache(maxsize=None)
def resolve_year_window(case_key: str, year: int, regime: str, reward_profile: str, seed: int) -> dict[str, int | str]:
    probe_config = build_config(
        case_key=case_key,
        battery_model="simple",
        days=1,
        seed=int(seed),
        regime=regime,
        reward_profile=reward_profile,
    )
    full_profiles = load_network_profiles(probe_config)
    timestamps = pd.DatetimeIndex(full_profiles.timestamps)
    mask = timestamps.year == int(year)
    if not mask.any():
        available_years = sorted({int(value) for value in timestamps.year})
        raise ValueError(f"Requested year {year} is unavailable for case '{case_key}'. Available years: {available_years}")
    indices = np.flatnonzero(mask.to_numpy() if hasattr(mask, "to_numpy") else np.asarray(mask, dtype=bool))
    start_step = int(indices[0])
    steps = int(len(indices))
    dt_steps_per_hour = steps_per_hour(probe_config.dt_seconds)
    dt_steps_per_day = steps_per_day(probe_config.dt_seconds)
    if start_step % dt_steps_per_hour != 0:
        raise ValueError(f"Year {year} for case '{case_key}' does not start on an hour boundary.")
    if steps % dt_steps_per_day != 0:
        raise ValueError(f"Year {year} for case '{case_key}' does not span an integer number of days.")
    expected = np.arange(start_step, start_step + steps, dtype=int)
    if not np.array_equal(indices, expected):
        raise ValueError(f"Year {year} for case '{case_key}' is not contiguous in the canonical dataset.")
    return {
        "year": int(year),
        "start_step": int(start_step),
        "start_hour": int(start_step // dt_steps_per_hour),
        "steps": int(steps),
        "days": int(steps // dt_steps_per_day),
        "start_timestamp": str(timestamps[start_step]),
        "end_timestamp": str(timestamps[start_step + steps - 1]),
    }


def resolve_window_metadata(
    *,
    case_key: str,
    regime: str,
    reward_profile: str,
    seed: int,
    year: int,
    episode_days: int,
    random_start_within_year: bool,
    stride_hours: int,
    start_offset_days_within_year: int = 0,
    exclude_tail_days: int = 0,
    excluded_window_offset_days_within_year: tuple[int, ...] = tuple(),
    excluded_window_days: int = 0,
) -> dict[str, int | str | tuple[int, ...] | bool] | None:
    probe_config = build_config(
        case_key=case_key,
        battery_model="simple",
        days=1,
        seed=int(seed),
        regime=regime,
        reward_profile=reward_profile,
    )
    full_profiles = load_network_profiles(probe_config)
    if int(year) <= 0:
        return None
    year_window = resolve_year_window(
        case_key=case_key,
        year=int(year),
        regime=regime,
        reward_profile=reward_profile,
        seed=int(seed),
    )
    max_days = int(year_window["days"])
    resolved_episode_days = max(int(episode_days), 1)
    offset_days = max(int(start_offset_days_within_year), 0)
    reserved_tail_days = max(int(exclude_tail_days), 0)
    excluded_offsets = tuple(sorted({max(int(value), 0) for value in excluded_window_offset_days_within_year}))
    excluded_window_span_days = max(int(excluded_window_days), 0)
    if offset_days >= max_days:
        raise ValueError(
            f"Requested start_offset_days_within_year={offset_days} exceeds available days={max_days} for year {year} case '{case_key}'."
        )
    if offset_days + resolved_episode_days + reserved_tail_days > max_days:
        raise ValueError(
            f"Requested window offset {offset_days} plus episode_days={resolved_episode_days} and exclude_tail_days={reserved_tail_days} exceeds available days={max_days} for year {year} case '{case_key}'."
        )
    resolved_stride_hours = max(int(stride_hours), 1)
    explicit_start_hours: tuple[int, ...] = tuple()
    random_episode_start = bool(random_start_within_year and resolved_episode_days < max_days)
    base_start_hour = int(year_window["start_hour"]) + offset_days * 24
    base_start_step = int(year_window["start_step"]) + offset_days * 24 * steps_per_hour(probe_config.dt_seconds)
    if random_episode_start:
        admissible_days = max_days - offset_days - reserved_tail_days - resolved_episode_days
        candidate_start_hours: list[int] = []
        for offset_hour in range(0, admissible_days * 24 + 1, resolved_stride_hours):
            candidate_start_hour = base_start_hour + offset_hour
            if excluded_offsets and excluded_window_span_days > 0:
                candidate_start_day = offset_days + (offset_hour / 24.0)
                candidate_end_day = candidate_start_day + resolved_episode_days
                overlaps_excluded_window = any(
                    candidate_start_day < float(excluded_offset + excluded_window_span_days)
                    and candidate_end_day > float(excluded_offset)
                    for excluded_offset in excluded_offsets
                )
                if overlaps_excluded_window:
                    continue
            candidate_start_hours.append(int(candidate_start_hour))
        explicit_start_hours = tuple(candidate_start_hours)
        if not explicit_start_hours:
            raise ValueError(
                "No admissible random training starts remain after excluding validation windows; "
                f"year={year} case='{case_key}' episode_days={resolved_episode_days} "
                f"excluded_offsets={excluded_offsets} excluded_window_days={excluded_window_span_days}."
            )
    dt_steps_per_day = steps_per_day(probe_config.dt_seconds)
    resolved_end_step = base_start_step + resolved_episode_days * dt_steps_per_day - 1
    return {
        "year": int(year),
        "start_hour": int(base_start_hour),
        "start_step": int(base_start_step),
        "days": int(resolved_episode_days),
        "window_days": int(year_window["days"]),
        "window_steps": int(year_window["steps"]),
        "window_start_timestamp": str(full_profiles.timestamps[base_start_step]),
        "window_end_timestamp": str(full_profiles.timestamps[resolved_end_step]),
        "random_episode_start": bool(random_episode_start),
        "full_year_random_start_hours": explicit_start_hours,
        "full_year_random_start_stride_hours": int(resolved_stride_hours),
        "excluded_tail_days": int(reserved_tail_days),
    }


def action_regularization_config(args: argparse.Namespace) -> dict[str, float | bool]:
    battery_feasibility_aware = bool(getattr(args, "battery_feasibility_aware", False))
    battery_infeasible_penalty = float(getattr(args, "battery_infeasible_penalty", -1.0))
    if not battery_feasibility_aware:
        battery_infeasible_penalty = 0.0
    return {
        "smoothing_coef": float(getattr(args, "action_smoothing_coef", 0.0)),
        "max_delta": float(getattr(args, "action_max_delta", 0.0)),
        "rate_penalty": float(getattr(args, "action_rate_penalty", 0.0)),
        "battery_feasibility_aware": battery_feasibility_aware,
        "battery_infeasible_penalty": battery_infeasible_penalty,
        "symmetric_battery_action": bool(getattr(args, "symmetric_battery_action", False)),
    }


def action_regularization_enabled(args: argparse.Namespace) -> bool:
    config = action_regularization_config(args)
    return any(
        (
            float(config["smoothing_coef"]) > 0.0,
            float(config["max_delta"]) > 0.0,
            float(config["rate_penalty"]) > 0.0,
            bool(config["battery_feasibility_aware"]),
            float(config["battery_infeasible_penalty"]) != 0.0,
            bool(config["symmetric_battery_action"]),
        )
    )


def rule_guidance_config(args: argparse.Namespace) -> dict[str, float | int]:
    return {
        "guidance_mix": float(np.clip(float(getattr(args, "rule_guidance_mix", 0.0)), 0.0, 1.0)),
        "guidance_decay_steps": max(int(getattr(args, "rule_guidance_decay_steps", 0)), 0),
        "guidance_policy": str(getattr(args, "rule_guidance_policy", "rule")).strip().lower() or "rule",
    }


def rule_guidance_enabled(args: argparse.Namespace) -> bool:
    return float(rule_guidance_config(args)["guidance_mix"]) > 0.0


def shield_config(args: argparse.Namespace) -> dict[str, float | int]:
    static_delta_penalty = max(float(getattr(args, "shield_delta_penalty_coef", 0.0)), 0.0)
    raw_start = float(getattr(args, "shield_delta_penalty_start", -1.0))
    raw_end = float(getattr(args, "shield_delta_penalty_end", -1.0))
    return {
        "reserve_discharge_min_fraction": float(np.clip(float(getattr(args, "shield_peak_reserve_min_fraction", 0.25)), 0.0, 1.0)),
        "soc_soft_buffer_fraction": max(float(getattr(args, "shield_soc_soft_buffer_fraction", 0.18)), 0.0),
        "soc_hard_buffer_fraction": max(float(getattr(args, "shield_soc_hard_buffer_fraction", 0.10)), 0.0),
        "hard_pullback_action": float(np.clip(float(getattr(args, "shield_hard_pullback_action", 0.25)), 0.0, 1.0)),
        "terminal_closure_horizon_fraction": float(
            np.clip(float(getattr(args, "shield_terminal_closure_horizon_fraction", 0.35)), 0.0, 1.0)
        ),
        "terminal_closure_urgency_soc": max(float(getattr(args, "shield_terminal_closure_urgency_soc", 0.20)), 1e-6),
        "shield_delta_penalty_coef": static_delta_penalty,
        "shield_delta_penalty_start": static_delta_penalty if raw_start < 0.0 else max(raw_start, 0.0),
        "shield_delta_penalty_end": static_delta_penalty if raw_end < 0.0 else max(raw_end, 0.0),
        "shield_delta_penalty_warmup_steps": max(int(getattr(args, "shield_delta_penalty_warmup_steps", 0)), 0),
    }


def shield_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "shield_enabled", False))


def online_safe_bc_enabled(args: argparse.Namespace) -> bool:
    return max(int(getattr(args, "online_safe_bc_gradient_steps", 0)), 0) > 0


def _online_safe_bc_small_replay_priority_scale(
    args: argparse.Namespace,
    *,
    available_replay_rows: int | None = None,
) -> float:
    if available_replay_rows is None:
        return 1.0
    replay_rows = max(int(available_replay_rows), 0)
    if replay_rows <= 0:
        return 1.0
    batch_size = max(int(getattr(args, "online_safe_bc_batch_size", 256)), 1)
    min_rows_multiplier = max(float(getattr(args, "online_safe_bc_small_replay_min_rows_multiplier", 4.0)), 1.0)
    full_strength_multiplier = max(
        float(getattr(args, "online_safe_bc_small_replay_full_strength_rows_multiplier", 16.0)),
        min_rows_multiplier,
    )
    min_rows_for_reduced_scale = max(int(np.ceil(batch_size * min_rows_multiplier)), batch_size)
    min_rows_for_full_strength = max(int(np.ceil(batch_size * full_strength_multiplier)), min_rows_for_reduced_scale)
    reduced_scale = float(np.clip(getattr(args, "online_safe_bc_small_replay_priority_scale", 0.5), 0.0, 1.0))
    if replay_rows <= min_rows_for_reduced_scale:
        return reduced_scale
    if replay_rows >= min_rows_for_full_strength or min_rows_for_full_strength <= min_rows_for_reduced_scale:
        return 1.0
    ramp_fraction = (replay_rows - min_rows_for_reduced_scale) / max(
        float(min_rows_for_full_strength - min_rows_for_reduced_scale),
        1.0,
    )
    return float(reduced_scale + (1.0 - reduced_scale) * np.clip(ramp_fraction, 0.0, 1.0))


def online_safe_bc_priority_config(
    args: argparse.Namespace,
    *,
    available_replay_rows: int | None = None,
) -> dict[str, float]:
    config = {
        "intervention_priority_coef": max(float(getattr(args, "online_safe_bc_intervention_priority_coef", 4.0)), 0.0),
        "boundary_priority_coef": max(float(getattr(args, "online_safe_bc_boundary_priority_coef", 2.0)), 0.0),
        "terminal_priority_coef": max(float(getattr(args, "online_safe_bc_terminal_priority_coef", 2.0)), 0.0),
        "reserve_priority_coef": max(float(getattr(args, "online_safe_bc_reserve_priority_coef", 1.0)), 0.0),
        "teacher_priority_coef": max(float(getattr(args, "online_safe_bc_teacher_priority_coef", 2.0)), 0.0),
        "peak_value_priority_coef": max(float(getattr(args, "online_safe_bc_peak_value_priority_coef", 0.75)), 0.0),
        "valley_value_priority_coef": max(float(getattr(args, "online_safe_bc_valley_value_priority_coef", 0.5)), 0.0),
        "delta_priority_coef": max(float(getattr(args, "online_safe_bc_delta_priority_coef", 2.0)), 0.0),
        "terminal_deviation_priority_coef": max(
            float(getattr(args, "online_safe_bc_terminal_deviation_priority_coef", 1.0)),
            0.0,
        ),
    }
    priority_scale = _online_safe_bc_small_replay_priority_scale(args, available_replay_rows=available_replay_rows)
    if priority_scale >= 1.0:
        return config
    for key in (
        "intervention_priority_coef",
        "boundary_priority_coef",
        "teacher_priority_coef",
        "peak_value_priority_coef",
        "valley_value_priority_coef",
        "delta_priority_coef",
    ):
        config[key] *= priority_scale
    return config


def adaptive_online_safe_bc_gradient_steps(args: argparse.Namespace, validation_state: dict[str, float | int | str]) -> int:
    base_steps = max(int(getattr(args, "online_safe_bc_gradient_steps", 0)), 0)
    if base_steps <= 0:
        return 0
    effective_steps = base_steps
    scale_factor = max(float(getattr(args, "online_safe_bc_adaptive_scale_factor", 2.0)), 1.0)
    material_threshold = max(float(getattr(args, "online_safe_bc_adaptive_shield_material_threshold", 0.40)), 0.0)
    delta_threshold = max(float(getattr(args, "online_safe_bc_adaptive_shield_delta_threshold", 0.015)), 0.0)
    teacher_activation_threshold = float(
        np.clip(getattr(args, "online_safe_bc_adaptive_inventory_teacher_activation_threshold", 0.50), 0.0, 1.0)
    )
    teacher_gap_threshold = max(
        float(getattr(args, "online_safe_bc_adaptive_inventory_teacher_gap_threshold", 0.10)),
        0.0,
    )
    midband_threshold = float(np.clip(getattr(args, "online_safe_bc_adaptive_midband_dwell_threshold", 0.75), 0.0, 1.0))
    soc_target_mae_threshold = max(float(getattr(args, "online_safe_bc_adaptive_soc_target_mae_threshold", 0.08)), 0.0)
    peak_discharge_action_threshold = float(
        np.clip(getattr(args, "online_safe_bc_adaptive_peak_discharge_action_threshold", 0.20), 0.0, 1.0)
    )
    valley_charge_action_threshold = float(
        np.clip(getattr(args, "online_safe_bc_adaptive_valley_charge_action_threshold", 0.20), 0.0, 1.0)
    )
    patience = max(int(getattr(args, "online_safe_bc_adaptive_patience", 2)), 0)
    max_steps = max(int(getattr(args, "online_safe_bc_adaptive_max_gradient_steps", 128)), base_steps)
    batch_size = max(int(getattr(args, "online_safe_bc_batch_size", 256)), 1)
    replay_rows = max(int(validation_state.get("online_safe_bc_replay_rows", 0)), 0)
    min_rows_for_upscale = max(batch_size * 4, batch_size)
    last_material = float(validation_state.get("last_validation_mean_shield_material_activation_fraction", 0.0))
    last_delta = float(validation_state.get("last_validation_mean_abs_shield_delta", 0.0))
    last_midband = float(validation_state.get("last_validation_mean_soc_midband_dwell_fraction", 1.0))
    last_soc_target_mae = float(validation_state.get("last_validation_mean_soc_target_tracking_mae", 0.0))
    last_peak_discharge_action = float(validation_state.get("last_validation_mean_peak_price_discharge_action_fraction", 1.0))
    last_valley_charge_action = float(validation_state.get("last_validation_mean_valley_price_charge_action_fraction", 1.0))
    last_teacher_activation = float(validation_state.get("last_validation_mean_inventory_teacher_activation_fraction", 0.0))
    last_teacher_gap = float(validation_state.get("last_validation_mean_abs_inventory_teacher_gap", 0.0))
    stale_rounds = max(int(validation_state.get("stale_validation_rounds", 0)), 0)

    weak_inventory_learning = (
        last_midband < midband_threshold
        or last_soc_target_mae > soc_target_mae_threshold
        or last_peak_discharge_action < peak_discharge_action_threshold
        or last_valley_charge_action < valley_charge_action_threshold
    )
    protocol_overdependence_flags = int(last_material > material_threshold) + int(last_delta > delta_threshold)
    protocol_overdependence_flags += int(last_teacher_activation > teacher_activation_threshold)
    protocol_overdependence_flags += int(last_teacher_gap > teacher_gap_threshold)
    weak_protocol_internalization = protocol_overdependence_flags > 0
    allow_upscale = replay_rows <= 0 or replay_rows >= min_rows_for_upscale
    if weak_protocol_internalization:
        backoff_divisor = max(scale_factor, 1.0) ** (2 if protocol_overdependence_flags >= 2 else 1)
        effective_steps = max(1, int(np.floor(base_steps / backoff_divisor)))
    elif weak_inventory_learning:
        effective_steps = int(np.ceil(base_steps * scale_factor)) if allow_upscale else int(base_steps)
    if patience > 0 and stale_rounds >= patience:
        stale_backoff_divisor = max(scale_factor, 1.0)
        effective_steps = min(effective_steps, max(1, int(np.floor(base_steps / stale_backoff_divisor))))
    return min(max(effective_steps, 1), max_steps)


def resolve_tensorboard_log_dir(args: argparse.Namespace) -> str | None:
    raw_path = str(getattr(args, "tensorboard_log", "")).strip()
    if not raw_path:
        return None
    path = Path(raw_path)
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def default_tb_log_name(case_key: str, regime: str, train_model: str, seed: int, args: argparse.Namespace) -> str:
    explicit = str(getattr(args, "tb_log_name", "")).strip()
    if explicit:
        return explicit
    agent = str(getattr(args, "agent", "agent")).strip().lower()
    return f"{agent}_{case_key}_{regime}_{train_model}_seed{int(seed)}".replace("+", "_to_")


def learn_agent(agent, *, total_timesteps: int, progress_bar: bool, reset_num_timesteps: bool = True, tb_log_name: str | None = None):
    learn_kwargs: dict[str, Any] = {
        "total_timesteps": int(total_timesteps),
        "progress_bar": bool(progress_bar),
    }
    if not bool(reset_num_timesteps):
        learn_kwargs["reset_num_timesteps"] = False
    if tb_log_name:
        learn_kwargs["tb_log_name"] = str(tb_log_name)
    try:
        return agent.learn(**learn_kwargs)
    except TypeError:
        learn_kwargs.pop("tb_log_name", None)
        return agent.learn(**learn_kwargs)


def _causal_heuristic_action(unwrapped_env, policy_name: str) -> np.ndarray:
    return compute_rule_guidance_action(unwrapped_env, policy_name)


def _seed_replay_buffer_with_causal_heuristic(agent, args: argparse.Namespace) -> int:
    warmstart_steps = max(int(getattr(args, "causal_heuristic_warmstart_steps", 0)), 0)
    if warmstart_steps <= 0 or not hasattr(agent, "replay_buffer"):
        return 0
    vec_env = getattr(agent, "get_env", lambda: None)()
    replay_buffer = getattr(agent, "replay_buffer", None)
    if vec_env is None or replay_buffer is None or not hasattr(vec_env, "envs") or len(getattr(vec_env, "envs", [])) != 1:
        return 0

    obs = vec_env.reset()
    collected_steps = 0
    heuristic_policy = str(getattr(args, "causal_heuristic_warmstart_policy", "blended"))
    while collected_steps < warmstart_steps:
        base_env = vec_env.envs[0].unwrapped
        action = _causal_heuristic_action(base_env, heuristic_policy).reshape((1, -1))
        next_obs, rewards, dones, infos = vec_env.step(action)
        stored_action = np.asarray(action, dtype=np.float32).copy()
        replay_buffer.add(obs, next_obs, stored_action, rewards, dones, infos)
        obs = next_obs
        collected_steps += 1
        if bool(np.asarray(dones).reshape(-1)[0]):
            obs = vec_env.reset()
    for env in getattr(vec_env, "envs", []):
        current = env
        while current is not None:
            reset_progress = getattr(current, "reset_guidance_progress", None)
            if callable(reset_progress):
                reset_progress()
            current = getattr(current, "env", None)
    return int(collected_steps)


def _apply_offline_bc_warmstart(agent, *, case_key: str, regime: str, args: argparse.Namespace) -> dict[str, float | int | str]:
    dataset_path = str(getattr(args, "offline_dataset", "")).strip()
    if not dataset_path:
        return {
            "dataset_rows": 0,
            "replay_seeded_transitions": 0,
            "actor_gradient_steps": 0,
            "actor_batch_size": int(max(int(getattr(args, "bc_pretrain_batch_size", 256)), 1)),
            "initial_actor_mse": np.nan,
            "final_actor_mse": np.nan,
        }
    controller_sources = _parse_csv_arg(str(getattr(args, "offline_dataset_controller_sources", "")))
    report = apply_bc_warmstart(
        agent,
        dataset_path,
        replay_seed_limit=(None if int(getattr(args, "offline_dataset_max_transitions", 0)) <= 0 else int(getattr(args, "offline_dataset_max_transitions", 0))),
        actor_prefit_gradient_steps=max(int(getattr(args, "bc_pretrain_gradient_steps", 0)), 0),
        actor_prefit_batch_size=max(int(getattr(args, "bc_pretrain_batch_size", 256)), 1),
        actor_prefit_learning_rate=(
            None
            if float(getattr(args, "bc_pretrain_learning_rate", 0.0)) <= 0.0
            else float(getattr(args, "bc_pretrain_learning_rate", 0.0))
        ),
        controller_sources=controller_sources or None,
        cases=[case_key],
        regimes=[regime],
        shuffle_seed=int(getattr(args, "seed", 0)),
    )
    return {
        "dataset_rows": int(report.dataset_rows),
        "replay_seeded_transitions": int(report.replay_seeded_transitions),
        "actor_gradient_steps": int(report.actor_gradient_steps),
        "actor_batch_size": int(report.actor_batch_size),
        "initial_actor_mse": float(report.initial_actor_mse),
        "final_actor_mse": float(report.final_actor_mse),
    }


def build_env(
    case_key: str,
    battery_model: str,
    days: int,
    seed: int,
    regime: str,
    args: argparse.Namespace | None = None,
    window_metadata: dict[str, int | str | tuple[int, ...] | bool] | None = None,
    training: bool = False,
):
    reward_profile = str(getattr(args, "reward_profile", "network")) if args is not None else "network"
    disable_explicit_battery_degradation_penalties = bool(
        training and args is not None and getattr(args, "train_disable_explicit_battery_degradation_penalties", False)
    )
    config = build_config(
        case_key=case_key,
        battery_model=battery_model,
        days=days,
        seed=seed,
        regime=regime,
        reward_profile=reward_profile,
    )
    config = maybe_disable_explicit_battery_degradation_penalties(config, disable=disable_explicit_battery_degradation_penalties)
    config = maybe_patch_training_peak_reserve_reward(
        config,
        training=bool(training),
        weight_scale=float(getattr(args, "train_peak_reserve_weight_scale", 1.0)) if args is not None else 1.0,
        power_floor=float(getattr(args, "train_peak_reserve_power_floor", 0.0)) if args is not None else 0.0,
    )
    if window_metadata is not None:
        config = replace(
            config,
            simulation_days=int(window_metadata["days"]),
            episode_start_hour=int(window_metadata["start_hour"]),
            random_episode_start=bool(window_metadata["random_episode_start"]),
            full_year_random_start_hours=tuple(int(value) for value in window_metadata["full_year_random_start_hours"]),
            full_year_random_start_stride_hours=int(window_metadata["full_year_random_start_stride_hours"]),
        )
    env = NetworkMicrogridEnv(config)
    if args is not None and shield_enabled(args):
        env = ShieldedActionWrapper(env=env, **shield_config(args))
    if args is not None and action_regularization_enabled(args):
        env = ContinuousActionRegularizationWrapper(env=env, **action_regularization_config(args))
    if args is not None and rule_guidance_enabled(args):
        env = RuleGuidedActionWrapper(
            env=env,
            **rule_guidance_config(args),
            guidance_enabled=bool(training),
        )
    return env


def _dwell_fraction(mask: pd.Series | list[bool]) -> float:
    series = pd.Series(mask, dtype=float)
    if series.empty:
        return 0.0
    return float(series.mean())


def _peak_price_reserve_metrics(
    trajectory: pd.DataFrame,
    *,
    peak_price_threshold: float,
    discharge_limit_scale_w: float,
    low_discharge_limit_threshold: float,
) -> dict[str, float]:
    if trajectory.empty or "price" not in trajectory.columns or "battery_discharge_power_limit_w" not in trajectory.columns:
        return {
            "peak_price_step_fraction": 0.0,
            "peak_price_mean_soc": 0.0,
            "peak_price_mean_discharge_limit_ratio": 0.0,
            "peak_price_low_discharge_limit_dwell_fraction": 0.0,
        }
    scale_w = max(float(discharge_limit_scale_w), 1e-9)
    low_limit_threshold = float(np.clip(low_discharge_limit_threshold, 0.0, 1.0))
    peak_mask = pd.Series(trajectory["price"], dtype=float) >= float(peak_price_threshold) - 1e-9
    if not bool(peak_mask.any()):
        return {
            "peak_price_step_fraction": 0.0,
            "peak_price_mean_soc": 0.0,
            "peak_price_mean_discharge_limit_ratio": 0.0,
            "peak_price_low_discharge_limit_dwell_fraction": 0.0,
        }
    discharge_limit_ratio = (
        pd.Series(trajectory["battery_discharge_power_limit_w"], dtype=float).clip(lower=0.0) / scale_w
    ).clip(lower=0.0, upper=1.0)
    peak_soc = pd.Series(trajectory["soc"], dtype=float)[peak_mask]
    peak_limit_ratio = discharge_limit_ratio[peak_mask]
    return {
        "peak_price_step_fraction": float(peak_mask.mean()),
        "peak_price_mean_soc": float(peak_soc.mean()) if not peak_soc.empty else 0.0,
        "peak_price_mean_discharge_limit_ratio": float(peak_limit_ratio.mean()) if not peak_limit_ratio.empty else 0.0,
        "peak_price_low_discharge_limit_dwell_fraction": float((peak_limit_ratio < low_limit_threshold).mean())
        if not peak_limit_ratio.empty
        else 0.0,
    }


def _inventory_behavior_metrics(
    trajectory: pd.DataFrame,
    *,
    target_soc: float,
    target_tolerance: float,
    soc_min: float,
    soc_max: float,
    valley_price_threshold: float,
    peak_price_threshold: float,
    charge_limit_scale_w: float,
    discharge_limit_scale_w: float,
) -> dict[str, float]:
    if trajectory.empty or "soc" not in trajectory.columns or "price" not in trajectory.columns:
        return {
            "soc_midband_dwell_fraction": 0.0,
            "soc_target_tracking_mae": 0.0,
            "soc_upper_parking_fraction": 0.0,
            "soc_lower_parking_fraction": 0.0,
            "mean_charge_headroom_ratio": 0.0,
            "mean_discharge_headroom_ratio": 0.0,
            "valley_price_step_fraction": 0.0,
            "peak_price_discharge_action_fraction": 0.0,
            "valley_price_charge_action_fraction": 0.0,
            "valley_price_mean_charge_limit_ratio": 0.0,
        }

    usable_soc_span = max(float(soc_max) - float(soc_min), 1e-9)
    midband_half_width = min(max(0.18 * usable_soc_span, float(target_tolerance) + 0.04), usable_soc_span / 2.0)
    parking_buffer = min(max(0.10 * usable_soc_span, float(target_tolerance) + 0.03), usable_soc_span / 2.0)
    midband_low = float(np.clip(float(target_soc) - midband_half_width, float(soc_min), float(soc_max)))
    midband_high = float(np.clip(float(target_soc) + midband_half_width, float(soc_min), float(soc_max)))

    soc_series = pd.Series(trajectory["soc"], dtype=float)
    price_series = pd.Series(trajectory["price"], dtype=float)
    battery_power_values = trajectory["battery_power_mw"] if "battery_power_mw" in trajectory.columns else np.zeros(len(trajectory), dtype=float)
    charge_limit_values = (
        trajectory["battery_charge_power_limit_w"] if "battery_charge_power_limit_w" in trajectory.columns else np.zeros(len(trajectory), dtype=float)
    )
    discharge_limit_values = (
        trajectory["battery_discharge_power_limit_w"]
        if "battery_discharge_power_limit_w" in trajectory.columns
        else np.zeros(len(trajectory), dtype=float)
    )
    battery_power_series = pd.Series(battery_power_values, dtype=float)
    charge_limit_ratio = (
        pd.Series(charge_limit_values, dtype=float).clip(lower=0.0)
        / max(float(charge_limit_scale_w), 1e-9)
    ).clip(lower=0.0, upper=1.0)
    discharge_limit_ratio = (
        pd.Series(discharge_limit_values, dtype=float).clip(lower=0.0)
        / max(float(discharge_limit_scale_w), 1e-9)
    ).clip(lower=0.0, upper=1.0)

    valley_mask = price_series <= float(valley_price_threshold) + 1e-9
    peak_mask = price_series >= float(peak_price_threshold) - 1e-9
    valley_charge_fraction = float((battery_power_series[valley_mask] < -1e-6).mean()) if bool(valley_mask.any()) else 0.0
    peak_discharge_fraction = float((battery_power_series[peak_mask] > 1e-6).mean()) if bool(peak_mask.any()) else 0.0

    return {
        "soc_midband_dwell_fraction": float(((soc_series >= midband_low) & (soc_series <= midband_high)).mean()),
        "soc_target_tracking_mae": float((soc_series - float(target_soc)).abs().mean()),
        "soc_upper_parking_fraction": float((soc_series >= float(soc_max) - parking_buffer).mean()),
        "soc_lower_parking_fraction": float((soc_series <= float(soc_min) + parking_buffer).mean()),
        "mean_charge_headroom_ratio": float(charge_limit_ratio.mean()) if not charge_limit_ratio.empty else 0.0,
        "mean_discharge_headroom_ratio": float(discharge_limit_ratio.mean()) if not discharge_limit_ratio.empty else 0.0,
        "valley_price_step_fraction": float(valley_mask.mean()) if not valley_mask.empty else 0.0,
        "peak_price_discharge_action_fraction": peak_discharge_fraction,
        "valley_price_charge_action_fraction": valley_charge_fraction,
        "valley_price_mean_charge_limit_ratio": float(charge_limit_ratio[valley_mask].mean()) if bool(valley_mask.any()) else 0.0,
    }


def resolve_train_window(case_key: str, regime: str, args: argparse.Namespace) -> dict[str, int | str | tuple[int, ...] | bool] | None:
    train_year = int(getattr(args, "train_year", 0))
    if train_year <= 0:
        return None
    train_episode_days = int(getattr(args, "train_episode_days", 0)) or int(args.days)
    validation_days = int(getattr(args, "train_validation_days", 0))
    validation_offsets = _parse_int_csv_arg(str(getattr(args, "train_validation_offset_days_within_year", "")))
    return resolve_window_metadata(
        case_key=case_key,
        regime=regime,
        reward_profile=str(args.reward_profile),
        seed=int(args.seed),
        year=train_year,
        episode_days=train_episode_days,
        random_start_within_year=bool(getattr(args, "train_random_start_within_year", False)),
        stride_hours=int(getattr(args, "year_start_stride_hours", 24)),
        start_offset_days_within_year=0,
        exclude_tail_days=0 if validation_offsets else validation_days,
        excluded_window_offset_days_within_year=tuple(validation_offsets),
        excluded_window_days=validation_days,
    )


def resolve_validation_windows(case_key: str, regime: str, args: argparse.Namespace) -> list[dict[str, int | str | tuple[int, ...] | bool]]:
    train_year = int(getattr(args, "train_year", 0))
    validation_days = int(getattr(args, "train_validation_days", 0))
    if train_year <= 0 or validation_days <= 0:
        return []
    explicit_offsets = _parse_int_csv_arg(str(getattr(args, "train_validation_offset_days_within_year", "")))
    if explicit_offsets:
        windows: list[dict[str, int | str | tuple[int, ...] | bool]] = []
        for offset_days in explicit_offsets:
            windows.append(
                resolve_window_metadata(
                    case_key=case_key,
                    regime=regime,
                    reward_profile=str(args.reward_profile),
                    seed=int(args.seed),
                    year=train_year,
                    episode_days=int(validation_days),
                    random_start_within_year=False,
                    stride_hours=int(getattr(args, "year_start_stride_hours", 24)),
                    start_offset_days_within_year=int(offset_days),
                )
            )
        return windows
    full_window = resolve_year_window(
        case_key=case_key,
        year=train_year,
        regime=regime,
        reward_profile=str(args.reward_profile),
        seed=int(args.seed),
    )
    return [
        resolve_window_metadata(
            case_key=case_key,
            regime=regime,
            reward_profile=str(args.reward_profile),
            seed=int(args.seed),
            year=train_year,
            episode_days=int(validation_days),
            random_start_within_year=False,
            stride_hours=int(getattr(args, "year_start_stride_hours", 24)),
            start_offset_days_within_year=max(int(full_window["days"]) - int(validation_days), 0),
        )
    ]


def resolve_eval_window(case_key: str, regime: str, args: argparse.Namespace) -> dict[str, int | str | tuple[int, ...] | bool] | None:
    eval_year = int(getattr(args, "eval_year", 0))
    if eval_year <= 0:
        return None
    eval_days = int(getattr(args, "eval_days", 0))
    if eval_days <= 0:
        full_window = resolve_year_window(
            case_key=case_key,
            year=eval_year,
            regime=regime,
            reward_profile=str(args.reward_profile),
            seed=int(args.seed),
        )
        eval_days = int(full_window["days"])
    return resolve_window_metadata(
        case_key=case_key,
        regime=regime,
        reward_profile=str(args.reward_profile),
        seed=int(args.seed),
        year=eval_year,
        episode_days=int(eval_days),
        random_start_within_year=False,
        stride_hours=int(getattr(args, "year_start_stride_hours", 24)),
        start_offset_days_within_year=int(getattr(args, "eval_offset_days_within_year", 0)),
    )


def validation_selection_enabled(args: argparse.Namespace) -> bool:
    return int(getattr(args, "train_validation_days", 0)) > 0


def validation_checkpoint_interval(args: argparse.Namespace) -> int:
    raw_interval = int(getattr(args, "train_validation_checkpoint_every", 0))
    if raw_interval > 0:
        return raw_interval
    return max(int(getattr(args, "train_steps", 0)), 0)


def validation_metric_config(args: argparse.Namespace) -> dict[str, float | str]:
    return {
        "metric": str(getattr(args, "train_validation_metric", "health_objective")),
        "terminal_penalty_weight": float(getattr(args, "train_validation_terminal_penalty_weight", 1.0)),
        "boundary_dwell_weight": float(getattr(args, "train_validation_boundary_dwell_weight", 20000.0)),
        "infeasible_dwell_weight": float(getattr(args, "train_validation_infeasible_dwell_weight", 20000.0)),
        "peak_reserve_weight": float(getattr(args, "train_validation_peak_reserve_weight", 0.0)),
        "peak_discharge_limit_threshold": float(getattr(args, "train_validation_peak_discharge_limit_threshold", 0.25)),
        "gate_dwell_threshold": float(getattr(args, "train_validation_gate_dwell_threshold", 0.05)),
        "gate_violation_weight": float(getattr(args, "train_validation_gate_violation_weight", 1_000_000.0)),
        "gate_peak_reserve_dwell_threshold": float(getattr(args, "train_validation_gate_peak_reserve_dwell_threshold", -1.0)),
        "shield_mean_delta_weight": float(getattr(args, "train_validation_shield_mean_delta_weight", 0.0)),
        "shield_material_dwell_weight": float(getattr(args, "train_validation_shield_material_dwell_weight", 0.0)),
        "shield_strong_dwell_weight": float(getattr(args, "train_validation_shield_strong_dwell_weight", 0.0)),
        "final_soc_deviation_weight": float(getattr(args, "train_validation_final_soc_deviation_weight", 0.0)),
        "midband_dwell_weight": float(getattr(args, "train_validation_midband_dwell_weight", 10000.0)),
        "soc_target_tracking_weight": float(getattr(args, "train_validation_soc_target_tracking_weight", 5000.0)),
        "peak_discharge_headroom_weight": float(getattr(args, "train_validation_peak_discharge_headroom_weight", 10000.0)),
        "valley_charge_weight": float(getattr(args, "train_validation_valley_charge_weight", 5000.0)),
        "peak_discharge_weight": float(getattr(args, "train_validation_peak_discharge_weight", 5000.0)),
        "shield_mean_delta_threshold": float(getattr(args, "train_validation_shield_mean_delta_threshold", -1.0)),
        "shield_material_dwell_threshold": float(getattr(args, "train_validation_shield_material_dwell_threshold", -1.0)),
        "shield_strong_dwell_threshold": float(getattr(args, "train_validation_shield_strong_dwell_threshold", -1.0)),
    }


def _validation_metric_value(summary: dict[str, float | int | str], metric: str, config: dict[str, float | str] | None = None) -> float:
    metric_cfg = dict(config or {})
    metric_name = str(metric)
    if metric_name == "reward":
        return -float(summary["total_reward"])
    if metric_name in {
        "health_objective",
        "health_objective_gate",
        "health_objective_gate_shield",
        "inventory_value",
        "inventory_value_balanced",
        "inventory_value_gate",
        "inventory_value_gate_shield",
    }:
        terminal_weight = float(metric_cfg.get("terminal_penalty_weight", 1.0))
        boundary_weight = float(metric_cfg.get("boundary_dwell_weight", 20000.0))
        infeasible_weight = float(metric_cfg.get("infeasible_dwell_weight", 20000.0))
        peak_reserve_weight = float(metric_cfg.get("peak_reserve_weight", 0.0))
        shield_mean_delta_weight = float(metric_cfg.get("shield_mean_delta_weight", 0.0))
        shield_material_weight = float(metric_cfg.get("shield_material_dwell_weight", 0.0))
        shield_strong_weight = float(metric_cfg.get("shield_strong_dwell_weight", 0.0))
        final_soc_deviation_weight = float(metric_cfg.get("final_soc_deviation_weight", 0.0))
        midband_dwell_weight = float(metric_cfg.get("midband_dwell_weight", 10000.0))
        soc_target_tracking_weight = float(metric_cfg.get("soc_target_tracking_weight", 5000.0))
        peak_discharge_headroom_weight = float(metric_cfg.get("peak_discharge_headroom_weight", 10000.0))
        valley_charge_weight = float(metric_cfg.get("valley_charge_weight", 5000.0))
        peak_discharge_weight = float(metric_cfg.get("peak_discharge_weight", 5000.0))
        peak_discharge_limit_threshold = float(metric_cfg.get("peak_discharge_limit_threshold", 0.25))
        upper_dwell = float(summary.get("soc_upper_dwell_fraction", 0.0))
        lower_dwell = float(summary.get("soc_lower_dwell_fraction", 0.0))
        boundary_dwell = upper_dwell + lower_dwell
        infeasible_dwell = float(summary.get("infeasible_action_dwell_fraction", 0.0))
        mean_abs_shield_delta = float(summary.get("mean_abs_shield_delta", 0.0))
        shield_material_dwell = float(summary.get("shield_material_activation_fraction", 0.0))
        shield_strong_dwell = float(summary.get("shield_strong_activation_fraction", 0.0))
        final_soc_deviation = abs(float(summary.get("final_terminal_soc_deviation", 0.0)))
        objective_cost = summary.get("final_cumulative_objective_cost")
        if objective_cost is None:
            objective_cost = float(summary["final_cumulative_cost"]) + float(summary.get("total_terminal_soc_penalty", 0.0))
        metric_value = (
            float(objective_cost)
            + (terminal_weight - 1.0) * float(summary.get("total_terminal_soc_penalty", 0.0))
            + boundary_weight * float(boundary_dwell)
            + infeasible_weight * float(infeasible_dwell)
            + peak_reserve_weight * float(summary.get("peak_price_low_discharge_limit_dwell_fraction", 0.0))
            + shield_mean_delta_weight * mean_abs_shield_delta
            + shield_material_weight * shield_material_dwell
            + shield_strong_weight * shield_strong_dwell
            + final_soc_deviation_weight * final_soc_deviation
        )
        if metric_name in {
            "inventory_value",
            "inventory_value_balanced",
            "inventory_value_gate",
            "inventory_value_gate_shield",
        }:
            soc_midband_dwell = float(summary.get("soc_midband_dwell_fraction", 0.0))
            soc_target_tracking_mae = float(summary.get("soc_target_tracking_mae", 0.0))
            peak_headroom_ratio = float(summary.get("peak_price_mean_discharge_limit_ratio", 0.0))
            valley_charge_fraction = float(summary.get("valley_price_charge_action_fraction", 0.0))
            peak_discharge_fraction = float(summary.get("peak_price_discharge_action_fraction", 0.0))
            valley_charge_headroom_ratio = float(summary.get("valley_price_mean_charge_limit_ratio", 0.0))
            peak_headroom_shortfall = max(peak_discharge_limit_threshold - peak_headroom_ratio, 0.0)
            valley_charge_gap = 0.0
            if float(summary.get("valley_price_step_fraction", 0.0)) > 0.0:
                valley_charge_gap = max(valley_charge_headroom_ratio - valley_charge_fraction, 0.0)
            peak_discharge_gap = 0.0
            if float(summary.get("peak_price_step_fraction", 0.0)) > 0.0:
                peak_discharge_gap = max(peak_headroom_ratio - peak_discharge_fraction, 0.0)
            metric_value += (
                midband_dwell_weight * max(1.0 - soc_midband_dwell, 0.0)
                + soc_target_tracking_weight * soc_target_tracking_mae
                + peak_discharge_headroom_weight * peak_headroom_shortfall
                + valley_charge_weight * valley_charge_gap
                + peak_discharge_weight * peak_discharge_gap
            )
        if metric_name in {
            "health_objective_gate",
            "health_objective_gate_shield",
            "inventory_value_gate",
            "inventory_value_gate_shield",
        }:
            gate_threshold = max(float(metric_cfg.get("gate_dwell_threshold", 0.05)), 0.0)
            gate_violation_weight = max(float(metric_cfg.get("gate_violation_weight", 1_000_000.0)), 0.0)
            if gate_violation_weight > 0.0 and gate_threshold > 0.0:
                gated_values: list[tuple[float, float]] = [
                    (upper_dwell, gate_threshold),
                    (lower_dwell, gate_threshold),
                    (infeasible_dwell, gate_threshold),
                ]
                reserve_gate_threshold = float(metric_cfg.get("gate_peak_reserve_dwell_threshold", -1.0))
                if reserve_gate_threshold >= 0.0:
                    gated_values.append(
                        (
                            float(summary.get("peak_price_low_discharge_limit_dwell_fraction", 0.0)),
                            max(reserve_gate_threshold, 1e-9),
                        )
                    )
                if metric_name in {"health_objective_gate_shield", "inventory_value_gate_shield"}:
                    shield_material_threshold = float(metric_cfg.get("shield_material_dwell_threshold", -1.0))
                    if shield_material_threshold < 0.0:
                        shield_material_threshold = gate_threshold
                    gated_values.append((shield_material_dwell, max(shield_material_threshold, 1e-9)))

                    shield_strong_threshold = float(metric_cfg.get("shield_strong_dwell_threshold", -1.0))
                    if shield_strong_threshold < 0.0:
                        shield_strong_threshold = gate_threshold
                    gated_values.append((shield_strong_dwell, max(shield_strong_threshold, 1e-9)))

                    shield_mean_delta_threshold = float(metric_cfg.get("shield_mean_delta_threshold", -1.0))
                    if shield_mean_delta_threshold >= 0.0:
                        gated_values.append((mean_abs_shield_delta, max(shield_mean_delta_threshold, 1e-9)))

                gate_violations = sum(1 for value, threshold in gated_values if float(value) >= float(threshold))
                normalized_gate_excess = 0.0
                for value, threshold in gated_values:
                    normalized_gate_excess += max(float(value) - float(threshold), 0.0) / float(threshold)
                metric_value += gate_violation_weight * float(gate_violations + normalized_gate_excess)
        return float(metric_value)
    return float(summary["final_cumulative_objective_cost"])


def _training_segments(total_steps: int, checkpoint_interval: int) -> list[int]:
    remaining = max(int(total_steps), 0)
    if remaining <= 0:
        return []
    interval = max(int(checkpoint_interval), 0)
    if interval <= 0 or interval >= remaining:
        return [remaining]
    chunks: list[int] = []
    while remaining > 0:
        chunk = min(interval, remaining)
        chunks.append(int(chunk))
        remaining -= int(chunk)
    return chunks


def train_short_agent(case_key: str, train_model: str, regime: str, args: argparse.Namespace):
    schedule = resolve_training_schedule(train_model=train_model, args=args)
    stages = [str(stage) for stage in schedule["stages"]]
    stage_steps = [int(value) for value in schedule["stage_steps"]]
    stage_learning_rates = [float(value) for value in schedule["stage_learning_rates"]]
    base_learning_rate = float(getattr(args, "learning_rate", 3e-4))
    train_window = resolve_train_window(case_key=case_key, regime=regime, args=args)
    validation_windows = resolve_validation_windows(case_key=case_key, regime=regime, args=args)
    tensorboard_log_dir = resolve_tensorboard_log_dir(args)
    tensorboard_run_name = default_tb_log_name(
        case_key=case_key,
        regime=regime,
        train_model=train_model,
        seed=int(args.seed),
        args=args,
    )
    validation_metric_cfg = validation_metric_config(args)
    validation_metric = str(validation_metric_cfg["metric"])
    validation_interval = validation_checkpoint_interval(args) if validation_selection_enabled(args) else 0
    env = build_env(
        case_key=case_key,
        battery_model=stages[0],
        days=args.days,
        seed=args.seed,
        regime=regime,
        args=args,
        window_metadata=train_window,
        training=True,
    )
    try:
        output_dir = getattr(args, "output_dir", None)
        resume_checkpoint = None
        if output_dir:
            output_path = Path(output_dir)
            checkpoint_last_path = output_path / "checkpoint_last.zip"
            checkpoint_state_path = output_path / "checkpoint_state.json"
            if checkpoint_last_path.exists() and checkpoint_state_path.exists():
                print(f"[resume] Found existing checkpoint at {checkpoint_last_path}. Resuming training...")
                try:
                    import json
                    with open(checkpoint_state_path, "r", encoding="utf-8") as f:
                        resume_data = json.load(f)
                    resume_checkpoint = {
                        "path": checkpoint_last_path,
                        "data": resume_data
                    }
                except Exception as e:
                    print(f"[resume] Failed to load checkpoint state: {e}. Starting fresh.")

        if resume_checkpoint:
            agent = load_agent(args.agent, str(resume_checkpoint["path"]), env=env, device=args.device)
            _patch_offpolicy_agent_to_store_effective_actions(agent)
            if hasattr(agent, "load_replay_buffer"):
                replay_buffer_path = Path(output_dir) / "checkpoint_replay_buffer.pkl"
                if replay_buffer_path.exists():
                    try:
                        agent.load_replay_buffer(str(replay_buffer_path))
                        print(f"[resume] Loaded replay buffer from {replay_buffer_path}")
                    except Exception as e:
                        print(f"[resume] Failed to load replay buffer: {e}")
            warmstart_steps_applied = int(resume_checkpoint["data"]["validation_state"].get("warmstart_steps_applied", 0))
            offline_bc_report = {
                "dataset_rows": int(resume_checkpoint["data"]["validation_state"].get("offline_bc_dataset_rows", 0)),
                "replay_seeded_transitions": int(resume_checkpoint["data"]["validation_state"].get("offline_bc_replay_seeded_transitions", 0)),
                "actor_gradient_steps": int(resume_checkpoint["data"]["validation_state"].get("offline_bc_actor_gradient_steps", 0)),
                "actor_batch_size": int(resume_checkpoint["data"]["validation_state"].get("offline_bc_actor_batch_size", 0)),
                "initial_actor_mse": float(resume_checkpoint["data"]["validation_state"].get("offline_bc_initial_actor_mse", np.nan)),
                "final_actor_mse": float(resume_checkpoint["data"]["validation_state"].get("offline_bc_final_actor_mse", np.nan)),
            }
            validation_history = list(resume_checkpoint["data"]["validation_history"])
            validation_state = dict(resume_checkpoint["data"]["validation_state"])
            current_total_steps = int(resume_checkpoint["data"]["current_total_steps"])
            print(f"[resume] Resuming from step {current_total_steps}")
        else:
            agent = create_agent(
                agent_name=args.agent,
                env=env,
                total_steps=int(args.train_steps),
                seed=int(args.seed),
                device=str(args.device),
                agent_hyperparams={
                    "learning_starts": min(128, max(16, int(args.train_steps // 10))),
                    "off_policy_batch_size": 64,
                    "learning_rate": base_learning_rate,
                    "gamma": 0.99,
                    "tau": 0.005,
                    "net_arch": (128, 128),
                },
                tensorboard_log=tensorboard_log_dir,
            )
            warmstart_steps_applied = _seed_replay_buffer_with_causal_heuristic(agent, args)
            offline_bc_report = _apply_offline_bc_warmstart(agent, case_key=case_key, regime=regime, args=args)
            validation_history = []
            validation_state = {
                "best_metric_value": np.nan,
                "best_total_reward": np.nan,
                "best_objective_cost": np.nan,
                "best_checkpoint_step": int(sum(stage_steps)),
                "metric": validation_metric,
                "checkpoint_interval": int(validation_interval),
                "terminal_penalty_weight": float(validation_metric_cfg["terminal_penalty_weight"]),
                "boundary_dwell_weight": float(validation_metric_cfg["boundary_dwell_weight"]),
                "infeasible_dwell_weight": float(validation_metric_cfg["infeasible_dwell_weight"]),
                "peak_reserve_weight": float(validation_metric_cfg["peak_reserve_weight"]),
                "peak_discharge_limit_threshold": float(validation_metric_cfg["peak_discharge_limit_threshold"]),
                "warmstart_steps_applied": int(warmstart_steps_applied),
                "warmstart_policy": str(getattr(args, "causal_heuristic_warmstart_policy", "blended")),
                "offline_bc_dataset_rows": int(offline_bc_report["dataset_rows"]),
                "offline_bc_replay_seeded_transitions": int(offline_bc_report["replay_seeded_transitions"]),
                "offline_bc_actor_gradient_steps": int(offline_bc_report["actor_gradient_steps"]),
                "offline_bc_actor_batch_size": int(offline_bc_report["actor_batch_size"]),
                "offline_bc_initial_actor_mse": float(offline_bc_report["initial_actor_mse"]),
                "offline_bc_final_actor_mse": float(offline_bc_report["final_actor_mse"]),
                "online_safe_bc_replay_rows": 0,
                "online_safe_bc_actor_gradient_steps": 0,
                "online_safe_bc_actor_batch_size": 0,
                "online_safe_bc_initial_actor_mse": np.nan,
                "online_safe_bc_final_actor_mse": np.nan,
                "online_safe_bc_intervention_rows": 0,
                "online_safe_bc_inventory_teacher_rows": 0,
                "online_safe_bc_mean_sample_weight": 1.0,
                "effective_online_safe_bc_gradient_steps": int(max(int(getattr(args, "online_safe_bc_gradient_steps", 0)), 0)),
                "effective_online_safe_bc_priority_scale": 1.0,
                "last_validation_mean_shield_material_activation_fraction": np.nan,
                "last_validation_mean_abs_shield_delta": np.nan,
                "last_validation_mean_soc_midband_dwell_fraction": np.nan,
                "last_validation_mean_soc_target_tracking_mae": np.nan,
                "last_validation_mean_peak_price_discharge_action_fraction": np.nan,
                "last_validation_mean_valley_price_charge_action_fraction": np.nan,
                "last_validation_mean_inventory_teacher_activation_fraction": np.nan,
                "last_validation_mean_abs_inventory_teacher_gap": np.nan,
                "stale_validation_rounds": 0,
            }
            current_total_steps = 0
        first_learn_call = True

        def _maybe_update_validation(current_agent, total_steps_done: int) -> None:
            nonlocal validation_state, best_parameters
            if not validation_windows:
                return
            validation_metric_values: list[float] = []
            validation_rewards: list[float] = []
            validation_objective_costs: list[float] = []
            validation_terminal_penalties: list[float] = []
            validation_upper_dwells: list[float] = []
            validation_lower_dwells: list[float] = []
            validation_boundary_dwells: list[float] = []
            validation_infeasible_dwells: list[float] = []
            validation_peak_limit_ratios: list[float] = []
            validation_peak_low_limit_dwells: list[float] = []
            validation_shield_mean_deltas: list[float] = []
            validation_shield_material_dwells: list[float] = []
            validation_shield_strong_dwells: list[float] = []
            validation_final_soc_deviations: list[float] = []
            validation_midband_dwells: list[float] = []
            validation_soc_target_tracking: list[float] = []
            validation_peak_discharge_actions: list[float] = []
            validation_valley_charge_actions: list[float] = []
            validation_teacher_activation_dwells: list[float] = []
            validation_teacher_mean_gaps: list[float] = []
            for validation_window in validation_windows:
                validation_summary, _, _ = evaluate_agent(
                    current_agent,
                    case_key=case_key,
                    test_model=stages[-1],
                    regime=regime,
                    args=args,
                    eval_window_override=validation_window,
                    eval_steps_override=0,
                    eval_full_horizon_override=True,
                )
                validation_metric_values.append(_validation_metric_value(validation_summary, validation_metric, validation_metric_cfg))
                validation_rewards.append(float(validation_summary["total_reward"]))
                validation_objective_costs.append(float(validation_summary["final_cumulative_objective_cost"]))
                validation_terminal_penalties.append(float(validation_summary.get("total_terminal_soc_penalty", 0.0)))
                validation_upper_dwells.append(float(validation_summary.get("soc_upper_dwell_fraction", 0.0)))
                validation_lower_dwells.append(float(validation_summary.get("soc_lower_dwell_fraction", 0.0)))
                validation_boundary_dwells.append(
                    float(validation_summary.get("soc_upper_dwell_fraction", 0.0))
                    + float(validation_summary.get("soc_lower_dwell_fraction", 0.0))
                )
                validation_infeasible_dwells.append(float(validation_summary.get("infeasible_action_dwell_fraction", 0.0)))
                validation_peak_limit_ratios.append(float(validation_summary.get("peak_price_mean_discharge_limit_ratio", 0.0)))
                validation_peak_low_limit_dwells.append(float(validation_summary.get("peak_price_low_discharge_limit_dwell_fraction", 0.0)))
                validation_shield_mean_deltas.append(float(validation_summary.get("mean_abs_shield_delta", 0.0)))
                validation_shield_material_dwells.append(float(validation_summary.get("shield_material_activation_fraction", 0.0)))
                validation_shield_strong_dwells.append(float(validation_summary.get("shield_strong_activation_fraction", 0.0)))
                validation_final_soc_deviations.append(abs(float(validation_summary.get("final_terminal_soc_deviation", 0.0))))
                validation_midband_dwells.append(float(validation_summary.get("soc_midband_dwell_fraction", 0.0)))
                validation_soc_target_tracking.append(float(validation_summary.get("soc_target_tracking_mae", 0.0)))
                validation_peak_discharge_actions.append(float(validation_summary.get("peak_price_discharge_action_fraction", 0.0)))
                validation_valley_charge_actions.append(float(validation_summary.get("valley_price_charge_action_fraction", 0.0)))
                validation_teacher_activation_dwells.append(
                    float(validation_summary.get("inventory_teacher_activation_fraction", 0.0))
                )
                validation_teacher_mean_gaps.append(float(validation_summary.get("mean_abs_inventory_teacher_gap", 0.0)))
            metric_value = float(np.mean(validation_metric_values)) if validation_metric_values else np.nan
            validation_history.append(
                {
                    "checkpoint_step": int(total_steps_done),
                    "metric_value": float(metric_value),
                    "mean_total_reward": float(np.mean(validation_rewards)) if validation_rewards else np.nan,
                    "mean_objective_cost": float(np.mean(validation_objective_costs)) if validation_objective_costs else np.nan,
                    "mean_terminal_soc_penalty": float(np.mean(validation_terminal_penalties)) if validation_terminal_penalties else np.nan,
                    "mean_soc_upper_dwell_fraction": float(np.mean(validation_upper_dwells)) if validation_upper_dwells else np.nan,
                    "mean_soc_lower_dwell_fraction": float(np.mean(validation_lower_dwells)) if validation_lower_dwells else np.nan,
                    "mean_boundary_dwell_fraction": float(np.mean(validation_boundary_dwells)) if validation_boundary_dwells else np.nan,
                    "mean_infeasible_action_dwell_fraction": float(np.mean(validation_infeasible_dwells))
                    if validation_infeasible_dwells
                    else np.nan,
                    "mean_peak_price_discharge_limit_ratio": float(np.mean(validation_peak_limit_ratios))
                    if validation_peak_limit_ratios
                    else np.nan,
                    "mean_peak_price_low_discharge_limit_dwell_fraction": float(np.mean(validation_peak_low_limit_dwells))
                    if validation_peak_low_limit_dwells
                    else np.nan,
                    "mean_abs_shield_delta": float(np.mean(validation_shield_mean_deltas)) if validation_shield_mean_deltas else np.nan,
                    "mean_shield_material_activation_fraction": float(np.mean(validation_shield_material_dwells))
                    if validation_shield_material_dwells
                    else np.nan,
                    "mean_shield_strong_activation_fraction": float(np.mean(validation_shield_strong_dwells))
                    if validation_shield_strong_dwells
                    else np.nan,
                    "mean_final_terminal_soc_deviation": float(np.mean(validation_final_soc_deviations))
                    if validation_final_soc_deviations
                    else np.nan,
                    "mean_soc_midband_dwell_fraction": float(np.mean(validation_midband_dwells))
                    if validation_midband_dwells
                    else np.nan,
                    "mean_soc_target_tracking_mae": float(np.mean(validation_soc_target_tracking))
                    if validation_soc_target_tracking
                    else np.nan,
                    "mean_peak_price_discharge_action_fraction": float(np.mean(validation_peak_discharge_actions))
                    if validation_peak_discharge_actions
                    else np.nan,
                    "mean_valley_price_charge_action_fraction": float(np.mean(validation_valley_charge_actions))
                    if validation_valley_charge_actions
                    else np.nan,
                }
            )
            best_value = validation_state["best_metric_value"]
            mean_shield_material = float(np.mean(validation_shield_material_dwells)) if validation_shield_material_dwells else np.nan
            mean_shield_delta = float(np.mean(validation_shield_mean_deltas)) if validation_shield_mean_deltas else np.nan
            mean_midband_dwell = float(np.mean(validation_midband_dwells)) if validation_midband_dwells else np.nan
            mean_soc_target_tracking = float(np.mean(validation_soc_target_tracking)) if validation_soc_target_tracking else np.nan
            mean_peak_discharge_action = (
                float(np.mean(validation_peak_discharge_actions)) if validation_peak_discharge_actions else np.nan
            )
            mean_valley_charge_action = (
                float(np.mean(validation_valley_charge_actions)) if validation_valley_charge_actions else np.nan
            )
            mean_teacher_activation = (
                float(np.mean(validation_teacher_activation_dwells)) if validation_teacher_activation_dwells else np.nan
            )
            mean_teacher_gap = float(np.mean(validation_teacher_mean_gaps)) if validation_teacher_mean_gaps else np.nan
            improved = bool(np.isnan(best_value)) or float(metric_value) < float(best_value)
            if improved:
                best_parameters = copy.deepcopy(current_agent.get_parameters())
                if output_dir:
                    try:
                        output_path = Path(output_dir)
                        output_path.mkdir(parents=True, exist_ok=True)
                        current_agent.save(str(output_path / "checkpoint_best.zip"))
                        print(f"  [checkpoint] Saved new best validation model to {output_path / 'checkpoint_best.zip'}")
                    except Exception as e:
                        print(f"  [checkpoint] Failed to save best model: {e}")
                validation_state = {
                    **validation_state,
                    "best_metric_value": float(metric_value),
                    "best_total_reward": float(np.mean(validation_rewards)) if validation_rewards else np.nan,
                    "best_objective_cost": float(np.mean(validation_objective_costs)) if validation_objective_costs else np.nan,
                    "best_checkpoint_step": int(total_steps_done),
                    "last_validation_mean_shield_material_activation_fraction": float(mean_shield_material),
                    "last_validation_mean_abs_shield_delta": float(mean_shield_delta),
                    "last_validation_mean_soc_midband_dwell_fraction": float(mean_midband_dwell),
                    "last_validation_mean_soc_target_tracking_mae": float(mean_soc_target_tracking),
                    "last_validation_mean_peak_price_discharge_action_fraction": float(mean_peak_discharge_action),
                    "last_validation_mean_valley_price_charge_action_fraction": float(mean_valley_charge_action),
                    "last_validation_mean_inventory_teacher_activation_fraction": float(mean_teacher_activation),
                    "last_validation_mean_abs_inventory_teacher_gap": float(mean_teacher_gap),
                    "stale_validation_rounds": 0,
                }
            else:
                validation_state = {
                    **validation_state,
                    "last_validation_mean_shield_material_activation_fraction": float(mean_shield_material),
                    "last_validation_mean_abs_shield_delta": float(mean_shield_delta),
                    "last_validation_mean_soc_midband_dwell_fraction": float(mean_midband_dwell),
                    "last_validation_mean_soc_target_tracking_mae": float(mean_soc_target_tracking),
                    "last_validation_mean_peak_price_discharge_action_fraction": float(mean_peak_discharge_action),
                    "last_validation_mean_valley_price_charge_action_fraction": float(mean_valley_charge_action),
                    "last_validation_mean_inventory_teacher_activation_fraction": float(mean_teacher_activation),
                    "last_validation_mean_abs_inventory_teacher_gap": float(mean_teacher_gap),
                    "stale_validation_rounds": int(validation_state.get("stale_validation_rounds", 0)) + 1,
                }

        for stage_index, (stage_model, steps, stage_learning_rate) in enumerate(
            zip(stages, stage_steps, stage_learning_rates),
            start=1,
        ):
            if stage_index == 1:
                current_env = env
            else:
                current_env = build_env(
                    case_key=case_key,
                    battery_model=stage_model,
                    days=args.days,
                    seed=args.seed,
                    regime=regime,
                    args=args,
                    window_metadata=train_window,
                    training=True,
                )
                agent.set_env(current_env)
            try:
                if steps <= 0:
                    continue
                _set_agent_learning_rate(agent, stage_learning_rate)
                for chunk_steps in _training_segments(steps, validation_interval):
                    chunk_end_step = current_total_steps + int(chunk_steps)
                    if resume_checkpoint and chunk_end_step <= int(resume_checkpoint["data"]["current_total_steps"]):
                        print(f"  [resume] Skipping already completed chunk: {current_total_steps} -> {chunk_end_step}")
                        current_total_steps = chunk_end_step
                        first_learn_call = False
                        continue

                    print(
                        f"  [train-progress] stage {stage_index}/{len(stages)} ({stage_model}) "
                        f"chunk_steps={chunk_steps} current_total={current_total_steps}"
                    )
                    learn_agent(
                        agent,
                        total_timesteps=int(chunk_steps),
                        progress_bar=False,
                        reset_num_timesteps=bool(first_learn_call),
                        tb_log_name=f"{tensorboard_run_name}_stage{stage_index}of{len(stages)}" if tensorboard_log_dir else None,
                    )
                    first_learn_call = False
                    current_total_steps += int(chunk_steps)
                    if online_safe_bc_enabled(args):
                        effective_online_safe_bc_gradient_steps = adaptive_online_safe_bc_gradient_steps(args, validation_state)
                        online_safe_bc_replay_rows_estimate = int(current_total_steps)
                        if int(getattr(args, "online_safe_bc_max_samples", 0)) > 0:
                            online_safe_bc_replay_rows_estimate = min(
                                online_safe_bc_replay_rows_estimate,
                                int(getattr(args, "online_safe_bc_max_samples", 0)),
                            )
                        resolved_online_safe_bc_priority = online_safe_bc_priority_config(
                            args,
                            available_replay_rows=online_safe_bc_replay_rows_estimate,
                        )
                        resolved_online_safe_bc_priority_scale = _online_safe_bc_small_replay_priority_scale(
                            args,
                            available_replay_rows=online_safe_bc_replay_rows_estimate,
                        )
                        replay_safe_bc_report = distill_sac_actor_from_replay_buffer(
                            agent,
                            gradient_steps=int(effective_online_safe_bc_gradient_steps),
                            batch_size=max(int(getattr(args, "online_safe_bc_batch_size", 256)), 1),
                            learning_rate=(
                                None
                                if float(getattr(args, "online_safe_bc_learning_rate", 0.0)) <= 0.0
                                else float(getattr(args, "online_safe_bc_learning_rate", 0.0))
                            ),
                            max_samples=(
                                None
                                if int(getattr(args, "online_safe_bc_max_samples", 0)) <= 0
                                else int(getattr(args, "online_safe_bc_max_samples", 0))
                            ),
                            shuffle_seed=int(getattr(args, "seed", 0)) + int(current_total_steps),
                            **resolved_online_safe_bc_priority,
                        )
                        validation_state = {
                            **validation_state,
                            "online_safe_bc_replay_rows": int(replay_safe_bc_report.replay_rows),
                            "online_safe_bc_actor_gradient_steps": int(replay_safe_bc_report.actor_gradient_steps),
                            "online_safe_bc_actor_batch_size": int(replay_safe_bc_report.actor_batch_size),
                            "online_safe_bc_initial_actor_mse": float(replay_safe_bc_report.initial_actor_mse),
                            "online_safe_bc_final_actor_mse": float(replay_safe_bc_report.final_actor_mse),
                            "online_safe_bc_intervention_rows": int(replay_safe_bc_report.intervention_rows),
                            "online_safe_bc_inventory_teacher_rows": int(replay_safe_bc_report.inventory_teacher_rows),
                            "online_safe_bc_mean_sample_weight": float(replay_safe_bc_report.mean_sample_weight),
                            "effective_online_safe_bc_gradient_steps": int(effective_online_safe_bc_gradient_steps),
                            "effective_online_safe_bc_priority_scale": float(resolved_online_safe_bc_priority_scale),
                        }
                    if validation_windows:
                        _maybe_update_validation(agent, current_total_steps)

                    if output_dir:
                        try:
                            output_path = Path(output_dir)
                            output_path.mkdir(parents=True, exist_ok=True)
                            agent.save(str(output_path / "checkpoint_last.zip"))
                            if hasattr(agent, "save_replay_buffer"):
                                agent.save_replay_buffer(str(output_path / "checkpoint_replay_buffer.pkl"))
                            import json
                            with open(output_path / "checkpoint_state.json", "w", encoding="utf-8") as f:
                                json.dump({
                                    "current_total_steps": int(current_total_steps),
                                    "validation_state": validation_state,
                                    "validation_history": validation_history
                                }, f, indent=2)
                            print(f"  [checkpoint] Saved intermediate checkpoint to {output_path / 'checkpoint_last.zip'}")
                        except Exception as e:
                            print(f"  [checkpoint] Failed to save intermediate checkpoint: {e}")
            finally:
                if current_env is not env:
                    current_env.close()

        if validation_windows:
            best_model_path = Path(output_dir) / "checkpoint_best.zip" if output_dir else None
            if best_model_path and best_model_path.exists():
                print(f"[resume] Restoring best validation parameters from {best_model_path}")
                agent = load_agent(args.agent, str(best_model_path), env=env, device=args.device)
            elif best_parameters is not None:
                agent.set_parameters(best_parameters, exact_match=True)
        return agent, schedule, train_window, {
            "tensorboard_log_dir": tensorboard_log_dir or "",
            "tensorboard_run_name": tensorboard_run_name,
            "validation_windows": validation_windows,
            "validation": validation_state,
            "validation_history": validation_history,
        }
    finally:
        env.close()


def evaluate_agent(
    agent,
    case_key: str,
    test_model: str,
    regime: str,
    args: argparse.Namespace,
    eval_window_override: dict[str, int | str | tuple[int, ...] | bool] | None = None,
    eval_steps_override: int | None = None,
    eval_full_horizon_override: bool | None = None,
) -> tuple[dict, pd.DataFrame]:
    eval_window = eval_window_override if eval_window_override is not None else resolve_eval_window(case_key=case_key, regime=regime, args=args)
    env = build_env(
        case_key=case_key,
        battery_model=test_model,
        days=args.days,
        seed=args.seed,
        regime=regime,
        args=args,
        window_metadata=eval_window,
        training=False,
    )
    rows: list[dict[str, float | int | str]] = []
    try:
        soc_min = float(env.unwrapped.config.battery_params.soc_min)
        soc_max = float(env.unwrapped.config.battery_params.soc_max)
        battery_command_scale_w = max(
            float(env.unwrapped.config.battery_params.p_charge_max),
            float(env.unwrapped.config.battery_params.p_discharge_max),
            1.0,
        )
        soc_tol = 1e-4
        infeasible_gap_tol = 1e-6
        internal_clip_gap_tol_w = max(1.0, 1e-4 * battery_command_scale_w)
        obs, info = env.reset()
        total_reward = 0.0
        eval_full_horizon = bool(getattr(args, "eval_full_horizon", False)) if eval_full_horizon_override is None else bool(eval_full_horizon_override)
        requested_eval_steps = int(args.eval_steps) if eval_steps_override is None else int(eval_steps_override)
        max_eval_steps = int(env.unwrapped.total_steps) if eval_full_horizon else int(requested_eval_steps)
        if max_eval_steps <= 0:
            max_eval_steps = int(env.unwrapped.total_steps)
        for step in range(max_eval_steps):
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            reward_after_terminal_penalty = float(info.get("reward_after_terminal_penalty", reward))
            rows.append(
                {
                    "step": int(step),
                    "timestamp": str(info.get("timestamp", "")),
                    "load_w": float(info.get("load_w", 0.0)),
                    "pv_w": float(info.get("pv_w", 0.0)),
                    "price": float(info.get("price", 0.0)),
                    "reward": float(reward),
                    "reward_wrapper_adjustment": float(float(reward) - reward_after_terminal_penalty),
                    "step_reward_before_clip": float(info.get("step_reward_before_clip", reward_after_terminal_penalty)),
                    "step_reward_after_clip": float(info.get("step_reward_after_clip", reward_after_terminal_penalty)),
                    "battery_shaping_penalty": float(info.get("battery_shaping_penalty", 0.0)),
                    "reward_after_battery_shaping": float(info.get("reward_after_battery_shaping", reward_after_terminal_penalty)),
                    "reward_after_peak_reserve_penalty": float(info.get("reward_after_peak_reserve_penalty", reward_after_terminal_penalty)),
                    "reward_after_terminal_penalty": reward_after_terminal_penalty,
                    "import_cost": float(info.get("import_cost", 0.0)),
                    "export_revenue": float(info.get("export_revenue", 0.0)),
                    "net_energy_cost": float(info.get("net_energy_cost", 0.0)),
                    "grid_limit_penalty_cost": float(info.get("grid_limit_penalty_cost", 0.0)),
                    "total_grid_cost": float(info.get("total_grid_cost", 0.0)),
                    "grid_import_limit_violation_mw": float(info.get("grid_import_limit_violation_mw", 0.0)),
                    "grid_export_limit_violation_mw": float(info.get("grid_export_limit_violation_mw", 0.0)),
                    "soc": float(info.get("soc", 0.0)),
                    "soc_violation": float(info.get("soc_violation", 0.0)),
                    "soc_center_penalty": float(info.get("soc_center_penalty", 0.0)),
                    "soc_edge_penalty": float(info.get("soc_edge_penalty", 0.0)),
                    "boundary_dwell_penalty": float(info.get("boundary_dwell_penalty", 0.0)),
                    "boundary_dwell_proximity": float(info.get("boundary_dwell_proximity", 0.0)),
                    "boundary_dwell_lower_proximity": float(info.get("boundary_dwell_lower_proximity", 0.0)),
                    "boundary_dwell_upper_proximity": float(info.get("boundary_dwell_upper_proximity", 0.0)),
                    "peak_reserve_shortfall": float(info.get("peak_reserve_shortfall", 0.0)),
                    "peak_reserve_penalty": float(info.get("peak_reserve_penalty", 0.0)),
                    "discharge_limit_ratio": float(info.get("discharge_limit_ratio", 0.0)),
                    "grid_import_mw": float(info.get("grid_import_mw", 0.0)),
                    "grid_export_mw": float(info.get("grid_export_mw", 0.0)),
                    "battery_power_mw": float(info.get("battery_power_mw", 0.0)),
                    "battery_loss_kwh": float(info.get("battery_loss_kwh", 0.0)),
                    "battery_stress_kwh": float(info.get("battery_stress_kwh", 0.0)),
                    "battery_throughput_kwh": float(info.get("battery_throughput_kwh", 0.0)),
                    "battery_action_raw": float(info.get("battery_action_raw", 0.0)),
                    "battery_action_applied": float(info.get("battery_action_applied", 0.0)),
                    "battery_action_applied_pre_shield": float(
                        info.get("battery_action_applied_pre_shield", info.get("battery_action_applied", 0.0))
                    ),
                    "battery_action_delta": float(info.get("battery_action_delta", 0.0)),
                    "action_rate_penalty": float(info.get("action_rate_penalty", 0.0)),
                    "policy_action_pre_guidance": float(info.get("policy_action_pre_guidance", 0.0)),
                    "rule_based_action_hint": float(info.get("rule_based_action_hint", 0.0)),
                    "rule_guided_action": float(info.get("rule_guided_action", 0.0)),
                    "rule_guidance_mix": float(info.get("rule_guidance_mix", 0.0)),
                    "action_after_rule_guidance": float(info.get("action_after_rule_guidance", info.get("battery_action_applied", 0.0))),
                    "battery_action_feasible_low": float(info.get("battery_action_feasible_low", -1.0)),
                    "battery_action_feasible_high": float(info.get("battery_action_feasible_high", 1.0)),
                    "battery_charge_fraction_feasible": float(info.get("battery_charge_fraction_feasible", 1.0)),
                    "battery_discharge_fraction_feasible": float(info.get("battery_discharge_fraction_feasible", 1.0)),
                    "battery_charge_power_limit_w": float(
                        info.get(
                            "battery_charge_power_limit_w",
                            float(info.get("battery_charge_fraction_feasible", 1.0))
                            * float(env.unwrapped.config.battery_params.p_charge_max),
                        )
                    ),
                    "battery_discharge_power_limit_w": float(
                        info.get(
                            "battery_discharge_power_limit_w",
                            float(info.get("battery_discharge_fraction_feasible", 1.0))
                            * float(env.unwrapped.config.battery_params.p_discharge_max),
                        )
                    ),
                    "battery_action_infeasible_gap": float(info.get("battery_action_infeasible_gap", 0.0)),
                    "battery_action_infeasible_penalty": float(info.get("battery_action_infeasible_penalty", 0.0)),
                    "shield_enabled": int(bool(info.get("shield_enabled", False))),
                    "shield_pre_action": float(info.get("shield_pre_action", info.get("battery_action_applied", 0.0))),
                    "shield_post_action": float(info.get("shield_post_action", info.get("battery_action_applied", 0.0))),
                    "shield_delta": float(info.get("shield_delta", 0.0)),
                    "shield_applied": int(bool(info.get("shield_applied", False))),
                    "shield_feasibility_clipped": int(bool(info.get("shield_feasibility_clipped", False))),
                    "shield_reserve_active": int(bool(info.get("shield_reserve_active", False))),
                    "shield_boundary_active": int(bool(info.get("shield_boundary_active", False))),
                    "shield_terminal_active": int(bool(info.get("shield_terminal_active", False))),
                    "shield_effective_low": float(info.get("shield_effective_low", -1.0)),
                    "shield_effective_high": float(info.get("shield_effective_high", 1.0)),
                    "shield_closure_action": float(info.get("shield_closure_action", 0.0)),
                    "shield_closure_mix": float(info.get("shield_closure_mix", 0.0)),
                    "shield_delta_penalty": float(info.get("shield_delta_penalty", 0.0)),
                    "shield_delta_penalty_coef_current": float(info.get("shield_delta_penalty_coef_current", 0.0)),
                    "shield_delta_penalty_progress": float(info.get("shield_delta_penalty_progress", 0.0)),
                    "inventory_teacher_action": float(info.get("inventory_teacher_action", 0.0)),
                    "inventory_teacher_active": int(bool(info.get("inventory_teacher_active", False))),
                    "inventory_teacher_boundary_active": int(bool(info.get("inventory_teacher_boundary_active", False))),
                    "inventory_teacher_terminal_active": int(bool(info.get("inventory_teacher_terminal_active", False))),
                    "inventory_teacher_reserve_active": int(bool(info.get("inventory_teacher_reserve_active", False))),
                    "inventory_teacher_weight": float(info.get("inventory_teacher_weight", 0.0)),
                    "battery_command_requested_w": float(info.get("requested_command", 0.0)),
                    "battery_command_applied_w": float(info.get("applied_command", info.get("requested_command", 0.0))),
                    "battery_internal_clip_gap_w": float(info.get("internal_clip_gap_w", 0.0)),
                    "p_max_w": float(info.get("p_max", 0.0)),
                    "p_max_trend_w": float(info.get("p_max_trend", 0.0)),
                    "soc_upper_bound_hit": int(float(info.get("soc", 0.0)) >= soc_max - soc_tol),
                    "soc_lower_bound_hit": int(float(info.get("soc", 0.0)) <= soc_min + soc_tol),
                    "battery_action_infeasible_flag": int(float(info.get("battery_action_infeasible_gap", 0.0)) > infeasible_gap_tol),
                    "battery_internal_clip_flag": int(float(info.get("internal_clip_gap_w", 0.0)) > internal_clip_gap_tol_w),
                    "min_bus_voltage_pu": float(info.get("min_bus_voltage_pu", 1.0)),
                    "max_bus_voltage_pu": float(info.get("max_bus_voltage_pu", 1.0)),
                    "undervoltage": float(info.get("undervoltage", 0.0)),
                    "overvoltage": float(info.get("overvoltage", 0.0)),
                    "line_overload_pct": float(info.get("line_overload_pct", 0.0)),
                    "transformer_overload_pct": float(info.get("transformer_overload_pct", 0.0)),
                    "max_line_loading_pct": float(info.get("max_line_loading_pct", 0.0)),
                    "max_line_current_ka": float(info.get("max_line_current_ka", 0.0)),
                    "mean_line_loading_pct": float(info.get("mean_line_loading_pct", 0.0)),
                    "temperature_c": float(info.get("temperature_c", 0.0)),
                    "terminal_soc_target": float(info.get("terminal_soc_target", 0.0)),
                    "terminal_soc_tolerance": float(info.get("terminal_soc_tolerance", 0.0)),
                    "terminal_soc_deviation": float(info.get("terminal_soc_deviation", 0.0)),
                    "terminal_soc_excess": float(info.get("terminal_soc_excess", 0.0)),
                    "terminal_soc_excess_kwh": float(info.get("terminal_soc_excess_kwh", 0.0)),
                    "terminal_soc_penalty": float(info.get("terminal_soc_penalty", 0.0)),
                    "power_flow_failure_penalty": float(info.get("power_flow_failure_penalty", 0.0)),
                    "power_flow_failed": int(bool(info.get("power_flow_failed", False))),
                    "cumulative_cost": float(info.get("cumulative_cost", 0.0)),
                    "cumulative_objective_cost": float(info.get("cumulative_objective_cost", info.get("cumulative_cost", 0.0))),
                }
            )
            if terminated or truncated:
                break
        trajectory = pd.DataFrame(rows)
        peak_price_metrics = _peak_price_reserve_metrics(
            trajectory,
            peak_price_threshold=float(getattr(env.unwrapped.config.reward, "peak_price", 0.51373)),
            discharge_limit_scale_w=float(env.unwrapped.config.battery_params.p_discharge_max),
            low_discharge_limit_threshold=float(getattr(args, "train_validation_peak_discharge_limit_threshold", 0.25)),
        )
        inventory_behavior_metrics = _inventory_behavior_metrics(
            trajectory,
            target_soc=float(
                getattr(env.unwrapped.config, "terminal_soc_target", None)
                if getattr(env.unwrapped.config, "terminal_soc_target", None) is not None
                else env.unwrapped.config.battery_params.soc_init
            ),
            target_tolerance=float(getattr(env.unwrapped.config, "terminal_soc_tolerance", 0.0)),
            soc_min=float(env.unwrapped.config.battery_params.soc_min),
            soc_max=float(env.unwrapped.config.battery_params.soc_max),
            valley_price_threshold=float(getattr(env.unwrapped.config.reward, "valley_price", 0.39073)),
            peak_price_threshold=float(getattr(env.unwrapped.config.reward, "peak_price", 0.51373)),
            charge_limit_scale_w=float(env.unwrapped.config.battery_params.p_charge_max),
            discharge_limit_scale_w=float(env.unwrapped.config.battery_params.p_discharge_max),
        )
        summary = {
            "steps": int(len(trajectory)),
            "total_reward": float(total_reward),
            "final_soc": float(trajectory["soc"].iloc[-1]) if not trajectory.empty else 0.0,
            "final_cumulative_cost": float(trajectory["cumulative_cost"].iloc[-1]) if not trajectory.empty else 0.0,
            "final_cumulative_objective_cost": float(trajectory["cumulative_objective_cost"].iloc[-1]) if not trajectory.empty else 0.0,
            "min_voltage_worst": float(trajectory["min_bus_voltage_pu"].min()) if not trajectory.empty else 1.0,
            "max_line_loading_peak": float(trajectory["max_line_loading_pct"].max()) if not trajectory.empty else 0.0,
            "max_line_current_peak_ka": float(trajectory["max_line_current_ka"].max()) if not trajectory.empty else 0.0,
            "mean_grid_import_mw": float(trajectory["grid_import_mw"].mean()) if not trajectory.empty else 0.0,
            "final_temperature_c": float(trajectory["temperature_c"].iloc[-1]) if not trajectory.empty else 0.0,
            "final_terminal_soc_deviation": float(trajectory["terminal_soc_deviation"].iloc[-1]) if not trajectory.empty else 0.0,
            "total_terminal_soc_penalty": float(trajectory["terminal_soc_penalty"].sum()) if not trajectory.empty else 0.0,
            "total_battery_loss_kwh": float(trajectory["battery_loss_kwh"].sum()) if not trajectory.empty else 0.0,
            "total_battery_stress_kwh": float(trajectory["battery_stress_kwh"].sum()) if not trajectory.empty else 0.0,
            "total_battery_throughput_kwh": float(trajectory["battery_throughput_kwh"].sum()) if not trajectory.empty else 0.0,
            "mean_abs_battery_action_delta": float(trajectory["battery_action_delta"].abs().mean()) if not trajectory.empty else 0.0,
            "total_action_rate_penalty": float(trajectory["action_rate_penalty"].sum()) if not trajectory.empty else 0.0,
            "shield_activation_fraction": _dwell_fraction(trajectory["shield_applied"]) if not trajectory.empty else 0.0,
            "shield_reserve_activation_fraction": _dwell_fraction(trajectory["shield_reserve_active"]) if not trajectory.empty else 0.0,
            "shield_boundary_activation_fraction": _dwell_fraction(trajectory["shield_boundary_active"]) if not trajectory.empty else 0.0,
            "shield_terminal_activation_fraction": _dwell_fraction(trajectory["shield_terminal_active"]) if not trajectory.empty else 0.0,
            "mean_abs_shield_delta": float(trajectory["shield_delta"].abs().mean()) if not trajectory.empty else 0.0,
            "shield_material_activation_fraction": _dwell_fraction(trajectory["shield_delta"].abs() > 0.01) if not trajectory.empty else 0.0,
            "shield_strong_activation_fraction": _dwell_fraction(trajectory["shield_delta"].abs() > 0.05) if not trajectory.empty else 0.0,
            "inventory_teacher_activation_fraction": _dwell_fraction(trajectory["inventory_teacher_active"]) if not trajectory.empty else 0.0,
            "inventory_teacher_boundary_activation_fraction": _dwell_fraction(trajectory["inventory_teacher_boundary_active"])
            if not trajectory.empty
            else 0.0,
            "inventory_teacher_terminal_activation_fraction": _dwell_fraction(trajectory["inventory_teacher_terminal_active"])
            if not trajectory.empty
            else 0.0,
            "inventory_teacher_reserve_activation_fraction": _dwell_fraction(trajectory["inventory_teacher_reserve_active"])
            if not trajectory.empty
            else 0.0,
            "mean_abs_inventory_teacher_gap": float(
                (
                    trajectory["inventory_teacher_action"].astype(float)
                    - trajectory["battery_action_applied_pre_shield"].astype(float)
                ).abs().mean()
            )
            if not trajectory.empty
            else 0.0,
            **peak_price_metrics,
            **inventory_behavior_metrics,
            "mean_battery_action_infeasible_gap": float(trajectory["battery_action_infeasible_gap"].mean()) if not trajectory.empty else 0.0,
            "mean_battery_internal_clip_gap_w": float(trajectory["battery_internal_clip_gap_w"].mean()) if not trajectory.empty else 0.0,
            "total_battery_action_infeasible_penalty": float(trajectory["battery_action_infeasible_penalty"].sum()) if not trajectory.empty else 0.0,
            "soc_upper_dwell_fraction": _dwell_fraction(trajectory["soc_upper_bound_hit"]) if not trajectory.empty else 0.0,
            "soc_lower_dwell_fraction": _dwell_fraction(trajectory["soc_lower_bound_hit"]) if not trajectory.empty else 0.0,
            "infeasible_action_dwell_fraction": _dwell_fraction(trajectory["battery_action_infeasible_flag"]) if not trajectory.empty else 0.0,
            "internal_clip_dwell_fraction": _dwell_fraction(trajectory["battery_internal_clip_flag"]) if not trajectory.empty else 0.0,
            "power_flow_failure_steps": int(trajectory["power_flow_failed"].sum()) if not trajectory.empty else 0,
        }
        return summary, trajectory, eval_window
    finally:
        env.close()


def main() -> int:
    raw_argv = sys.argv[1:]
    args = build_parser().parse_args(raw_argv)
    case_keys = _parse_csv_arg(args.cases)
    regimes = _parse_csv_arg(args.regimes)
    train_models = _parse_csv_arg(args.train_models)
    test_models = _parse_csv_arg(args.test_models)
    seeds = _parse_seed_list(args.seeds, args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectories_dir = output_dir / "trajectories"
    trajectories_dir.mkdir(parents=True, exist_ok=True)
    validation_history_dir = output_dir / "validation_history"
    validation_history_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, float | int | str]] = []
    for seed in seeds:
        for case_key in case_keys:
            for regime in regimes:
                for train_model in train_models:
                    run_args = argparse.Namespace(**{**vars(args), "seed": int(seed)})
                    run_args = apply_ieee33_sac_default_protocol(run_args, case_key=case_key, raw_argv=raw_argv)
                    run_args = apply_ieee33_full_fair_protocol(
                        run_args,
                        case_key=case_key,
                        train_model=train_model,
                        raw_argv=raw_argv,
                    )
                    if bool(getattr(run_args, "ieee33_sac_default_protocol_applied", False)):
                        print(
                            "[protocol] IEEE33 SAC research default -> "
                            f"train_year={int(getattr(run_args, 'train_year', 0))} "
                            f"eval_year={int(getattr(run_args, 'eval_year', 0))} "
                            f"train_episode_days={int(getattr(run_args, 'train_episode_days', 0) or int(getattr(run_args, 'days', 0)))} "
                            f"eval_days={int(getattr(run_args, 'eval_days', 0) or int(getattr(run_args, 'days', 0)))} "
                            f"validation_days={int(getattr(run_args, 'train_validation_days', 0))} "
                            f"validation_offsets={str(getattr(run_args, 'train_validation_offset_days_within_year', ''))} "
                            f"checkpoint_every={int(getattr(run_args, 'train_validation_checkpoint_every', 0))} "
                            f"validation_metric={str(getattr(run_args, 'train_validation_metric', ''))}"
                        )
                    if bool(getattr(run_args, "ieee33_full_fair_protocol_applied", False)):
                        print(
                            "[protocol] IEEE33 SAC full-fair preset -> "
                            f"train_steps={int(getattr(run_args, 'train_steps', 0))} "
                            f"learning_rate={float(getattr(run_args, 'learning_rate', 0.0))} "
                            f"rule_guidance_mix={float(getattr(run_args, 'rule_guidance_mix', 0.0))} "
                            f"rule_guidance_decay_steps={int(getattr(run_args, 'rule_guidance_decay_steps', 0))} "
                            f"checkpoint_every={int(getattr(run_args, 'train_validation_checkpoint_every', 0))} "
                            f"validation_metric={str(getattr(run_args, 'train_validation_metric', ''))}"
                        )
                    if bool(getattr(run_args, "ieee33_full_fair_closure_protocol_applied", False)):
                        print(
                            "[protocol] IEEE33 SAC full-fair closure aid -> "
                            f"rule_guidance_policy={str(getattr(run_args, 'rule_guidance_policy', 'rule'))} "
                            f"warmstart_steps={int(getattr(run_args, 'causal_heuristic_warmstart_steps', 0))} "
                            f"warmstart_policy={str(getattr(run_args, 'causal_heuristic_warmstart_policy', 'blended'))}"
                        )
                    print(f"[train] case={case_key} regime={regime} model={train_model} seed={seed} steps={run_args.train_steps}")
                    regularization_cfg = action_regularization_config(run_args)
                    rule_cfg = rule_guidance_config(run_args)
                    agent, train_schedule, train_window, tb_metadata = train_short_agent(case_key=case_key, train_model=train_model, regime=regime, args=run_args)
                    validation_state = dict(tb_metadata.get("validation", {}))
                    validation_history = list(tb_metadata.get("validation_history", []))
                    for test_model in test_models:
                        print(f"[eval] case={case_key} regime={regime} train={train_model} test={test_model} seed={seed}")
                        summary, trajectory, eval_window = evaluate_agent(agent, case_key=case_key, test_model=test_model, regime=regime, args=run_args)
                        row = {
                            "case": case_key,
                            "regime": regime,
                            "reward_profile": str(run_args.reward_profile),
                            "agent": str(run_args.agent),
                            "seed": int(seed),
                            "train_model": train_model,
                            "test_model": test_model,
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
                            "shield_enabled": int(bool(shield_enabled(run_args))),
                            "shield_soc_soft_buffer_fraction": float(shield_config(run_args)["soc_soft_buffer_fraction"]),
                            "shield_soc_hard_buffer_fraction": float(shield_config(run_args)["soc_hard_buffer_fraction"]),
                            "shield_peak_reserve_min_fraction": float(shield_config(run_args)["reserve_discharge_min_fraction"]),
                            "shield_hard_pullback_action": float(shield_config(run_args)["hard_pullback_action"]),
                            "shield_terminal_closure_horizon_fraction": float(
                                shield_config(run_args)["terminal_closure_horizon_fraction"]
                            ),
                            "shield_terminal_closure_urgency_soc": float(shield_config(run_args)["terminal_closure_urgency_soc"]),
                            "rule_guidance_mix": float(rule_cfg["guidance_mix"]),
                            "rule_guidance_decay_steps": int(rule_cfg["guidance_decay_steps"]),
                            "rule_guidance_policy": str(rule_cfg["guidance_policy"]),
                            "protocol_profile": str(protocol_profile(run_args)),
                            "ieee33_sac_default_protocol_applied": int(
                                bool(getattr(run_args, "ieee33_sac_default_protocol_applied", False))
                            ),
                            "ieee33_full_fair_protocol_applied": int(
                                bool(getattr(run_args, "ieee33_full_fair_protocol_applied", False))
                            ),
                            "ieee33_full_fair_closure_protocol_applied": int(
                                bool(getattr(run_args, "ieee33_full_fair_closure_protocol_applied", False))
                            ),
                            "train_disable_explicit_battery_degradation_penalties": int(
                                bool(getattr(run_args, "train_disable_explicit_battery_degradation_penalties", False))
                            ),
                            "train_year": int(getattr(run_args, "train_year", 0)),
                            "eval_year": int(getattr(run_args, "eval_year", 0)),
                            "train_episode_days": int(train_window["days"]) if train_window is not None else int(run_args.days),
                            "eval_config_days": int(eval_window["days"]) if eval_window is not None else int(run_args.days),
                            "train_window_start": str(train_window["window_start_timestamp"]) if train_window is not None else "",
                            "train_window_end": str(train_window["window_end_timestamp"]) if train_window is not None else "",
                            "eval_window_start": str(eval_window["window_start_timestamp"]) if eval_window is not None else "",
                            "eval_window_end": str(eval_window["window_end_timestamp"]) if eval_window is not None else "",
                            "train_random_start_within_year": int(bool(train_window["random_episode_start"])) if train_window is not None else 0,
                            "train_validation_days": int(getattr(run_args, "train_validation_days", 0)),
                            "train_validation_offset_days_within_year": str(getattr(run_args, "train_validation_offset_days_within_year", "")),
                            "train_validation_window_count": len(tb_metadata.get("validation_windows", [])),
                            "train_validation_checkpoint_every": int(getattr(run_args, "train_validation_checkpoint_every", 0)),
                            "train_validation_metric": str(getattr(run_args, "train_validation_metric", "objective_cost")),
                            "train_validation_terminal_penalty_weight": float(getattr(run_args, "train_validation_terminal_penalty_weight", 1.0)),
                            "train_validation_boundary_dwell_weight": float(getattr(run_args, "train_validation_boundary_dwell_weight", 20000.0)),
                            "train_validation_infeasible_dwell_weight": float(getattr(run_args, "train_validation_infeasible_dwell_weight", 20000.0)),
                             "train_validation_peak_reserve_weight": float(getattr(run_args, "train_validation_peak_reserve_weight", 0.0)),
                             "train_validation_peak_discharge_limit_threshold": float(
                                 getattr(run_args, "train_validation_peak_discharge_limit_threshold", 0.25)
                             ),
                            "train_validation_gate_dwell_threshold": float(
                                getattr(run_args, "train_validation_gate_dwell_threshold", 0.05)
                            ),
                            "train_validation_gate_violation_weight": float(
                                getattr(run_args, "train_validation_gate_violation_weight", 1_000_000.0)
                            ),
                            "train_validation_gate_peak_reserve_dwell_threshold": float(
                                getattr(run_args, "train_validation_gate_peak_reserve_dwell_threshold", -1.0)
                            ),
                            "train_validation_shield_mean_delta_weight": float(
                                getattr(run_args, "train_validation_shield_mean_delta_weight", 0.0)
                            ),
                            "train_validation_shield_material_dwell_weight": float(
                                getattr(run_args, "train_validation_shield_material_dwell_weight", 0.0)
                            ),
                            "train_validation_shield_strong_dwell_weight": float(
                                getattr(run_args, "train_validation_shield_strong_dwell_weight", 0.0)
                            ),
                            "train_validation_final_soc_deviation_weight": float(
                                getattr(run_args, "train_validation_final_soc_deviation_weight", 0.0)
                            ),
                            "train_validation_shield_mean_delta_threshold": float(
                                getattr(run_args, "train_validation_shield_mean_delta_threshold", -1.0)
                            ),
                            "train_validation_shield_material_dwell_threshold": float(
                                getattr(run_args, "train_validation_shield_material_dwell_threshold", -1.0)
                            ),
                            "train_validation_shield_strong_dwell_threshold": float(
                                getattr(run_args, "train_validation_shield_strong_dwell_threshold", -1.0)
                            ),
                             "train_validation_midband_dwell_weight": float(
                                 getattr(run_args, "train_validation_midband_dwell_weight", 10000.0)
                             ),
                            "train_validation_soc_target_tracking_weight": float(
                                getattr(run_args, "train_validation_soc_target_tracking_weight", 5000.0)
                            ),
                            "train_validation_peak_discharge_headroom_weight": float(
                                getattr(run_args, "train_validation_peak_discharge_headroom_weight", 10000.0)
                            ),
                            "train_validation_valley_charge_weight": float(
                                getattr(run_args, "train_validation_valley_charge_weight", 5000.0)
                            ),
                            "train_validation_peak_discharge_weight": float(
                                getattr(run_args, "train_validation_peak_discharge_weight", 5000.0)
                            ),
                            "validation_best_metric_value": float(validation_state.get("best_metric_value", np.nan)),
                            "validation_best_total_reward": float(validation_state.get("best_total_reward", np.nan)),
                            "validation_best_objective_cost": float(validation_state.get("best_objective_cost", np.nan)),
                            "validation_best_checkpoint_step": int(validation_state.get("best_checkpoint_step", int(run_args.train_steps))),
                            "offline_dataset": str(getattr(run_args, "offline_dataset", "")),
                            "offline_dataset_controller_sources": str(getattr(run_args, "offline_dataset_controller_sources", "")),
                            "offline_dataset_max_transitions": int(getattr(run_args, "offline_dataset_max_transitions", 0)),
                            "bc_pretrain_gradient_steps": int(getattr(run_args, "bc_pretrain_gradient_steps", 0)),
                            "bc_pretrain_batch_size": int(getattr(run_args, "bc_pretrain_batch_size", 256)),
                            "bc_pretrain_learning_rate": float(getattr(run_args, "bc_pretrain_learning_rate", 0.0)),
                            "shield_delta_penalty_coef": float(getattr(run_args, "shield_delta_penalty_coef", 0.0)),
                            "shield_delta_penalty_start": float(shield_config(run_args)["shield_delta_penalty_start"]),
                            "shield_delta_penalty_end": float(shield_config(run_args)["shield_delta_penalty_end"]),
                            "shield_delta_penalty_warmup_steps": int(
                                shield_config(run_args)["shield_delta_penalty_warmup_steps"]
                            ),
                             "online_safe_bc_gradient_steps": int(getattr(run_args, "online_safe_bc_gradient_steps", 0)),
                             "online_safe_bc_batch_size": int(getattr(run_args, "online_safe_bc_batch_size", 256)),
                             "online_safe_bc_max_samples": int(getattr(run_args, "online_safe_bc_max_samples", 0)),
                             "online_safe_bc_learning_rate": float(getattr(run_args, "online_safe_bc_learning_rate", 0.0)),
                             "online_safe_bc_replay_rows": int(validation_state.get("online_safe_bc_replay_rows", 0)),
                             "online_safe_bc_actor_gradient_steps_applied": int(validation_state.get("online_safe_bc_actor_gradient_steps", 0)),
                             "online_safe_bc_inventory_teacher_rows": int(validation_state.get("online_safe_bc_inventory_teacher_rows", 0)),
                             "online_safe_bc_intervention_rows": int(validation_state.get("online_safe_bc_intervention_rows", 0)),
                             "online_safe_bc_mean_sample_weight": float(validation_state.get("online_safe_bc_mean_sample_weight", 1.0)),
                             "effective_online_safe_bc_gradient_steps": int(validation_state.get("effective_online_safe_bc_gradient_steps", 0)),
                             "effective_online_safe_bc_priority_scale": float(
                                 validation_state.get("effective_online_safe_bc_priority_scale", 1.0)
                             ),
                             "online_safe_bc_intervention_priority_coef": float(getattr(run_args, "online_safe_bc_intervention_priority_coef", 4.0)),
                             "online_safe_bc_boundary_priority_coef": float(getattr(run_args, "online_safe_bc_boundary_priority_coef", 2.0)),
                             "online_safe_bc_terminal_priority_coef": float(getattr(run_args, "online_safe_bc_terminal_priority_coef", 2.0)),
                            "online_safe_bc_teacher_priority_coef": float(getattr(run_args, "online_safe_bc_teacher_priority_coef", 2.0)),
                            "online_safe_bc_peak_value_priority_coef": float(getattr(run_args, "online_safe_bc_peak_value_priority_coef", 0.75)),
                            "online_safe_bc_valley_value_priority_coef": float(getattr(run_args, "online_safe_bc_valley_value_priority_coef", 0.5)),
                            "offline_bc_dataset_rows": int(validation_state.get("offline_bc_dataset_rows", 0)),
                            "offline_bc_replay_seeded_transitions": int(validation_state.get("offline_bc_replay_seeded_transitions", 0)),
                            "offline_bc_actor_gradient_steps": int(validation_state.get("offline_bc_actor_gradient_steps", 0)),
                            "offline_bc_actor_batch_size": int(validation_state.get("offline_bc_actor_batch_size", 0)),
                            "offline_bc_initial_actor_mse": float(validation_state.get("offline_bc_initial_actor_mse", np.nan)),
                            "offline_bc_final_actor_mse": float(validation_state.get("offline_bc_final_actor_mse", np.nan)),
                            "online_safe_bc_replay_rows": int(validation_state.get("online_safe_bc_replay_rows", 0)),
                            "online_safe_bc_actor_gradient_steps_applied": int(validation_state.get("online_safe_bc_actor_gradient_steps", 0)),
                            "online_safe_bc_initial_actor_mse": float(validation_state.get("online_safe_bc_initial_actor_mse", np.nan)),
                            "online_safe_bc_final_actor_mse": float(validation_state.get("online_safe_bc_final_actor_mse", np.nan)),
                            "online_safe_bc_intervention_rows": int(validation_state.get("online_safe_bc_intervention_rows", 0)),
                            "online_safe_bc_inventory_teacher_rows": int(validation_state.get("online_safe_bc_inventory_teacher_rows", 0)),
                            "online_safe_bc_mean_sample_weight": float(validation_state.get("online_safe_bc_mean_sample_weight", 1.0)),
                            "effective_online_safe_bc_gradient_steps": int(validation_state.get("effective_online_safe_bc_gradient_steps", 0)),
                            "last_validation_mean_shield_material_activation_fraction": float(
                                validation_state.get("last_validation_mean_shield_material_activation_fraction", np.nan)
                            ),
                            "last_validation_mean_abs_shield_delta": float(
                                validation_state.get("last_validation_mean_abs_shield_delta", np.nan)
                            ),
                            "last_validation_mean_soc_midband_dwell_fraction": float(
                                validation_state.get("last_validation_mean_soc_midband_dwell_fraction", np.nan)
                            ),
                            "last_validation_mean_soc_target_tracking_mae": float(
                                validation_state.get("last_validation_mean_soc_target_tracking_mae", np.nan)
                            ),
                            "last_validation_mean_peak_price_discharge_action_fraction": float(
                                validation_state.get("last_validation_mean_peak_price_discharge_action_fraction", np.nan)
                            ),
                            "last_validation_mean_valley_price_charge_action_fraction": float(
                                validation_state.get("last_validation_mean_valley_price_charge_action_fraction", np.nan)
                            ),
                            "last_validation_mean_inventory_teacher_activation_fraction": float(
                                validation_state.get("last_validation_mean_inventory_teacher_activation_fraction", np.nan)
                            ),
                            "last_validation_mean_abs_inventory_teacher_gap": float(
                                validation_state.get("last_validation_mean_abs_inventory_teacher_gap", np.nan)
                            ),
                            "causal_heuristic_warmstart_steps": int(getattr(run_args, "causal_heuristic_warmstart_steps", 0)),
                            "causal_heuristic_warmstart_policy": str(getattr(run_args, "causal_heuristic_warmstart_policy", "blended")),
                            "causal_heuristic_warmstart_steps_applied": int(validation_state.get("warmstart_steps_applied", 0)),
                            "eval_full_horizon": int(bool(getattr(run_args, "eval_full_horizon", False))),
                            "mixed_fidelity_stage_fractions": str(getattr(run_args, "mixed_fidelity_stage_fractions", "")),
                            "mixed_fidelity_stage_learning_rates": str(getattr(run_args, "mixed_fidelity_stage_learning_rates", "")),
                            "resolved_train_stages": ",".join(str(stage) for stage in train_schedule["stages"]),
                            "resolved_train_stage_count": int(train_schedule["stage_count"]),
                            "resolved_train_stage_fractions": ",".join(f"{float(value):.6f}" for value in train_schedule["stage_fractions"]),
                            "resolved_train_stage_steps": ",".join(str(int(value)) for value in train_schedule["stage_steps"]),
                            "resolved_train_stage_learning_rates": ",".join(f"{float(value):.8g}" for value in train_schedule["stage_learning_rates"]),
                            **summary,
                        }
                        summary_rows.append(row)
                        stem = f"{case_key}_{regime}_{args.agent}_train-{train_model}_test-{test_model}_seed{seed}"
                        trajectories_dir.mkdir(parents=True, exist_ok=True)
                        trajectory.to_csv(trajectories_dir / f"{stem}.csv", index=False)
                        if validation_history:
                            pd.DataFrame(validation_history).to_csv(validation_history_dir / f"{stem}.csv", index=False)

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        ordered_columns = [
            "case",
            "regime",
            "train_model",
            "test_model",
            "seed",
            "reward_profile",
            "agent",
            "train_steps",
            "eval_steps",
            "steps",
            "final_cumulative_objective_cost",
            "final_cumulative_cost",
            "total_reward",
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
            "peak_price_low_discharge_limit_dwell_fraction",
            "infeasible_action_dwell_fraction",
            "total_battery_throughput_kwh",
            "total_battery_loss_kwh",
            "total_battery_stress_kwh",
            "mean_grid_import_mw",
            "min_voltage_worst",
            "max_line_loading_peak",
            "max_line_current_peak_ka",
            "final_temperature_c",
            "shield_activation_fraction",
            "shield_reserve_activation_fraction",
            "shield_boundary_activation_fraction",
            "shield_terminal_activation_fraction",
            "mean_abs_shield_delta",
            "shield_material_activation_fraction",
            "shield_strong_activation_fraction",
            "inventory_teacher_activation_fraction",
            "inventory_teacher_boundary_activation_fraction",
            "inventory_teacher_terminal_activation_fraction",
            "inventory_teacher_reserve_activation_fraction",
            "mean_abs_inventory_teacher_gap",
            "validation_best_metric_value",
            "validation_best_total_reward",
            "validation_best_objective_cost",
            "validation_best_checkpoint_step",
            "learning_rate",
            "offline_dataset",
            "offline_dataset_controller_sources",
            "offline_dataset_max_transitions",
            "bc_pretrain_gradient_steps",
            "bc_pretrain_batch_size",
            "bc_pretrain_learning_rate",
            "shield_delta_penalty_coef",
            "shield_delta_penalty_start",
            "shield_delta_penalty_end",
            "shield_delta_penalty_warmup_steps",
             "online_safe_bc_gradient_steps",
             "online_safe_bc_batch_size",
             "online_safe_bc_max_samples",
             "online_safe_bc_learning_rate",
             "online_safe_bc_replay_rows",
             "online_safe_bc_actor_gradient_steps_applied",
             "online_safe_bc_inventory_teacher_rows",
             "online_safe_bc_intervention_rows",
             "online_safe_bc_mean_sample_weight",
             "effective_online_safe_bc_gradient_steps",
             "effective_online_safe_bc_priority_scale",
             "offline_bc_dataset_rows",
            "offline_bc_replay_seeded_transitions",
            "offline_bc_actor_gradient_steps",
            "offline_bc_actor_batch_size",
            "offline_bc_initial_actor_mse",
            "offline_bc_final_actor_mse",
            "causal_heuristic_warmstart_steps",
            "causal_heuristic_warmstart_policy",
            "causal_heuristic_warmstart_steps_applied",
            "eval_full_horizon",
            "mixed_fidelity_stage_fractions",
            "mixed_fidelity_stage_learning_rates",
            "resolved_train_stages",
            "resolved_train_stage_count",
            "resolved_train_stage_fractions",
            "resolved_train_stage_steps",
            "resolved_train_stage_learning_rates",
            "mean_abs_battery_action_delta",
            "total_action_rate_penalty",
            "peak_price_step_fraction",
            "peak_price_mean_soc",
            "valley_price_step_fraction",
            "mean_charge_headroom_ratio",
            "mean_discharge_headroom_ratio",
            "mean_battery_action_infeasible_gap",
            "mean_battery_internal_clip_gap_w",
            "total_battery_action_infeasible_penalty",
            "internal_clip_dwell_fraction",
            "power_flow_failure_steps",
            "tensorboard_log_dir",
            "tensorboard_run_name",
            "action_smoothing_coef",
            "action_max_delta",
            "action_rate_penalty",
            "battery_feasibility_aware",
            "battery_infeasible_penalty",
            "symmetric_battery_action",
            "shield_enabled",
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
            "ieee33_sac_default_protocol_applied",
            "ieee33_full_fair_protocol_applied",
            "ieee33_full_fair_closure_protocol_applied",
            "train_disable_explicit_battery_degradation_penalties",
            "train_year",
            "eval_year",
            "train_episode_days",
            "eval_config_days",
            "train_window_start",
            "train_window_end",
            "eval_window_start",
            "eval_window_end",
            "train_random_start_within_year",
            "train_validation_days",
            "train_validation_offset_days_within_year",
            "train_validation_window_count",
            "train_validation_checkpoint_every",
            "train_validation_metric",
            "train_validation_terminal_penalty_weight",
            "train_validation_boundary_dwell_weight",
            "train_validation_infeasible_dwell_weight",
             "train_validation_peak_reserve_weight",
             "train_validation_peak_discharge_limit_threshold",
             "train_validation_gate_dwell_threshold",
             "train_validation_gate_violation_weight",
             "train_validation_gate_peak_reserve_dwell_threshold",
             "train_validation_shield_mean_delta_weight",
             "train_validation_shield_material_dwell_weight",
             "train_validation_shield_strong_dwell_weight",
             "train_validation_final_soc_deviation_weight",
             "train_validation_shield_mean_delta_threshold",
             "train_validation_shield_material_dwell_threshold",
             "train_validation_shield_strong_dwell_threshold",
             "train_validation_midband_dwell_weight",
            "train_validation_soc_target_tracking_weight",
            "train_validation_peak_discharge_headroom_weight",
            "train_validation_valley_charge_weight",
            "train_validation_peak_discharge_weight",
        ]
        summary_df = summary_df[[col for col in ordered_columns if col in summary_df.columns]]
    summary_csv = output_dir / "summary.csv"
    summary_json = output_dir / "summary.json"
    summary_df.to_csv(summary_csv, index=False)
    summary_json.write_text(json.dumps(summary_df.to_dict(orient="records"), indent=2), encoding="utf-8")

    print("\n=== Short Cross-Fidelity Summary ===")
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary CSV: {summary_csv}")
    print(f"Saved summary JSON: {summary_json}")
    print(f"Saved trajectories: {trajectories_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
