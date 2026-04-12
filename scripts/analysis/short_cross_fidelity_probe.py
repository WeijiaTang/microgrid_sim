#!/usr/bin/env python3
"""Run short train-test mismatch probes on the network-first microgrid cases."""
# Ref: docs/spec/task.md (Task-ID: SPEC-FIDELITY-MISMATCH-001)

from __future__ import annotations

import argparse
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
from microgrid_sim.envs.wrappers import ContinuousActionRegularizationWrapper
from microgrid_sim.rl_utils import create_agent
from microgrid_sim.time_utils import steps_per_day, steps_per_hour


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Short cross-fidelity train-test probe for network microgrids.")
    parser.add_argument("--cases", type=str, default="cigre,ieee33", help="Comma-separated case keys: cigre, ieee33")
    parser.add_argument("--regimes", type=str, default="base", help="Comma-separated operating regimes: base, high_load, high_pv, network_stress, tight_soc")
    parser.add_argument(
        "--train-models",
        type=str,
        default="simple,thevenin",
        help="Comma-separated battery-model training specs. Supports single-stage specs like none/simple/thevenin/thevenin_loss_only and mixed specs like simple+thevenin.",
    )
    parser.add_argument("--test-models", type=str, default="simple,thevenin", help="Comma-separated battery models for evaluation")
    parser.add_argument("--reward-profile", type=str, default="network", help="Reward profile: network, paper_aligned, or paper_balanced")
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
    parser.add_argument("--battery-infeasible-penalty", type=float, default=0.0, help="Penalty weight for requesting battery actions outside the current SOC-feasible range")
    parser.add_argument(
        "--symmetric-battery-action",
        action="store_true",
        help="Scale positive battery actions to enforce symmetric usable charge/discharge range",
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


def _parse_train_spec(train_spec: str) -> list[str]:
    stages = [part.strip().lower() for part in str(train_spec).split("+") if part.strip()]
    if not stages:
        raise ValueError(f"Unsupported empty training spec '{train_spec}'.")
    for stage in stages:
        if stage not in {"none", "simple", "thevenin", "thevenin_loss_only"}:
            raise ValueError(f"Unsupported training stage '{stage}' in '{train_spec}'. Expected none/simple/thevenin/thevenin_loss_only.")
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
    if offset_days >= max_days:
        raise ValueError(
            f"Requested start_offset_days_within_year={offset_days} exceeds available days={max_days} for year {year} case '{case_key}'."
        )
    if offset_days + resolved_episode_days > max_days:
        raise ValueError(
            f"Requested window offset {offset_days} plus episode_days={resolved_episode_days} exceeds available days={max_days} for year {year} case '{case_key}'."
        )
    resolved_stride_hours = max(int(stride_hours), 1)
    explicit_start_hours: tuple[int, ...] = tuple()
    random_episode_start = bool(random_start_within_year and resolved_episode_days < max_days)
    base_start_hour = int(year_window["start_hour"]) + offset_days * 24
    base_start_step = int(year_window["start_step"]) + offset_days * 24 * steps_per_hour(probe_config.dt_seconds)
    if random_episode_start:
        admissible_days = max_days - offset_days - resolved_episode_days
        explicit_start_hours = tuple(
            base_start_hour + offset_hour
            for offset_hour in range(0, admissible_days * 24 + 1, resolved_stride_hours)
        )
        if not explicit_start_hours:
            explicit_start_hours = (base_start_hour,)
            random_episode_start = False
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
    }


def action_regularization_config(args: argparse.Namespace) -> dict[str, float | bool]:
    return {
        "smoothing_coef": float(getattr(args, "action_smoothing_coef", 0.0)),
        "max_delta": float(getattr(args, "action_max_delta", 0.0)),
        "rate_penalty": float(getattr(args, "action_rate_penalty", 0.0)),
        "battery_feasibility_aware": bool(getattr(args, "battery_feasibility_aware", False)),
        "battery_infeasible_penalty": float(getattr(args, "battery_infeasible_penalty", 0.0)),
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
            float(config["battery_infeasible_penalty"]) > 0.0,
            bool(config["symmetric_battery_action"]),
        )
    )


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


def build_env(
    case_key: str,
    battery_model: str,
    days: int,
    seed: int,
    regime: str,
    args: argparse.Namespace | None = None,
    window_metadata: dict[str, int | str | tuple[int, ...] | bool] | None = None,
):
    reward_profile = str(getattr(args, "reward_profile", "network")) if args is not None else "network"
    config = build_config(
        case_key=case_key,
        battery_model=battery_model,
        days=days,
        seed=seed,
        regime=regime,
        reward_profile=reward_profile,
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
    if args is not None and action_regularization_enabled(args):
        env = ContinuousActionRegularizationWrapper(env=env, **action_regularization_config(args))
    return env


def _dwell_fraction(mask: pd.Series | list[bool]) -> float:
    series = pd.Series(mask, dtype=float)
    if series.empty:
        return 0.0
    return float(series.mean())


def resolve_train_window(case_key: str, regime: str, args: argparse.Namespace) -> dict[str, int | str | tuple[int, ...] | bool] | None:
    train_year = int(getattr(args, "train_year", 0))
    if train_year <= 0:
        return None
    train_episode_days = int(getattr(args, "train_episode_days", 0)) or int(args.days)
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
    )


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


def train_short_agent(case_key: str, train_model: str, regime: str, args: argparse.Namespace):
    schedule = resolve_training_schedule(train_model=train_model, args=args)
    stages = [str(stage) for stage in schedule["stages"]]
    stage_steps = [int(value) for value in schedule["stage_steps"]]
    stage_learning_rates = [float(value) for value in schedule["stage_learning_rates"]]
    base_learning_rate = float(getattr(args, "learning_rate", 3e-4))
    train_window = resolve_train_window(case_key=case_key, regime=regime, args=args)
    tensorboard_log_dir = resolve_tensorboard_log_dir(args)
    tensorboard_run_name = default_tb_log_name(
        case_key=case_key,
        regime=regime,
        train_model=train_model,
        seed=int(args.seed),
        args=args,
    )
    env = build_env(
        case_key=case_key,
        battery_model=stages[0],
        days=args.days,
        seed=args.seed,
        regime=regime,
        args=args,
        window_metadata=train_window,
    )
    try:
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
        if stage_steps[0] > 0:
            _set_agent_learning_rate(agent, stage_learning_rates[0])
            learn_agent(
                agent,
                total_timesteps=int(stage_steps[0]),
                progress_bar=False,
                tb_log_name=f"{tensorboard_run_name}_stage1of{len(stages)}" if tensorboard_log_dir else None,
            )
        for stage_index, (stage_model, steps, stage_learning_rate) in enumerate(
            zip(stages[1:], stage_steps[1:], stage_learning_rates[1:]),
            start=2,
        ):
            if steps <= 0:
                continue
            next_env = build_env(
                case_key=case_key,
                battery_model=stage_model,
                days=args.days,
                seed=args.seed,
                regime=regime,
                args=args,
                window_metadata=train_window,
            )
            try:
                agent.set_env(next_env)
                _set_agent_learning_rate(agent, stage_learning_rate)
                learn_agent(
                    agent,
                    total_timesteps=int(steps),
                    progress_bar=False,
                    reset_num_timesteps=False,
                    tb_log_name=f"{tensorboard_run_name}_stage{stage_index}of{len(stages)}" if tensorboard_log_dir else None,
                )
            finally:
                next_env.close()
        return agent, schedule, train_window, {"tensorboard_log_dir": tensorboard_log_dir or "", "tensorboard_run_name": tensorboard_run_name}
    finally:
        env.close()


def evaluate_agent(agent, case_key: str, test_model: str, regime: str, args: argparse.Namespace) -> tuple[dict, pd.DataFrame]:
    eval_window = resolve_eval_window(case_key=case_key, regime=regime, args=args)
    env = build_env(
        case_key=case_key,
        battery_model=test_model,
        days=args.days,
        seed=args.seed,
        regime=regime,
        args=args,
        window_metadata=eval_window,
    )
    rows: list[dict[str, float | int | str]] = []
    try:
        soc_min = float(env.unwrapped.config.battery_params.soc_min)
        soc_max = float(env.unwrapped.config.battery_params.soc_max)
        soc_tol = 1e-4
        infeasible_gap_tol = 1e-6
        obs, info = env.reset()
        total_reward = 0.0
        max_eval_steps = int(env.unwrapped.total_steps) if bool(getattr(args, "eval_full_horizon", False)) else int(args.eval_steps)
        if max_eval_steps <= 0:
            max_eval_steps = int(env.unwrapped.total_steps)
        for step in range(max_eval_steps):
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            rows.append(
                {
                    "step": int(step),
                    "timestamp": str(info.get("timestamp", "")),
                    "reward": float(reward),
                    "soc": float(info.get("soc", 0.0)),
                    "grid_import_mw": float(info.get("grid_import_mw", 0.0)),
                    "grid_export_mw": float(info.get("grid_export_mw", 0.0)),
                    "battery_power_mw": float(info.get("battery_power_mw", 0.0)),
                    "battery_loss_kwh": float(info.get("battery_loss_kwh", 0.0)),
                    "battery_stress_kwh": float(info.get("battery_stress_kwh", 0.0)),
                    "battery_throughput_kwh": float(info.get("battery_throughput_kwh", 0.0)),
                    "battery_action_raw": float(info.get("battery_action_raw", 0.0)),
                    "battery_action_applied": float(info.get("battery_action_applied", 0.0)),
                    "battery_action_delta": float(info.get("battery_action_delta", 0.0)),
                    "action_rate_penalty": float(info.get("action_rate_penalty", 0.0)),
                    "battery_action_feasible_low": float(info.get("battery_action_feasible_low", -1.0)),
                    "battery_action_feasible_high": float(info.get("battery_action_feasible_high", 1.0)),
                    "battery_charge_fraction_feasible": float(info.get("battery_charge_fraction_feasible", 1.0)),
                    "battery_discharge_fraction_feasible": float(info.get("battery_discharge_fraction_feasible", 1.0)),
                    "battery_action_infeasible_gap": float(info.get("battery_action_infeasible_gap", 0.0)),
                    "battery_action_infeasible_penalty": float(info.get("battery_action_infeasible_penalty", 0.0)),
                    "soc_upper_bound_hit": int(float(info.get("soc", 0.0)) >= soc_max - soc_tol),
                    "soc_lower_bound_hit": int(float(info.get("soc", 0.0)) <= soc_min + soc_tol),
                    "battery_action_infeasible_flag": int(float(info.get("battery_action_infeasible_gap", 0.0)) > infeasible_gap_tol),
                    "min_bus_voltage_pu": float(info.get("min_bus_voltage_pu", 1.0)),
                    "max_line_loading_pct": float(info.get("max_line_loading_pct", 0.0)),
                    "max_line_current_ka": float(info.get("max_line_current_ka", 0.0)),
                    "mean_line_loading_pct": float(info.get("mean_line_loading_pct", 0.0)),
                    "temperature_c": float(info.get("temperature_c", 0.0)),
                    "terminal_soc_target": float(info.get("terminal_soc_target", 0.0)),
                    "terminal_soc_tolerance": float(info.get("terminal_soc_tolerance", 0.0)),
                    "terminal_soc_deviation": float(info.get("terminal_soc_deviation", 0.0)),
                    "terminal_soc_excess": float(info.get("terminal_soc_excess", 0.0)),
                    "terminal_soc_penalty": float(info.get("terminal_soc_penalty", 0.0)),
                    "power_flow_failed": int(bool(info.get("power_flow_failed", False))),
                    "cumulative_cost": float(info.get("cumulative_cost", 0.0)),
                    "cumulative_objective_cost": float(info.get("cumulative_objective_cost", info.get("cumulative_cost", 0.0))),
                }
            )
            if terminated or truncated:
                break
        trajectory = pd.DataFrame(rows)
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
            "mean_battery_action_infeasible_gap": float(trajectory["battery_action_infeasible_gap"].mean()) if not trajectory.empty else 0.0,
            "total_battery_action_infeasible_penalty": float(trajectory["battery_action_infeasible_penalty"].sum()) if not trajectory.empty else 0.0,
            "soc_upper_dwell_fraction": _dwell_fraction(trajectory["soc_upper_bound_hit"]) if not trajectory.empty else 0.0,
            "soc_lower_dwell_fraction": _dwell_fraction(trajectory["soc_lower_bound_hit"]) if not trajectory.empty else 0.0,
            "infeasible_action_dwell_fraction": _dwell_fraction(trajectory["battery_action_infeasible_flag"]) if not trajectory.empty else 0.0,
            "power_flow_failure_steps": int(trajectory["power_flow_failed"].sum()) if not trajectory.empty else 0,
        }
        return summary, trajectory, eval_window
    finally:
        env.close()


def main() -> int:
    args = build_parser().parse_args()
    case_keys = _parse_csv_arg(args.cases)
    regimes = _parse_csv_arg(args.regimes)
    train_models = _parse_csv_arg(args.train_models)
    test_models = _parse_csv_arg(args.test_models)
    seeds = _parse_seed_list(args.seeds, args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectories_dir = output_dir / "trajectories"
    trajectories_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, float | int | str]] = []
    for seed in seeds:
        for case_key in case_keys:
            for regime in regimes:
                for train_model in train_models:
                    print(f"[train] case={case_key} regime={regime} model={train_model} seed={seed} steps={args.train_steps}")
                    run_args = argparse.Namespace(**{**vars(args), "seed": int(seed)})
                    agent, train_schedule, train_window, tb_metadata = train_short_agent(case_key=case_key, train_model=train_model, regime=regime, args=run_args)
                    for test_model in test_models:
                        print(f"[eval] case={case_key} regime={regime} train={train_model} test={test_model} seed={seed}")
                        summary, trajectory, eval_window = evaluate_agent(agent, case_key=case_key, test_model=test_model, regime=regime, args=run_args)
                        row = {
                            "case": case_key,
                            "regime": regime,
                            "reward_profile": str(args.reward_profile),
                            "agent": str(args.agent),
                            "seed": int(seed),
                            "train_model": train_model,
                            "test_model": test_model,
                            "train_steps": int(args.train_steps),
                            "eval_steps": int(summary["steps"]),
                            "learning_rate": float(args.learning_rate),
                            "tensorboard_log_dir": str(tb_metadata["tensorboard_log_dir"]),
                            "tensorboard_run_name": str(tb_metadata["tensorboard_run_name"]),
                            "action_smoothing_coef": float(args.action_smoothing_coef),
                            "action_max_delta": float(args.action_max_delta),
                            "action_rate_penalty": float(args.action_rate_penalty),
                            "battery_feasibility_aware": int(bool(args.battery_feasibility_aware)),
                            "battery_infeasible_penalty": float(args.battery_infeasible_penalty),
                            "symmetric_battery_action": int(bool(args.symmetric_battery_action)),
                            "train_year": int(getattr(args, "train_year", 0)),
                            "eval_year": int(getattr(args, "eval_year", 0)),
                            "train_episode_days": int(train_window["days"]) if train_window is not None else int(args.days),
                            "eval_config_days": int(eval_window["days"]) if eval_window is not None else int(args.days),
                            "train_window_start": str(train_window["window_start_timestamp"]) if train_window is not None else "",
                            "train_window_end": str(train_window["window_end_timestamp"]) if train_window is not None else "",
                            "eval_window_start": str(eval_window["window_start_timestamp"]) if eval_window is not None else "",
                            "eval_window_end": str(eval_window["window_end_timestamp"]) if eval_window is not None else "",
                            "train_random_start_within_year": int(bool(train_window["random_episode_start"])) if train_window is not None else 0,
                            "eval_full_horizon": int(bool(getattr(args, "eval_full_horizon", False))),
                            "mixed_fidelity_stage_fractions": str(getattr(args, "mixed_fidelity_stage_fractions", "")),
                            "mixed_fidelity_stage_learning_rates": str(getattr(args, "mixed_fidelity_stage_learning_rates", "")),
                            "resolved_train_stages": ",".join(str(stage) for stage in train_schedule["stages"]),
                            "resolved_train_stage_count": int(train_schedule["stage_count"]),
                            "resolved_train_stage_fractions": ",".join(f"{float(value):.6f}" for value in train_schedule["stage_fractions"]),
                            "resolved_train_stage_steps": ",".join(str(int(value)) for value in train_schedule["stage_steps"]),
                            "resolved_train_stage_learning_rates": ",".join(f"{float(value):.8g}" for value in train_schedule["stage_learning_rates"]),
                            **summary,
                        }
                        summary_rows.append(row)
                        stem = f"{case_key}_{regime}_{args.agent}_train-{train_model}_test-{test_model}_seed{seed}"
                        trajectory.to_csv(trajectories_dir / f"{stem}.csv", index=False)

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        ordered_columns = [
            "case",
            "regime",
            "reward_profile",
            "agent",
            "seed",
            "train_model",
            "test_model",
            "train_steps",
            "eval_steps",
            "learning_rate",
            "tensorboard_log_dir",
            "tensorboard_run_name",
            "action_smoothing_coef",
            "action_max_delta",
            "action_rate_penalty",
            "battery_feasibility_aware",
            "battery_infeasible_penalty",
            "symmetric_battery_action",
            "train_year",
            "eval_year",
            "train_episode_days",
            "eval_config_days",
            "train_window_start",
            "train_window_end",
            "eval_window_start",
            "eval_window_end",
            "train_random_start_within_year",
            "eval_full_horizon",
            "mixed_fidelity_stage_fractions",
            "mixed_fidelity_stage_learning_rates",
            "resolved_train_stages",
            "resolved_train_stage_count",
            "resolved_train_stage_fractions",
            "resolved_train_stage_steps",
            "resolved_train_stage_learning_rates",
            "steps",
            "total_reward",
            "final_soc",
            "final_temperature_c",
            "final_cumulative_cost",
            "final_cumulative_objective_cost",
            "final_terminal_soc_deviation",
            "total_terminal_soc_penalty",
            "min_voltage_worst",
            "max_line_loading_peak",
            "max_line_current_peak_ka",
            "mean_grid_import_mw",
            "total_battery_loss_kwh",
            "total_battery_stress_kwh",
            "total_battery_throughput_kwh",
            "mean_abs_battery_action_delta",
            "total_action_rate_penalty",
            "mean_battery_action_infeasible_gap",
            "total_battery_action_infeasible_penalty",
            "soc_upper_dwell_fraction",
            "soc_lower_dwell_fraction",
            "infeasible_action_dwell_fraction",
            "power_flow_failure_steps",
        ]
        summary_df = summary_df[[col for col in ordered_columns if col in summary_df.columns]]
    summary_csv = output_dir / "summary.csv"
    summary_json = output_dir / "summary.json"
    summary_df.to_csv(summary_csv, index=False)
    summary_json.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")

    print("\n=== Short Cross-Fidelity Summary ===")
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary CSV: {summary_csv}")
    print(f"Saved summary JSON: {summary_json}")
    print(f"Saved trajectories: {trajectories_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
