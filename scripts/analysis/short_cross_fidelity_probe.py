#!/usr/bin/env python3
"""Run short train-test mismatch probes on the network-first microgrid cases."""
# Ref: docs/spec/task.md (Task-ID: SPEC-FIDELITY-MISMATCH-001)

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from microgrid_sim.cases import CIGREEuropeanLVConfig, IEEE33Config
from microgrid_sim.envs.network_microgrid import NetworkMicrogridEnv
from microgrid_sim.envs.wrappers import ContinuousActionRegularizationWrapper
from microgrid_sim.rl_utils import create_agent


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


def build_env(case_key: str, battery_model: str, days: int, seed: int, regime: str, args: argparse.Namespace | None = None):
    reward_profile = str(getattr(args, "reward_profile", "network")) if args is not None else "network"
    env = NetworkMicrogridEnv(
        build_config(
            case_key=case_key,
            battery_model=battery_model,
            days=days,
            seed=seed,
            regime=regime,
            reward_profile=reward_profile,
        )
    )
    if args is not None and action_regularization_enabled(args):
        env = ContinuousActionRegularizationWrapper(env=env, **action_regularization_config(args))
    return env


def _dwell_fraction(mask: pd.Series | list[bool]) -> float:
    series = pd.Series(mask, dtype=float)
    if series.empty:
        return 0.0
    return float(series.mean())


def train_short_agent(case_key: str, train_model: str, regime: str, args: argparse.Namespace):
    schedule = resolve_training_schedule(train_model=train_model, args=args)
    stages = [str(stage) for stage in schedule["stages"]]
    stage_steps = [int(value) for value in schedule["stage_steps"]]
    stage_learning_rates = [float(value) for value in schedule["stage_learning_rates"]]
    base_learning_rate = float(getattr(args, "learning_rate", 3e-4))
    env = build_env(case_key=case_key, battery_model=stages[0], days=args.days, seed=args.seed, regime=regime, args=args)
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
        )
        if stage_steps[0] > 0:
            _set_agent_learning_rate(agent, stage_learning_rates[0])
            agent.learn(total_timesteps=int(stage_steps[0]), progress_bar=False)
        for stage_model, steps, stage_learning_rate in zip(stages[1:], stage_steps[1:], stage_learning_rates[1:]):
            if steps <= 0:
                continue
            next_env = build_env(case_key=case_key, battery_model=stage_model, days=args.days, seed=args.seed, regime=regime, args=args)
            try:
                agent.set_env(next_env)
                _set_agent_learning_rate(agent, stage_learning_rate)
                agent.learn(total_timesteps=int(steps), progress_bar=False, reset_num_timesteps=False)
            finally:
                next_env.close()
        return agent, schedule
    finally:
        env.close()


def evaluate_agent(agent, case_key: str, test_model: str, regime: str, args: argparse.Namespace) -> tuple[dict, pd.DataFrame]:
    env = build_env(case_key=case_key, battery_model=test_model, days=args.days, seed=args.seed, regime=regime, args=args)
    rows: list[dict[str, float | int | str]] = []
    try:
        soc_min = float(env.unwrapped.config.battery_params.soc_min)
        soc_max = float(env.unwrapped.config.battery_params.soc_max)
        soc_tol = 1e-4
        infeasible_gap_tol = 1e-6
        obs, info = env.reset()
        total_reward = 0.0
        for step in range(int(args.eval_steps)):
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            rows.append(
                {
                    "step": int(step),
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
        return summary, trajectory
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
                    agent, train_schedule = train_short_agent(case_key=case_key, train_model=train_model, regime=regime, args=run_args)
                    for test_model in test_models:
                        print(f"[eval] case={case_key} regime={regime} train={train_model} test={test_model} seed={seed}")
                        summary, trajectory = evaluate_agent(agent, case_key=case_key, test_model=test_model, regime=regime, args=run_args)
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
                            "action_smoothing_coef": float(args.action_smoothing_coef),
                            "action_max_delta": float(args.action_max_delta),
                            "action_rate_penalty": float(args.action_rate_penalty),
                            "battery_feasibility_aware": int(bool(args.battery_feasibility_aware)),
                            "battery_infeasible_penalty": float(args.battery_infeasible_penalty),
                            "symmetric_battery_action": int(bool(args.symmetric_battery_action)),
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
            "action_smoothing_coef",
            "action_max_delta",
            "action_rate_penalty",
            "battery_feasibility_aware",
            "battery_infeasible_penalty",
            "symmetric_battery_action",
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
