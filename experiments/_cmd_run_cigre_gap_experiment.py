"""Focused MG-CIGRE PBM-vs-EBM experiment with year-split data."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
import pandas as pd
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from microgrid_sim.cases import CIGREConfig
from microgrid_sim.envs import CIGREMicrogridEnv, ContinuousActionRegularizationWrapper, DiscreteActionWrapper
from microgrid_sim.rl_utils import (
    SUPPORTED_AGENT_NAMES,
    canonicalize_agent_name,
    create_agent,
    load_agent,
    replay_buffer_size_for,
)

CIGRE_DQN_ACTION_BINS = 21
CIGRE_VALIDATION_SELECTION_DAYS = 30
CIGRE_VALIDATION_INTERVAL_STEPS = 5_000
CIGRE_VALIDATION_START_HOURS = (0, 91 * 24, 182 * 24, 273 * 24)


def get_device(force_cpu: bool = False) -> str:
    import torch

    if not force_cpu and torch.cuda.is_available() and int(torch.cuda.device_count()) > 0:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return "cuda"
    print("Using CPU")
    return "cpu"


def parse_curriculum_days(raw: str, fallback_days: int) -> list[int]:
    values = [int(part.strip()) for part in str(raw).split(",") if part.strip()]
    cleaned = [day for day in values if day > 0]
    return cleaned or [int(fallback_days)]


def parse_int_list(raw: str | None) -> list[int]:
    if raw is None:
        return []
    values: list[int] = []
    for part in str(raw).split(","):
        token = part.strip()
        if not token:
            continue
        values.append(int(token))
    return values


def parse_hidden_sizes(raw: str | None, fallback: Sequence[int] = (256, 128, 64)) -> list[int]:
    values = [int(size) for size in parse_int_list(raw) if int(size) > 0]
    return values or [int(size) for size in fallback if int(size) > 0]


def _dedupe_ints(values: Sequence[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for val in values:
        x = int(val)
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def allocate_phase_steps(total_steps: int, phase_days: Sequence[int]) -> list[int]:
    if len(phase_days) <= 1:
        return [int(total_steps)]
    weights = np.arange(1, len(phase_days) + 1, dtype=float)
    weights /= np.sum(weights)
    steps = np.floor(total_steps * weights).astype(int)
    remainder = int(total_steps - int(np.sum(steps)))
    steps[-1] += remainder
    return [max(int(step), 1) for step in steps]


def scale_battery_params(base_params, power_scale: float, energy_scale: float):
    scaled_power = max(float(power_scale), 1e-6)
    scaled_energy = max(float(energy_scale), 1e-6)
    nominal_energy_kwh = base_params.nominal_energy_wh / 1000.0 * scaled_energy
    return replace(
        base_params,
        nominal_energy_kwh=nominal_energy_kwh,
        p_charge_max=base_params.p_charge_max * scaled_power,
        p_discharge_max=base_params.p_discharge_max * scaled_power,
    )


def build_agent_hyperparams(args: argparse.Namespace) -> dict:
    on_policy_rollout_steps = max(int(getattr(args, "on_policy_rollout_steps", 2048)), 1)
    on_policy_batch_size = max(int(getattr(args, "on_policy_batch_size", 128)), 1)
    return {
        "net_arch": parse_hidden_sizes(getattr(args, "policy_net_arch", None)),
        "learning_rate": float(getattr(args, "rl_learning_rate", 3e-4)),
        "learning_starts": max(int(getattr(args, "rl_learning_starts", 1000)), 0),
        "off_policy_batch_size": max(int(getattr(args, "rl_batch_size", 384)), 1),
        "ppo_batch_size": on_policy_batch_size,
        "ppo_n_steps": on_policy_rollout_steps,
        "trpo_n_steps": max(int(getattr(args, "trpo_rollout_steps", on_policy_rollout_steps)), 1),
        "gamma": float(getattr(args, "rl_gamma", 0.985)),
        "tau": float(getattr(args, "rl_tau", 0.003)),
        "sac_ent_coef": str(getattr(args, "sac_ent_coef", "auto_0.02")),
        "sac_target_entropy_scale": float(getattr(args, "sac_target_entropy_scale", 0.35)),
        "td3_action_noise_sigma": max(float(getattr(args, "td3_action_noise_sigma", 0.10)), 0.0),
        "td3_policy_delay": max(int(getattr(args, "td3_policy_delay", 2)), 1),
        "td3_target_policy_noise": max(float(getattr(args, "td3_target_policy_noise", 0.2)), 0.0),
        "td3_target_noise_clip": max(float(getattr(args, "td3_target_noise_clip", 0.5)), 0.0),
        "ddpg_action_noise_sigma": max(
            float(getattr(args, "ddpg_action_noise_sigma", getattr(args, "td3_action_noise_sigma", 0.10))),
            0.0,
        ),
        "d4pg_action_noise_sigma": max(
            float(getattr(args, "d4pg_action_noise_sigma", getattr(args, "ddpg_action_noise_sigma", 0.10))),
            0.0,
        ),
        "d4pg_learning_starts": max(int(getattr(args, "d4pg_learning_starts", 512)), 0),
        "d4pg_batch_size": max(int(getattr(args, "d4pg_batch_size", 256)), 1),
        "d4pg_collect_n_sample": max(int(getattr(args, "d4pg_collect_n_sample", 32)), 1),
        "d4pg_update_per_collect": max(int(getattr(args, "d4pg_update_per_collect", 2)), 1),
        "d4pg_n_step": max(int(getattr(args, "d4pg_n_step", 3)), 1),
        "d4pg_n_atom": max(int(getattr(args, "d4pg_n_atom", 51)), 2),
        "d4pg_v_min": float(getattr(args, "d4pg_v_min", -12000.0)),
        "d4pg_v_max": float(getattr(args, "d4pg_v_max", 50.0)),
        "trpo_target_kl": max(float(getattr(args, "trpo_target_kl", 0.01)), 1e-6),
        "trpo_cg_damping": max(float(getattr(args, "trpo_cg_damping", 0.1)), 0.0),
    }


def build_action_regularization(args: argparse.Namespace) -> dict:
    smoothing_coef = float(np.clip(float(getattr(args, "action_smoothing_coef", 0.0)), 0.0, 0.995))
    max_delta = max(float(getattr(args, "action_max_delta", 0.0)), 0.0)
    rate_penalty = max(float(getattr(args, "action_rate_penalty", 0.0)), 0.0)
    symmetric_battery_action = bool(getattr(args, "enable_symmetric_battery_action", False))
    enabled = bool(
        symmetric_battery_action
        or smoothing_coef > 1e-9
        or max_delta > 1e-9
        or rate_penalty > 1e-12
    )
    return {
        "enabled": enabled,
        "smoothing_coef": smoothing_coef,
        "max_delta": max_delta,
        "rate_penalty": rate_penalty,
        "symmetric_battery_action": symmetric_battery_action,
    }


def make_optimistic_ebm_params(
    base_params,
    soc_min: float,
    soc_max: float,
    power_scale: float,
    efficiency: float,
):
    optimistic_soc_min = max(0.0, min(float(soc_min), 0.98))
    optimistic_soc_max = min(max(float(soc_max), optimistic_soc_min + 1e-6), 1.0)
    optimistic_soc_init = min(max(0.50, optimistic_soc_min), optimistic_soc_max)
    optimistic_power_scale = max(float(power_scale), 1.0)
    optimistic_efficiency = min(max(float(efficiency), 0.0), 1.0)
    return replace(
        base_params,
        soc_min=optimistic_soc_min,
        soc_max=optimistic_soc_max,
        soc_init=optimistic_soc_init,
        p_charge_max=base_params.p_charge_max * optimistic_power_scale,
        p_discharge_max=base_params.p_discharge_max * optimistic_power_scale,
        eta_charge=max(base_params.eta_charge, optimistic_efficiency),
        eta_discharge=max(base_params.eta_discharge, optimistic_efficiency),
        thermal_dynamics_enabled=False,
    )


def build_config(
    battery_model: str,
    simulation_days: int,
    seed: int,
    data_dir: str | None,
    data_year: int,
    random_episode_start: bool,
    episode_start_hour: int,
    reward_mode: str,
    component_commitment_enabled: bool,
    include_component_cost_in_objective: bool,
    random_initial_soc: bool,
    initial_soc_min: float,
    initial_soc_max: float,
    price_spread_multiplier: float,
    peak_import_penalty_per_kw: float,
    peak_import_threshold_kw: float,
    midday_pv_boost_multiplier: float,
    evening_load_boost_multiplier: float,
    stress_episode_sampling: bool,
    stress_sampling_strength: float,
    battery_power_scale: float,
    battery_energy_scale: float,
    optimistic_ebm_training: bool,
    optimistic_ebm_soc_min: float,
    optimistic_ebm_soc_max: float,
    optimistic_ebm_power_scale: float,
    optimistic_ebm_efficiency: float,
    optimistic_ebm_soc_penalty_scale: float,
) -> CIGREConfig:
    config = CIGREConfig(simulation_days=simulation_days, seed=seed)
    config.data_dir = data_dir
    config.data_year = int(data_year)
    config.random_episode_start = bool(random_episode_start)
    config.episode_start_hour = int(episode_start_hour)
    config.reward_mode = reward_mode
    config.component_commitment_enabled = bool(component_commitment_enabled)
    config.include_component_cost_in_objective = bool(include_component_cost_in_objective)
    config.random_initial_soc = bool(random_initial_soc)
    config.initial_soc_min = float(initial_soc_min)
    config.initial_soc_max = float(initial_soc_max)
    config.tou_price_spread_multiplier = float(price_spread_multiplier)
    config.peak_import_penalty_per_kw = float(peak_import_penalty_per_kw)
    config.peak_import_penalty_threshold_w = float(peak_import_threshold_kw) * 1000.0
    config.midday_pv_boost_multiplier = float(midday_pv_boost_multiplier)
    config.evening_load_boost_multiplier = float(evening_load_boost_multiplier)
    config.stress_episode_sampling = bool(stress_episode_sampling)
    config.stress_sampling_strength = float(stress_sampling_strength)
    config.battery_params = scale_battery_params(
        config.battery_params,
        power_scale=battery_power_scale,
        energy_scale=battery_energy_scale,
    )
    if battery_model == "simple" and optimistic_ebm_training:
        config.battery_params = make_optimistic_ebm_params(
            config.battery_params,
            soc_min=optimistic_ebm_soc_min,
            soc_max=optimistic_ebm_soc_max,
            power_scale=optimistic_ebm_power_scale,
            efficiency=optimistic_ebm_efficiency,
        )
        config.reward = replace(
            config.reward,
            w_soc_violation=config.reward.w_soc_violation * max(float(optimistic_ebm_soc_penalty_scale), 0.0),
        )
    return config


def create_env(
    config: CIGREConfig,
    battery_model: str,
    monitor_file: Path | None = None,
    agent_name: str = "sac",
    dqn_action_bins: int = CIGRE_DQN_ACTION_BINS,
    action_regularization: dict | None = None,
):
    def _factory():
        env = CIGREMicrogridEnv(config=copy.deepcopy(config), battery_model=battery_model)
        normalized_agent_name = canonicalize_agent_name(agent_name)
        if normalized_agent_name == "dqn":
            env = DiscreteActionWrapper(env, action_bins=dqn_action_bins)
        else:
            regularization = dict(action_regularization or {})
            if regularization.get("enabled", False):
                env = ContinuousActionRegularizationWrapper(
                    env,
                    smoothing_coef=float(regularization.get("smoothing_coef", 0.0)),
                    max_delta=float(regularization.get("max_delta", 0.0)),
                    rate_penalty=float(regularization.get("rate_penalty", 0.0)),
                    symmetric_battery_action=bool(regularization.get("symmetric_battery_action", False)),
                )
        env.reset(seed=int(config.seed))
        env.action_space.seed(int(config.seed))
        env.observation_space.seed(int(config.seed))
        filename = None
        if monitor_file is not None:
            monitor_file.parent.mkdir(parents=True, exist_ok=True)
            filename = str(monitor_file)
        return Monitor(env, filename=filename)

    return DummyVecEnv([_factory])


def train_agent(
    agent_name: str,
    battery_model: str,
    steps: int,
    train_days: int,
    seed: int,
    data_dir: str | None,
    data_year: int,
    random_episode_start: bool,
    episode_start_hour: int,
    reward_mode: str,
    component_commitment_enabled: bool,
    include_component_cost_in_objective: bool,
    random_initial_soc: bool,
    initial_soc_min: float,
    initial_soc_max: float,
    price_spread_multiplier: float,
    peak_import_penalty_per_kw: float,
    peak_import_threshold_kw: float,
    midday_pv_boost_multiplier: float,
    evening_load_boost_multiplier: float,
    stress_episode_sampling: bool,
    stress_sampling_strength: float,
    curriculum_days: Sequence[int],
    battery_power_scale: float,
    battery_energy_scale: float,
    optimistic_ebm_training: bool,
    optimistic_ebm_soc_min: float,
    optimistic_ebm_soc_max: float,
    optimistic_ebm_power_scale: float,
    optimistic_ebm_efficiency: float,
    optimistic_ebm_soc_penalty_scale: float,
    model_path: Path,
    dqn_action_bins: int = CIGRE_DQN_ACTION_BINS,
    force_cpu: bool = False,
    validation_selection_enabled: bool = False,
    validation_days: int = CIGRE_VALIDATION_SELECTION_DAYS,
    validation_start_hours: Sequence[int] = (),
    validation_interval_steps: int = CIGRE_VALIDATION_INTERVAL_STEPS,
    validation_eval_battery_model: str = "thevenin",
    agent_hyperparams: dict | None = None,
    action_regularization: dict | None = None,
) -> tuple[BaseAlgorithm, dict]:
    agent_name = canonicalize_agent_name(agent_name)
    device = get_device(force_cpu=bool(force_cpu))
    phase_days = [int(day) for day in curriculum_days if int(day) > 0]
    if not phase_days:
        phase_days = [int(train_days)]
    phase_steps = allocate_phase_steps(int(steps), phase_days)
    learning_starts = max(int((agent_hyperparams or {}).get("learning_starts", 1000)), 0)
    validation_selection_enabled = bool(validation_selection_enabled)
    validation_days = max(int(validation_days), 1)
    validation_interval_steps = max(int(validation_interval_steps), 1)
    validation_hours = [int(hour) for hour in validation_start_hours if int(hour) >= 0] or [
        int(hour) for hour in CIGRE_VALIDATION_START_HOURS
    ]
    validation_history: list[dict] = []
    best_validation_record: dict | None = None
    best_validation_mean_cost = float("inf")
    total_steps_trained = 0
    agent: BaseAlgorithm | None = None
    best_model_path = model_path.parent / f"{model_path.stem}_best.zip"
    last_model_path = model_path.parent / f"{model_path.stem}_last.zip"
    first_learn = True

    for phase_index, (phase_day_count, phase_step_count) in enumerate(zip(phase_days, phase_steps), start=1):
        phase_episode_steps = int(phase_day_count) * 24
        if int(phase_step_count) < phase_episode_steps:
            print(
                f"  Warning: phase {phase_index} gets only {int(phase_step_count):,} training steps "
                f"for a {phase_day_count}d episode ({phase_episode_steps:,} steps). "
                "Short-budget validation and monitor statistics may be uninformative."
            )
        if learning_starts > 0 and int(phase_step_count) <= learning_starts:
            print(
                f"  Warning: phase {phase_index} gets {int(phase_step_count):,} steps, "
                f"which does not exceed learning_starts={learning_starts:,}. "
                "Off-policy agents may not perform any parameter updates in this phase."
            )
        phase_random_episode_start = bool(random_episode_start or phase_day_count < train_days)
        phase_config = build_config(
            battery_model=battery_model,
            simulation_days=phase_day_count,
            seed=seed,
            data_dir=data_dir,
            data_year=data_year,
            random_episode_start=phase_random_episode_start,
            episode_start_hour=episode_start_hour,
            reward_mode=reward_mode,
            component_commitment_enabled=component_commitment_enabled,
            include_component_cost_in_objective=include_component_cost_in_objective,
            random_initial_soc=random_initial_soc,
            initial_soc_min=initial_soc_min,
            initial_soc_max=initial_soc_max,
            price_spread_multiplier=price_spread_multiplier,
            peak_import_penalty_per_kw=peak_import_penalty_per_kw,
            peak_import_threshold_kw=peak_import_threshold_kw,
            midday_pv_boost_multiplier=midday_pv_boost_multiplier,
            evening_load_boost_multiplier=evening_load_boost_multiplier,
            stress_episode_sampling=stress_episode_sampling,
            stress_sampling_strength=stress_sampling_strength,
            battery_power_scale=battery_power_scale,
            battery_energy_scale=battery_energy_scale,
            optimistic_ebm_training=optimistic_ebm_training,
            optimistic_ebm_soc_min=optimistic_ebm_soc_min,
            optimistic_ebm_soc_max=optimistic_ebm_soc_max,
            optimistic_ebm_power_scale=optimistic_ebm_power_scale,
            optimistic_ebm_efficiency=optimistic_ebm_efficiency,
            optimistic_ebm_soc_penalty_scale=optimistic_ebm_soc_penalty_scale,
        )
        phase_env = create_env(
            config=phase_config,
            battery_model=battery_model,
            monitor_file=model_path.parent / f"{model_path.stem}_phase{phase_index}.csv",
            agent_name=agent_name,
            dqn_action_bins=int(dqn_action_bins),
            action_regularization=action_regularization,
        )
        if agent is None:
            obs_dim = int(np.prod(getattr(phase_env.observation_space, "shape", (0,)) or (0,)))
            if hasattr(phase_env.action_space, "n"):
                action_dim = int(getattr(phase_env.action_space, "n", 0))
            else:
                action_dim = int(np.prod(getattr(phase_env.action_space, "shape", (0,)) or (0,)))
            replay_buffer_size = int(replay_buffer_size_for(agent_name, steps))
            print(f"  Replay buffer: {replay_buffer_size:,} (obs_dim={obs_dim}, action_dim={action_dim})")
            agent = create_agent(
                agent_name=agent_name,
                env=phase_env,
                total_steps=steps,
                seed=seed,
                device=device,
                agent_hyperparams=agent_hyperparams,
            )
        else:
            agent.set_env(phase_env)
        remaining_phase_steps = int(phase_step_count)
        phase_steps_trained = 0
        phase_chunk_index = 0
        while remaining_phase_steps > 0:
            chunk_steps = remaining_phase_steps if not validation_selection_enabled else min(remaining_phase_steps, validation_interval_steps)
            phase_chunk_index += 1
            print(
                f"Training {agent_name}:{battery_model} phase {phase_index}/{len(phase_days)} "
                f"chunk {phase_chunk_index} for {chunk_steps:,}/{phase_step_count:,} steps "
                f"on {phase_day_count}d windows (year {data_year}) ..."
            )
            agent.learn(total_timesteps=chunk_steps, progress_bar=True, reset_num_timesteps=first_learn)
            first_learn = False
            remaining_phase_steps -= int(chunk_steps)
            phase_steps_trained += int(chunk_steps)
            total_steps_trained += int(chunk_steps)
            if validation_selection_enabled:
                validation_record = evaluate_validation_windows(
                    agent=agent,
                    agent_name=agent_name,
                    validation_label=f"{model_path.stem}_phase{phase_index}_chunk{phase_chunk_index}",
                    validation_days=validation_days,
                    validation_start_hours=validation_hours,
                    seed=seed,
                    data_dir=data_dir,
                    data_year=data_year,
                    reward_mode=reward_mode,
                    validation_eval_battery_model=validation_eval_battery_model,
                    component_commitment_enabled=component_commitment_enabled,
                    include_component_cost_in_objective=include_component_cost_in_objective,
                    price_spread_multiplier=price_spread_multiplier,
                    peak_import_penalty_per_kw=peak_import_penalty_per_kw,
                    peak_import_threshold_kw=peak_import_threshold_kw,
                    midday_pv_boost_multiplier=midday_pv_boost_multiplier,
                    evening_load_boost_multiplier=evening_load_boost_multiplier,
                    battery_power_scale=battery_power_scale,
                    battery_energy_scale=battery_energy_scale,
                    dqn_action_bins=dqn_action_bins,
                    action_regularization=action_regularization,
                )
                validation_record.update(
                    {
                        "phase_index": int(phase_index),
                        "phase_day_count": int(phase_day_count),
                        "phase_chunk_index": int(phase_chunk_index),
                        "phase_steps_trained": int(phase_steps_trained),
                        "total_steps_trained": int(total_steps_trained),
                    }
                )
                validation_history.append(validation_record)
                print(
                    f"  Validation ({validation_eval_battery_model}, {validation_days}d x {len(validation_hours)} windows): "
                    f"mean={validation_record['validation_mean_cost']:.2f}, "
                    f"worst={validation_record['validation_worst_cost']:.2f}, "
                    f"failed_windows={validation_record.get('validation_failed_windows', 0)}"
                )
                if float(validation_record["validation_mean_cost"]) < float(best_validation_mean_cost) - 1e-9:
                    best_validation_mean_cost = float(validation_record["validation_mean_cost"])
                    best_validation_record = dict(validation_record)
                    agent.save(str(best_model_path))
                    print(f"  New best checkpoint saved: mean cost {best_validation_mean_cost:.2f}")
        phase_env.close()

    model_path.parent.mkdir(parents=True, exist_ok=True)
    if validation_selection_enabled:
        agent.save(str(last_model_path))
        if best_validation_record is not None and best_model_path.exists():
            agent = load_agent(agent_name, str(best_model_path), device=device)
        if validation_history:
            pd.DataFrame(validation_history).to_csv(model_path.parent / f"{model_path.stem}_validation.csv", index=False)
    agent.save(str(model_path))
    training_meta = {
        "validation_selection_enabled": bool(validation_selection_enabled),
        "validation_days": int(validation_days),
        "validation_interval_steps": int(validation_interval_steps),
        "validation_start_hours": [int(hour) for hour in validation_hours],
        "validation_eval_battery_model": str(validation_eval_battery_model),
        "selected_checkpoint": best_validation_record,
        "validation_history_rows": int(len(validation_history)),
        "best_model_path": str(best_model_path) if best_validation_record is not None else "",
        "last_model_path": str(last_model_path) if validation_selection_enabled else "",
    }
    return agent, training_meta


def evaluate_rollout(
    agent: BaseAlgorithm,
    agent_name: str,
    label: str,
    eval_days: int,
    seed: int,
    data_dir: str | None,
    data_year: int,
    episode_start_hour: int,
    reward_mode: str,
    env_battery_model: str,
    component_commitment_enabled: bool,
    include_component_cost_in_objective: bool,
    price_spread_multiplier: float,
    peak_import_penalty_per_kw: float,
    peak_import_threshold_kw: float,
    midday_pv_boost_multiplier: float,
    evening_load_boost_multiplier: float,
    battery_power_scale: float,
    battery_energy_scale: float,
    dqn_action_bins: int = CIGRE_DQN_ACTION_BINS,
    action_regularization: dict | None = None,
    allow_prediction_failure: bool = False,
) -> dict:
    agent_name = canonicalize_agent_name(agent_name)
    config = build_config(
        battery_model=env_battery_model,
        simulation_days=eval_days,
        seed=seed,
        data_dir=data_dir,
        data_year=data_year,
        random_episode_start=False,
        episode_start_hour=episode_start_hour,
        reward_mode=reward_mode,
        component_commitment_enabled=component_commitment_enabled,
        include_component_cost_in_objective=include_component_cost_in_objective,
        random_initial_soc=False,
        initial_soc_min=0.0,
        initial_soc_max=1.0,
        price_spread_multiplier=price_spread_multiplier,
        peak_import_penalty_per_kw=peak_import_penalty_per_kw,
        peak_import_threshold_kw=peak_import_threshold_kw,
        midday_pv_boost_multiplier=midday_pv_boost_multiplier,
        evening_load_boost_multiplier=evening_load_boost_multiplier,
        stress_episode_sampling=False,
        stress_sampling_strength=0.0,
        battery_power_scale=battery_power_scale,
        battery_energy_scale=battery_energy_scale,
        optimistic_ebm_training=False,
        optimistic_ebm_soc_min=0.0,
        optimistic_ebm_soc_max=1.0,
        optimistic_ebm_power_scale=1.0,
        optimistic_ebm_efficiency=1.0,
        optimistic_ebm_soc_penalty_scale=1.0,
    )
    env = CIGREMicrogridEnv(config=config, battery_model=env_battery_model)
    if agent_name == "dqn":
        env = DiscreteActionWrapper(env, action_bins=int(dqn_action_bins))
    else:
        regularization = dict(action_regularization or {})
        if regularization.get("enabled", False):
            env = ContinuousActionRegularizationWrapper(
                env,
                smoothing_coef=float(regularization.get("smoothing_coef", 0.0)),
                max_delta=float(regularization.get("max_delta", 0.0)),
                rate_penalty=float(regularization.get("rate_penalty", 0.0)),
                symmetric_battery_action=bool(regularization.get("symmetric_battery_action", False)),
            )
    obs, _info = env.reset()

    steps: list[int] = []
    soc_hist: list[float] = []
    soh_hist: list[float] = []
    cost_hist: list[float] = []
    import_cost_hist: list[float] = []
    peak_penalty_hist: list[float] = []
    p_grid_kw_hist: list[float] = []
    grid_import_kw_hist: list[float] = []
    p_cmd_kw_hist: list[float] = []
    p_actual_kw_hist: list[float] = []
    p_cmd_abs_hist: list[float] = []
    p_actual_abs_hist: list[float] = []
    clip_hist: list[float] = []

    evaluation_failed = False
    failure_reason = ""

    for step in range(eval_days * 24):
        try:
            action, _ = agent.predict(obs, deterministic=True)
        except Exception as exc:
            if not allow_prediction_failure:
                env.close()
                raise
            evaluation_failed = True
            failure_reason = f"{exc.__class__.__name__}: {exc}"
            break
        action_array = np.asarray(action, dtype=float)
        if action_array.size and not np.all(np.isfinite(action_array)):
            if not allow_prediction_failure:
                env.close()
                raise ValueError(f"Non-finite action predicted during rollout '{label}'.")
            evaluation_failed = True
            failure_reason = "Non-finite action predicted."
            break
        obs, reward, terminated, truncated, info = env.step(action)
        del reward
        steps.append(step)
        soc_hist.append(float(info.get("soc", 0.0)))
        soh_hist.append(float(info.get("soh", 1.0)))
        cost_hist.append(float(info.get("cumulative_cost", 0.0)))
        import_cost_hist.append(float(info.get("import_cost", 0.0)))
        peak_penalty_hist.append(float(info.get("peak_import_penalty", 0.0)))
        p_grid_kw = float(info.get("p_grid", 0.0)) / 1000.0
        p_cmd_kw = float(info.get("p_cmd", 0.0)) / 1000.0
        p_actual_kw = float(info.get("p_actual", 0.0)) / 1000.0
        p_grid_kw_hist.append(p_grid_kw)
        grid_import_kw_hist.append(max(p_grid_kw, 0.0))
        p_cmd_kw_hist.append(p_cmd_kw)
        p_actual_kw_hist.append(p_actual_kw)
        p_cmd_abs_hist.append(abs(p_cmd_kw))
        p_actual_abs_hist.append(abs(p_actual_kw))
        clip_hist.append(1.0 if abs(p_actual_kw - p_cmd_kw) > 1e-6 else 0.0)
        if terminated or truncated:
            break

    env.close()
    return {
        "label": label,
        "eval_year": int(data_year),
        "evaluation_failed": bool(evaluation_failed),
        "failure_reason": failure_reason,
        "total_cost": float("inf") if evaluation_failed else float(cost_hist[-1] if cost_hist else 0.0),
        "final_soh": float(soh_hist[-1] if soh_hist else 1.0),
        "soc_min": float(min(soc_hist) if soc_hist else 0.0),
        "soc_max": float(max(soc_hist) if soc_hist else 0.0),
        "import_cost_sum": float(sum(import_cost_hist)),
        "peak_import_penalty_sum": float(sum(peak_penalty_hist)),
        "max_grid_import_kw": float(max(grid_import_kw_hist, default=0.0)),
        "p_cmd_abs_p95_kw": float(np.percentile(p_cmd_abs_hist, 95)) if p_cmd_abs_hist else 0.0,
        "p_actual_abs_p95_kw": float(np.percentile(p_actual_abs_hist, 95)) if p_actual_abs_hist else 0.0,
        "clip_ratio": float(np.mean(clip_hist)) if clip_hist else 0.0,
        "steps": steps,
        "soc": soc_hist,
        "soh": soh_hist,
        "cost": cost_hist,
        "import_cost": import_cost_hist,
        "peak_import_penalty": peak_penalty_hist,
        "grid_power_kw": p_grid_kw_hist,
        "grid_import_kw": grid_import_kw_hist,
        "battery_command_kw": p_cmd_kw_hist,
        "battery_actual_kw": p_actual_kw_hist,
    }


def evaluate_validation_windows(
    agent: BaseAlgorithm,
    agent_name: str,
    validation_label: str,
    validation_days: int,
    validation_start_hours: Sequence[int],
    seed: int,
    data_dir: str | None,
    data_year: int,
    reward_mode: str,
    validation_eval_battery_model: str,
    component_commitment_enabled: bool,
    include_component_cost_in_objective: bool,
    price_spread_multiplier: float,
    peak_import_penalty_per_kw: float,
    peak_import_threshold_kw: float,
    midday_pv_boost_multiplier: float,
    evening_load_boost_multiplier: float,
    battery_power_scale: float,
    battery_energy_scale: float,
    dqn_action_bins: int = CIGRE_DQN_ACTION_BINS,
    action_regularization: dict | None = None,
) -> dict:
    validation_rows: list[dict] = []
    for start_hour in validation_start_hours:
        result = evaluate_rollout(
            agent=agent,
            agent_name=agent_name,
            label=f"{validation_label}_sh{int(start_hour)}",
            eval_days=validation_days,
            seed=seed,
            data_dir=data_dir,
            data_year=data_year,
            episode_start_hour=int(start_hour),
            reward_mode=reward_mode,
            env_battery_model=validation_eval_battery_model,
            component_commitment_enabled=component_commitment_enabled,
            include_component_cost_in_objective=include_component_cost_in_objective,
            price_spread_multiplier=price_spread_multiplier,
            peak_import_penalty_per_kw=peak_import_penalty_per_kw,
            peak_import_threshold_kw=peak_import_threshold_kw,
            midday_pv_boost_multiplier=midday_pv_boost_multiplier,
            evening_load_boost_multiplier=evening_load_boost_multiplier,
            battery_power_scale=battery_power_scale,
            battery_energy_scale=battery_energy_scale,
            dqn_action_bins=dqn_action_bins,
            action_regularization=action_regularization,
            allow_prediction_failure=True,
        )
        validation_rows.append(
            {
                "start_hour": int(start_hour),
                "total_cost": float(result["total_cost"]),
                "max_grid_import_kw": float(result.get("max_grid_import_kw", 0.0)),
                "p_actual_abs_p95_kw": float(result.get("p_actual_abs_p95_kw", 0.0)),
                "failed": bool(result.get("evaluation_failed", False)),
                "failure_reason": str(result.get("failure_reason", "")),
            }
        )
    costs = [float(row["total_cost"]) for row in validation_rows]
    return {
        "validation_mean_cost": float(np.mean(costs)) if costs else float("inf"),
        "validation_worst_cost": float(np.max(costs)) if costs else float("inf"),
        "validation_best_cost": float(np.min(costs)) if costs else float("inf"),
        "validation_failed_windows": int(sum(1 for row in validation_rows if bool(row.get("failed", False)))),
        "validation_window_costs": json.dumps(validation_rows, ensure_ascii=False),
    }


def evaluate_under_pbm(
    agent: BaseAlgorithm,
    agent_name: str,
    label: str,
    eval_days: int,
    seed: int,
    data_dir: str | None,
    data_year: int,
    episode_start_hour: int,
    reward_mode: str,
    component_commitment_enabled: bool,
    include_component_cost_in_objective: bool,
    price_spread_multiplier: float,
    peak_import_penalty_per_kw: float,
    peak_import_threshold_kw: float,
    midday_pv_boost_multiplier: float,
    evening_load_boost_multiplier: float,
    battery_power_scale: float,
    battery_energy_scale: float,
    dqn_action_bins: int = CIGRE_DQN_ACTION_BINS,
    action_regularization: dict | None = None,
) -> dict:
    return evaluate_rollout(
        agent=agent,
        agent_name=agent_name,
        label=label,
        eval_days=eval_days,
        seed=seed,
        data_dir=data_dir,
        data_year=data_year,
        episode_start_hour=episode_start_hour,
        reward_mode=reward_mode,
        env_battery_model="thevenin",
        component_commitment_enabled=component_commitment_enabled,
        include_component_cost_in_objective=include_component_cost_in_objective,
        price_spread_multiplier=price_spread_multiplier,
        peak_import_penalty_per_kw=peak_import_penalty_per_kw,
        peak_import_threshold_kw=peak_import_threshold_kw,
        midday_pv_boost_multiplier=midday_pv_boost_multiplier,
        evening_load_boost_multiplier=evening_load_boost_multiplier,
        battery_power_scale=battery_power_scale,
        battery_energy_scale=battery_energy_scale,
        dqn_action_bins=dqn_action_bins,
        action_regularization=action_regularization,
    )


def evaluate_idle_baseline(
    eval_days: int,
    seed: int,
    data_dir: str | None,
    data_year: int,
    episode_start_hour: int,
    reward_mode: str,
    component_commitment_enabled: bool,
    include_component_cost_in_objective: bool,
    price_spread_multiplier: float,
    peak_import_penalty_per_kw: float,
    peak_import_threshold_kw: float,
    midday_pv_boost_multiplier: float,
    evening_load_boost_multiplier: float,
    battery_power_scale: float,
    battery_energy_scale: float,
) -> dict:
    config = build_config(
        battery_model="thevenin",
        simulation_days=eval_days,
        seed=seed,
        data_dir=data_dir,
        data_year=data_year,
        random_episode_start=False,
        episode_start_hour=episode_start_hour,
        reward_mode=reward_mode,
        component_commitment_enabled=component_commitment_enabled,
        include_component_cost_in_objective=include_component_cost_in_objective,
        random_initial_soc=False,
        initial_soc_min=0.0,
        initial_soc_max=1.0,
        price_spread_multiplier=price_spread_multiplier,
        peak_import_penalty_per_kw=peak_import_penalty_per_kw,
        peak_import_threshold_kw=peak_import_threshold_kw,
        midday_pv_boost_multiplier=midday_pv_boost_multiplier,
        evening_load_boost_multiplier=evening_load_boost_multiplier,
        stress_episode_sampling=False,
        stress_sampling_strength=0.0,
        battery_power_scale=battery_power_scale,
        battery_energy_scale=battery_energy_scale,
        optimistic_ebm_training=False,
        optimistic_ebm_soc_min=0.0,
        optimistic_ebm_soc_max=1.0,
        optimistic_ebm_power_scale=1.0,
        optimistic_ebm_efficiency=1.0,
        optimistic_ebm_soc_penalty_scale=1.0,
    )
    env = CIGREMicrogridEnv(config=config, battery_model="thevenin")
    obs, _info = env.reset()
    del obs

    total_cost = 0.0
    total_peak_penalty = 0.0
    max_grid_import_kw = 0.0
    for _step in range(eval_days * 24):
        _obs, _reward, terminated, truncated, info = env.step([0.0])
        total_cost = float(info.get("cumulative_cost", total_cost))
        total_peak_penalty += float(info.get("peak_import_penalty", 0.0))
        max_grid_import_kw = max(max_grid_import_kw, max(float(info.get("p_grid", 0.0)), 0.0) / 1000.0)
        if terminated or truncated:
            break

    env.close()
    return {
        "label": "idle_baseline",
        "total_cost": total_cost,
        "peak_import_penalty_sum": total_peak_penalty,
        "max_grid_import_kw": max_grid_import_kw,
    }


SUMMARY_DROP_COLUMNS = [
    "steps",
    "soc",
    "soh",
    "cost",
    "import_cost",
    "peak_import_penalty",
    "grid_power_kw",
    "grid_import_kw",
    "battery_command_kw",
    "battery_actual_kw",
]


def export_timeseries(result: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "Hour": result["steps"],
        "SOC": result["soc"],
        "SOH": result["soh"],
        "Cumulative_Cost": result["cost"],
    }
    if "import_cost" in result:
        payload["Import_Cost"] = result["import_cost"]
    if "peak_import_penalty" in result:
        payload["Peak_Import_Penalty"] = result["peak_import_penalty"]
    if "grid_power_kw" in result:
        payload["Grid_Power_kW"] = result["grid_power_kw"]
    if "grid_import_kw" in result:
        payload["Grid_Import_kW"] = result["grid_import_kw"]
    if "battery_command_kw" in result:
        payload["Battery_Command_kW"] = result["battery_command_kw"]
    if "battery_actual_kw" in result:
        payload["Battery_Actual_kW"] = result["battery_actual_kw"]
    pd.DataFrame(payload).to_csv(out_path, index=False)


def run_seed_experiment(
    args: argparse.Namespace,
    seed: int,
    output_dir: Path,
    models_dir: Path,
    curriculum_days: list[int],
    random_train_soc: bool,
    stress_episode_sampling: bool,
    optimistic_ebm_training: bool,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    agent_name = canonicalize_agent_name(getattr(args, "agent", "sac"))
    dqn_action_bins = int(getattr(args, "dqn_action_bins", CIGRE_DQN_ACTION_BINS))
    device = get_device(force_cpu=bool(getattr(args, "cpu", False)))
    agent_hyperparams = build_agent_hyperparams(args)
    action_regularization = build_action_regularization(args)
    validation_selection_enabled = bool(getattr(args, "validation_selection", False))
    validation_days = max(int(getattr(args, "validation_days", CIGRE_VALIDATION_SELECTION_DAYS)), 1)
    validation_interval_steps = max(
        int(getattr(args, "validation_interval_steps", CIGRE_VALIDATION_INTERVAL_STEPS)),
        1,
    )
    validation_eval_battery_model = str(getattr(args, "validation_eval_battery_model", "thevenin"))
    validation_start_hours = parse_int_list(getattr(args, "validation_start_hours", None)) or [
        int(hour) for hour in CIGRE_VALIDATION_START_HOURS
    ]
    suffix = f"seed{seed}_{args.steps}_train{args.train_year}_eval{args.eval_year}"
    replay_buffer_size = int(replay_buffer_size_for(agent_name, args.steps))
    pbm_model_path = models_dir / f"pbm_{agent_name}_{suffix}.zip"
    ebm_model_path = models_dir / f"ebm_{agent_name}_{suffix}.zip"

    if args.skip_train and pbm_model_path.exists() and ebm_model_path.exists():
        pbm_agent = load_agent(agent_name, str(pbm_model_path), device=device)
        ebm_agent = load_agent(agent_name, str(ebm_model_path), device=device)
        pbm_train_meta = {
            "validation_selection_enabled": bool(validation_selection_enabled),
            "validation_days": int(validation_days),
            "validation_interval_steps": int(validation_interval_steps),
            "validation_start_hours": [int(hour) for hour in validation_start_hours],
            "validation_eval_battery_model": str(validation_eval_battery_model),
            "selected_checkpoint": None,
            "validation_history_rows": 0,
            "best_model_path": "",
            "last_model_path": "",
        }
        ebm_train_meta = dict(pbm_train_meta)
    else:
        pbm_agent, pbm_train_meta = train_agent(
            agent_name=agent_name,
            battery_model="thevenin",
            steps=args.steps,
            train_days=args.train_days,
            seed=seed,
            data_dir=args.data_dir,
            data_year=args.train_year,
            random_episode_start=args.random_train_window,
            episode_start_hour=args.train_start_hour,
            reward_mode=args.reward_mode,
            component_commitment_enabled=args.enable_commitment,
            include_component_cost_in_objective=args.include_component_cost,
            random_initial_soc=random_train_soc,
            initial_soc_min=args.train_soc_min,
            initial_soc_max=args.train_soc_max,
            price_spread_multiplier=args.price_spread_multiplier,
            peak_import_penalty_per_kw=args.peak_import_penalty_per_kw,
            peak_import_threshold_kw=args.peak_import_threshold_kw,
            midday_pv_boost_multiplier=args.midday_pv_boost_multiplier,
            evening_load_boost_multiplier=args.evening_load_boost_multiplier,
            stress_episode_sampling=stress_episode_sampling,
            stress_sampling_strength=args.stress_sampling_strength,
            curriculum_days=curriculum_days,
            battery_power_scale=args.battery_power_scale,
            battery_energy_scale=args.battery_energy_scale,
            optimistic_ebm_training=False,
            optimistic_ebm_soc_min=args.optimistic_ebm_soc_min,
            optimistic_ebm_soc_max=args.optimistic_ebm_soc_max,
            optimistic_ebm_power_scale=args.optimistic_ebm_power_scale,
            optimistic_ebm_efficiency=args.optimistic_ebm_efficiency,
            optimistic_ebm_soc_penalty_scale=args.optimistic_ebm_soc_penalty_scale,
            model_path=pbm_model_path,
            dqn_action_bins=dqn_action_bins,
            force_cpu=bool(getattr(args, "cpu", False)),
            validation_selection_enabled=validation_selection_enabled,
            validation_days=validation_days,
            validation_start_hours=validation_start_hours,
            validation_interval_steps=validation_interval_steps,
            validation_eval_battery_model=validation_eval_battery_model,
            agent_hyperparams=agent_hyperparams,
            action_regularization=action_regularization,
        )
        ebm_agent, ebm_train_meta = train_agent(
            agent_name=agent_name,
            battery_model="simple",
            steps=args.steps,
            train_days=args.train_days,
            seed=seed,
            data_dir=args.data_dir,
            data_year=args.train_year,
            random_episode_start=args.random_train_window,
            episode_start_hour=args.train_start_hour,
            reward_mode=args.reward_mode,
            component_commitment_enabled=args.enable_commitment,
            include_component_cost_in_objective=args.include_component_cost,
            random_initial_soc=random_train_soc,
            initial_soc_min=args.train_soc_min,
            initial_soc_max=args.train_soc_max,
            price_spread_multiplier=args.price_spread_multiplier,
            peak_import_penalty_per_kw=args.peak_import_penalty_per_kw,
            peak_import_threshold_kw=args.peak_import_threshold_kw,
            midday_pv_boost_multiplier=args.midday_pv_boost_multiplier,
            evening_load_boost_multiplier=args.evening_load_boost_multiplier,
            stress_episode_sampling=stress_episode_sampling,
            stress_sampling_strength=args.stress_sampling_strength,
            curriculum_days=curriculum_days,
            battery_power_scale=args.battery_power_scale,
            battery_energy_scale=args.battery_energy_scale,
            optimistic_ebm_training=optimistic_ebm_training,
            optimistic_ebm_soc_min=args.optimistic_ebm_soc_min,
            optimistic_ebm_soc_max=args.optimistic_ebm_soc_max,
            optimistic_ebm_power_scale=args.optimistic_ebm_power_scale,
            optimistic_ebm_efficiency=args.optimistic_ebm_efficiency,
            optimistic_ebm_soc_penalty_scale=args.optimistic_ebm_soc_penalty_scale,
            model_path=ebm_model_path,
            dqn_action_bins=dqn_action_bins,
            force_cpu=bool(getattr(args, "cpu", False)),
            validation_selection_enabled=validation_selection_enabled,
            validation_days=validation_days,
            validation_start_hours=validation_start_hours,
            validation_interval_steps=validation_interval_steps,
            validation_eval_battery_model=validation_eval_battery_model,
            agent_hyperparams=agent_hyperparams,
            action_regularization=action_regularization,
        )

    pbm_res = evaluate_under_pbm(
        pbm_agent,
        agent_name,
        f"pbm_{suffix}",
        args.eval_days,
        seed,
        args.data_dir,
        args.eval_year,
        args.eval_start_hour,
        args.reward_mode,
        args.enable_commitment,
        args.include_component_cost,
        args.price_spread_multiplier,
        args.peak_import_penalty_per_kw,
        args.peak_import_threshold_kw,
        args.midday_pv_boost_multiplier,
        args.evening_load_boost_multiplier,
        args.battery_power_scale,
        args.battery_energy_scale,
        dqn_action_bins=dqn_action_bins,
        action_regularization=action_regularization,
    )
    ebm_res = evaluate_under_pbm(
        ebm_agent,
        agent_name,
        f"ebm_{suffix}",
        args.eval_days,
        seed,
        args.data_dir,
        args.eval_year,
        args.eval_start_hour,
        args.reward_mode,
        args.enable_commitment,
        args.include_component_cost,
        args.price_spread_multiplier,
        args.peak_import_penalty_per_kw,
        args.peak_import_threshold_kw,
        args.midday_pv_boost_multiplier,
        args.evening_load_boost_multiplier,
        args.battery_power_scale,
        args.battery_energy_scale,
        dqn_action_bins=dqn_action_bins,
        action_regularization=action_regularization,
    )
    idle_res = evaluate_idle_baseline(
        args.eval_days,
        seed,
        args.data_dir,
        args.eval_year,
        args.eval_start_hour,
        args.reward_mode,
        args.enable_commitment,
        args.include_component_cost,
        args.price_spread_multiplier,
        args.peak_import_penalty_per_kw,
        args.peak_import_threshold_kw,
        args.midday_pv_boost_multiplier,
        args.evening_load_boost_multiplier,
        args.battery_power_scale,
        args.battery_energy_scale,
    )

    export_timeseries(pbm_res, output_dir / f"pbm_{suffix}_timeseries.csv")
    export_timeseries(ebm_res, output_dir / f"ebm_{suffix}_timeseries.csv")

    gap_pct = 100.0 * (ebm_res["total_cost"] - pbm_res["total_cost"]) / max(abs(pbm_res["total_cost"]), 1e-9)
    threshold_met = gap_pct >= args.gap_threshold
    pbm_value_vs_idle = idle_res["total_cost"] - pbm_res["total_cost"]
    ebm_value_vs_idle = idle_res["total_cost"] - ebm_res["total_cost"]

    summary_df = pd.DataFrame([pbm_res, ebm_res]).drop(columns=SUMMARY_DROP_COLUMNS, errors="ignore")
    summary_df.insert(0, "agent", agent_name)
    summary_df["idle_baseline_cost"] = idle_res["total_cost"]
    summary_df["gap_threshold_pct"] = args.gap_threshold
    summary_df.to_csv(output_dir / "summary.csv", index=False)

    report = {
        "agent": agent_name,
        "dqn_action_bins": int(dqn_action_bins),
        "seed": seed,
        "steps": args.steps,
        "pbm_replay_buffer_size": int(replay_buffer_size),
        "ebm_replay_buffer_size": int(replay_buffer_size),
        "validation_selection_enabled": bool(validation_selection_enabled),
        "validation_days": int(validation_days),
        "validation_interval_steps": int(validation_interval_steps),
        "validation_start_hours": [int(hour) for hour in validation_start_hours],
        "validation_eval_battery_model": str(validation_eval_battery_model),
        "train_days": args.train_days,
        "eval_days": args.eval_days,
        "train_year": args.train_year,
        "eval_year": args.eval_year,
        "data_dir": args.data_dir,
        "reward_mode": args.reward_mode,
        "component_commitment_enabled": bool(args.enable_commitment),
        "include_component_cost_in_objective": bool(args.include_component_cost),
        "random_train_window": bool(args.random_train_window),
        "random_train_soc": bool(random_train_soc),
        "stress_episode_sampling": bool(stress_episode_sampling),
        "stress_sampling_strength": args.stress_sampling_strength,
        "curriculum_days": curriculum_days,
        "train_soc_min": args.train_soc_min,
        "train_soc_max": args.train_soc_max,
        "optimistic_ebm_training": bool(optimistic_ebm_training),
        "optimistic_ebm_soc_min": args.optimistic_ebm_soc_min,
        "optimistic_ebm_soc_max": args.optimistic_ebm_soc_max,
        "optimistic_ebm_power_scale": args.optimistic_ebm_power_scale,
        "optimistic_ebm_soc_penalty_scale": args.optimistic_ebm_soc_penalty_scale,
        "price_spread_multiplier": args.price_spread_multiplier,
        "peak_import_penalty_per_kw": args.peak_import_penalty_per_kw,
        "peak_import_threshold_kw": args.peak_import_threshold_kw,
        "midday_pv_boost_multiplier": args.midday_pv_boost_multiplier,
        "evening_load_boost_multiplier": args.evening_load_boost_multiplier,
        "battery_power_scale": args.battery_power_scale,
        "battery_energy_scale": args.battery_energy_scale,
        "idle_cost": idle_res["total_cost"],
        "idle_peak_penalty": idle_res["peak_import_penalty_sum"],
        "idle_max_grid_import_kw": idle_res["max_grid_import_kw"],
        "pbm_cost": pbm_res["total_cost"],
        "ebm_cost": ebm_res["total_cost"],
        "pbm_value_vs_idle": pbm_value_vs_idle,
        "ebm_value_vs_idle": ebm_value_vs_idle,
        "pbm_peak_penalty": pbm_res["peak_import_penalty_sum"],
        "ebm_peak_penalty": ebm_res["peak_import_penalty_sum"],
        "pbm_max_grid_import_kw": pbm_res["max_grid_import_kw"],
        "ebm_max_grid_import_kw": ebm_res["max_grid_import_kw"],
        "pbm_p_cmd_abs_p95_kw": pbm_res["p_cmd_abs_p95_kw"],
        "ebm_p_cmd_abs_p95_kw": ebm_res["p_cmd_abs_p95_kw"],
        "pbm_p_actual_abs_p95_kw": pbm_res["p_actual_abs_p95_kw"],
        "ebm_p_actual_abs_p95_kw": ebm_res["p_actual_abs_p95_kw"],
        "pbm_clip_ratio": pbm_res["clip_ratio"],
        "ebm_clip_ratio": ebm_res["clip_ratio"],
        "gap_pct": gap_pct,
        "gap_threshold_pct": args.gap_threshold,
        "threshold_met": bool(threshold_met),
        "cpu": bool(getattr(args, "cpu", False)),
        "agent_hyperparams": agent_hyperparams,
        "action_regularization": action_regularization,
        "pbm_training_meta": pbm_train_meta,
        "ebm_training_meta": ebm_train_meta,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate PBM vs EBM on MG-CIGRE with year-split data")
    parser.add_argument("--agent", type=str, default="sac", choices=list(SUPPORTED_AGENT_NAMES))
    parser.add_argument("--dqn-action-bins", type=int, default=CIGRE_DQN_ACTION_BINS)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--train-days", type=int, default=365)
    parser.add_argument("--eval-days", type=int, default=365)
    parser.add_argument("--train-year", type=int, default=2023)
    parser.add_argument("--eval-year", type=int, default=2024)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds (overrides --seed)")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--train-start-hour", type=int, default=0)
    parser.add_argument("--eval-start-hour", type=int, default=0)
    parser.add_argument("--random-train-window", action="store_true")
    parser.add_argument("--reward-mode", type=str, default="cost", choices=["cost", "legacy"])
    parser.add_argument("--enable-commitment", action="store_true")
    parser.add_argument("--include-component-cost", action="store_true")
    parser.add_argument("--disable-random-train-soc", action="store_true")
    parser.add_argument("--train-soc-min", type=float, default=0.35)
    parser.add_argument("--train-soc-max", type=float, default=0.80)
    parser.add_argument("--disable-optimistic-ebm", action="store_true")
    parser.add_argument("--optimistic-ebm-soc-min", type=float, default=0.00)
    parser.add_argument("--optimistic-ebm-soc-max", type=float, default=1.00)
    parser.add_argument("--optimistic-ebm-power-scale", type=float, default=1.80)
    parser.add_argument("--optimistic-ebm-efficiency", type=float, default=1.00)
    parser.add_argument("--optimistic-ebm-soc-penalty-scale", type=float, default=0.10)
    parser.add_argument("--price-spread-multiplier", type=float, default=8.0)
    parser.add_argument("--peak-import-penalty-per-kw", type=float, default=1.50)
    parser.add_argument("--peak-import-threshold-kw", type=float, default=10.0)
    parser.add_argument("--midday-pv-boost-multiplier", type=float, default=1.25)
    parser.add_argument("--evening-load-boost-multiplier", type=float, default=1.35)
    parser.add_argument("--disable-stress-episode-sampling", action="store_true")
    parser.add_argument("--stress-sampling-strength", type=float, default=6.0)
    parser.add_argument("--curriculum-days", type=str, default="90,180,365")
    parser.add_argument("--battery-power-scale", type=float, default=1.30)
    parser.add_argument("--battery-energy-scale", type=float, default=1.10)
    parser.add_argument("--policy-net-arch", type=str, default="256,128,64")
    parser.add_argument("--rl-learning-rate", type=float, default=3e-4)
    parser.add_argument("--rl-learning-starts", type=int, default=1000)
    parser.add_argument("--rl-batch-size", type=int, default=384)
    parser.add_argument("--rl-gamma", type=float, default=0.985)
    parser.add_argument("--rl-tau", type=float, default=0.003)
    parser.add_argument("--sac-ent-coef", type=str, default="auto_0.02")
    parser.add_argument("--sac-target-entropy-scale", type=float, default=0.35)
    parser.add_argument("--td3-action-noise-sigma", type=float, default=0.10)
    parser.add_argument("--td3-policy-delay", type=int, default=2)
    parser.add_argument("--td3-target-policy-noise", type=float, default=0.2)
    parser.add_argument("--td3-target-noise-clip", type=float, default=0.5)
    parser.add_argument("--ddpg-action-noise-sigma", type=float, default=0.10)
    parser.add_argument("--d4pg-action-noise-sigma", type=float, default=0.10)
    parser.add_argument("--d4pg-learning-starts", type=int, default=512)
    parser.add_argument("--d4pg-batch-size", type=int, default=256)
    parser.add_argument("--d4pg-collect-n-sample", type=int, default=32)
    parser.add_argument("--d4pg-update-per-collect", type=int, default=2)
    parser.add_argument("--d4pg-n-step", type=int, default=3)
    parser.add_argument("--d4pg-n-atom", type=int, default=51)
    parser.add_argument("--d4pg-v-min", type=float, default=-12000.0)
    parser.add_argument("--d4pg-v-max", type=float, default=50.0)
    parser.add_argument("--on-policy-batch-size", type=int, default=128)
    parser.add_argument("--on-policy-rollout-steps", type=int, default=2048)
    parser.add_argument("--trpo-rollout-steps", type=int, default=2048)
    parser.add_argument("--trpo-target-kl", type=float, default=0.01)
    parser.add_argument("--trpo-cg-damping", type=float, default=0.1)
    parser.add_argument("--action-smoothing-coef", type=float, default=0.0)
    parser.add_argument("--action-max-delta", type=float, default=0.0)
    parser.add_argument("--action-rate-penalty", type=float, default=0.0)
    parser.add_argument("--enable-symmetric-battery-action", action="store_true")
    parser.add_argument("--enable-validation-selection", dest="validation_selection", action="store_true")
    parser.add_argument("--disable-validation-selection", dest="validation_selection", action="store_false")
    parser.add_argument("--validation-days", type=int, default=CIGRE_VALIDATION_SELECTION_DAYS)
    parser.add_argument("--validation-interval-steps", type=int, default=CIGRE_VALIDATION_INTERVAL_STEPS)
    parser.add_argument(
        "--validation-start-hours",
        type=str,
        default="0,2184,4368,6552",
        help="Comma-separated validation window start hours used when validation selection is enabled.",
    )
    parser.add_argument("--validation-eval-battery-model", type=str, default="thevenin")
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "results" / "cigre_yearsplit_gap_100k_fourthcut"))
    parser.add_argument("--models-dir", type=str, default=str(PROJECT_ROOT / "models" / "cigre_yearsplit_gap_100k_fourthcut"))
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--gap-threshold", type=float, default=1.5)
    parser.add_argument("--cpu", action="store_true", help="Force CPU training/evaluation.")
    parser.set_defaults(validation_selection=False)
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    models_dir = Path(args.models_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    random_train_soc = not args.disable_random_train_soc
    optimistic_ebm_training = not args.disable_optimistic_ebm
    stress_episode_sampling = not args.disable_stress_episode_sampling
    curriculum_days = parse_curriculum_days(args.curriculum_days, args.train_days)

    seeds = _dedupe_ints(parse_int_list(args.seeds)) if args.seeds is not None else [int(args.seed)]
    if not seeds:
        seeds = [int(args.seed)]

    if len(seeds) == 1:
        report = run_seed_experiment(
            args=args,
            seed=int(seeds[0]),
            output_dir=output_dir,
            models_dir=models_dir,
            curriculum_days=curriculum_days,
            random_train_soc=random_train_soc,
            stress_episode_sampling=stress_episode_sampling,
            optimistic_ebm_training=optimistic_ebm_training,
        )
        print(json.dumps(report, indent=2))
        return

    seed_rows: list[dict] = []
    for seed in seeds:
        seed_output_dir = output_dir / f"seed{seed}"
        seed_models_dir = models_dir / f"seed{seed}"
        report = run_seed_experiment(
            args=args,
            seed=int(seed),
            output_dir=seed_output_dir,
            models_dir=seed_models_dir,
            curriculum_days=curriculum_days,
            random_train_soc=random_train_soc,
            stress_episode_sampling=stress_episode_sampling,
            optimistic_ebm_training=optimistic_ebm_training,
        )
        seed_rows.append(
            {
                "agent": str(report.get("agent", canonicalize_agent_name(args.agent))),
                "seed": int(seed),
                "pbm_cost": float(report.get("pbm_cost", 0.0)),
                "ebm_cost": float(report.get("ebm_cost", 0.0)),
                "gap_pct": float(report.get("gap_pct", 0.0)),
                "threshold_met": bool(report.get("threshold_met", False)),
            }
        )

    summary_path = output_dir / "summary_seeds.csv"
    summary_df = pd.DataFrame(seed_rows)
    if summary_path.exists():
        try:
            existing = pd.read_csv(summary_path)
        except Exception:
            existing = None
        if existing is not None and not existing.empty:
            combined = pd.concat([existing, summary_df], ignore_index=True)
            if "agent" in combined.columns and "seed" in combined.columns:
                combined["seed"] = combined["seed"].astype(int)
                combined = combined.drop_duplicates(subset=["agent", "seed"], keep="last").sort_values(["agent", "seed"])
            elif "seed" in combined.columns:
                combined["seed"] = combined["seed"].astype(int)
                combined = combined.drop_duplicates(subset=["seed"], keep="last").sort_values("seed")
            summary_df = combined
    summary_df.to_csv(summary_path, index=False)
    print(f"Wrote {len(summary_df)} seeds to {summary_path}")


if __name__ == "__main__":
    main()
