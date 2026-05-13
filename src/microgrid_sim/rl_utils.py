"""Shared SB3 algorithm helpers for focused experiment entrypoints."""

from __future__ import annotations

import types
from typing import TYPE_CHECKING, Any

import numpy as np
from gymnasium import spaces

if TYPE_CHECKING:
    from stable_baselines3.common.base_class import BaseAlgorithm

    from .d4pg_agent import D4PGAgent

SUPPORTED_AGENT_NAMES = ("sac", "ppo", "td3", "ddpg", "d4pg", "dqn", "tqc", "trpo")
OFF_POLICY_AGENT_NAMES = frozenset({"sac", "td3", "ddpg", "d4pg", "dqn", "tqc"})
SAFE_REPLAY_META_FIELDS = (
    "offline_dataset",
    "shield_applied",
    "shield_delta_abs",
    "shield_boundary_active",
    "shield_terminal_active",
    "shield_reserve_active",
    "inventory_teacher_action",
    "inventory_teacher_active",
    "inventory_teacher_boundary_active",
    "inventory_teacher_terminal_active",
    "inventory_teacher_reserve_active",
    "inventory_teacher_peak_value_active",
    "inventory_teacher_valley_value_active",
    "inventory_teacher_weight",
    "battery_action_raw",
    "battery_action_applied_pre_shield",
    "soc",
    "terminal_soc_deviation",
    "peak_reserve_shortfall",
)
SAFE_REPLAY_MATERIAL_DELTA_THRESHOLD = 0.01


def _load_d4pg_agent_class():
    from .d4pg_agent import D4PGAgent

    return D4PGAgent


def _load_sb3_algorithm(agent_name: str):
    agent = canonicalize_agent_name(agent_name)
    if agent == "sac":
        from stable_baselines3 import SAC

        return SAC
    if agent == "ppo":
        from stable_baselines3 import PPO

        return PPO
    if agent == "td3":
        from stable_baselines3 import TD3

        return TD3
    if agent == "ddpg":
        from stable_baselines3 import DDPG

        return DDPG
    if agent == "dqn":
        from stable_baselines3 import DQN

        return DQN
    if agent == "tqc":
        from sb3_contrib import TQC

        return TQC
    if agent == "trpo":
        from sb3_contrib import TRPO

        return TRPO
    raise ValueError(f"Unsupported lazy-loaded SB3 algorithm '{agent_name}'.")


def canonicalize_agent_name(agent_name: str) -> str:
    normalized = str(agent_name).strip().lower()
    if normalized not in SUPPORTED_AGENT_NAMES:
        raise ValueError(f"Unsupported agent '{agent_name}'. Expected one of {SUPPORTED_AGENT_NAMES}.")
    return normalized


def uses_discrete_actions(agent_name: str) -> bool:
    return canonicalize_agent_name(agent_name) == "dqn"


def replay_buffer_size_for(agent_name: str, total_steps: int) -> int:
    return int(min(1_000_000, max(50_000, int(total_steps)))) if canonicalize_agent_name(agent_name) in OFF_POLICY_AGENT_NAMES else 0


def model_class_for(agent_name: str):
    agent = canonicalize_agent_name(agent_name)
    if agent == "d4pg":
        return _load_d4pg_agent_class()
    return _load_sb3_algorithm(agent)


def _extract_effective_buffer_action_from_infos(
    *,
    infos: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    fallback_buffer_action: np.ndarray,
) -> np.ndarray:
    effective = np.asarray(fallback_buffer_action, dtype=np.float32).copy()
    if effective.ndim == 1:
        effective = effective.reshape(1, -1)
    if not infos:
        return effective
    row_count = min(len(infos), int(effective.shape[0]))
    for idx in range(row_count):
        info = infos[idx] or {}
        if "shield_post_action" in info:
            value = float(info["shield_post_action"])
        elif "battery_action_applied" in info:
            value = float(info["battery_action_applied"])
        elif "action_after_rule_guidance" in info:
            value = float(info["action_after_rule_guidance"])
        else:
            continue
        effective[idx, 0] = np.clip(value, -1.0, 1.0)
    return effective.astype(np.float32, copy=False)


def _extract_safe_replay_metadata_from_infos(
    *,
    infos: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    row_count: int,
) -> dict[str, np.ndarray]:
    metadata = {
        field: np.zeros((max(int(row_count), 0),), dtype=np.float32)
        for field in SAFE_REPLAY_META_FIELDS
    }
    if not infos:
        return metadata
    for idx in range(min(len(infos), int(row_count))):
        info = infos[idx] or {}
        metadata["offline_dataset"][idx] = float(bool(info.get("offline_dataset", False)))
        shield_delta_abs = abs(float(info.get("shield_delta", 0.0)))
        shield_boundary_active = bool(info.get("shield_boundary_active", 0))
        shield_terminal_active = bool(info.get("shield_terminal_active", 0))
        shield_reserve_active = bool(info.get("shield_reserve_active", 0))
        material_shield = (
            shield_delta_abs > SAFE_REPLAY_MATERIAL_DELTA_THRESHOLD
            or shield_boundary_active
            or shield_reserve_active
        )
        metadata["shield_applied"][idx] = float(material_shield)
        metadata["shield_delta_abs"][idx] = float(shield_delta_abs)
        metadata["shield_boundary_active"][idx] = float(shield_boundary_active)
        metadata["shield_terminal_active"][idx] = float(shield_terminal_active and shield_delta_abs > SAFE_REPLAY_MATERIAL_DELTA_THRESHOLD)
        metadata["shield_reserve_active"][idx] = float(shield_reserve_active)
        teacher_action = float(info.get("inventory_teacher_action", info.get("battery_action_applied", 0.0)))
        pre_shield_action = float(info.get("battery_action_applied_pre_shield", info.get("battery_action_raw", 0.0)))
        teacher_gap = abs(teacher_action - pre_shield_action)
        teacher_boundary_active = bool(info.get("inventory_teacher_boundary_active", 0))
        teacher_terminal_active = bool(info.get("inventory_teacher_terminal_active", 0))
        teacher_reserve_active = bool(info.get("inventory_teacher_reserve_active", 0))
        teacher_peak_value_active = bool(info.get("inventory_teacher_peak_value_active", 0))
        teacher_valley_value_active = bool(info.get("inventory_teacher_valley_value_active", 0))
        material_teacher = (
            teacher_gap > SAFE_REPLAY_MATERIAL_DELTA_THRESHOLD
            or teacher_boundary_active
            or teacher_reserve_active
        )
        metadata["inventory_teacher_action"][idx] = float(teacher_action)
        metadata["inventory_teacher_active"][idx] = float(material_teacher)
        metadata["inventory_teacher_boundary_active"][idx] = float(teacher_boundary_active)
        metadata["inventory_teacher_terminal_active"][idx] = float(
            teacher_terminal_active and teacher_gap > SAFE_REPLAY_MATERIAL_DELTA_THRESHOLD
        )
        metadata["inventory_teacher_reserve_active"][idx] = float(teacher_reserve_active)
        metadata["inventory_teacher_peak_value_active"][idx] = float(
            teacher_peak_value_active and teacher_gap > SAFE_REPLAY_MATERIAL_DELTA_THRESHOLD
        )
        metadata["inventory_teacher_valley_value_active"][idx] = float(
            teacher_valley_value_active and teacher_gap > SAFE_REPLAY_MATERIAL_DELTA_THRESHOLD
        )
        metadata["inventory_teacher_weight"][idx] = float(info.get("inventory_teacher_weight", 0.0))
        metadata["battery_action_raw"][idx] = float(info.get("battery_action_raw", 0.0))
        metadata["battery_action_applied_pre_shield"][idx] = float(pre_shield_action)
        metadata["soc"][idx] = float(info.get("soc", 0.0))
        metadata["terminal_soc_deviation"][idx] = abs(float(info.get("terminal_soc_deviation", 0.0)))
        metadata["peak_reserve_shortfall"][idx] = float(info.get("peak_reserve_shortfall", 0.0))
    return metadata


def _safe_replay_meta_attr(field: str) -> str:
    return f"_safe_replay_meta_{field}"


def _ensure_safe_replay_metadata_arrays(replay_buffer, *, row_count: int) -> None:
    capacity = int(getattr(replay_buffer, "buffer_size", 0))
    if capacity <= 0 and hasattr(replay_buffer, "observations"):
        capacity = int(np.asarray(replay_buffer.observations).shape[0])
    env_count = max(int(getattr(replay_buffer, "n_envs", row_count or 1)), int(row_count), 1)
    for field in SAFE_REPLAY_META_FIELDS:
        attr = _safe_replay_meta_attr(field)
        current = getattr(replay_buffer, attr, None)
        expected_shape = (capacity, env_count)
        if isinstance(current, np.ndarray) and current.shape == expected_shape:
            continue
        setattr(replay_buffer, attr, np.zeros(expected_shape, dtype=np.float32))


def _write_safe_replay_metadata(replay_buffer, *, insert_pos: int, metadata: dict[str, np.ndarray]) -> None:
    row_count = max((int(values.shape[0]) for values in metadata.values()), default=0)
    _ensure_safe_replay_metadata_arrays(replay_buffer, row_count=row_count)
    for field, values in metadata.items():
        target = getattr(replay_buffer, _safe_replay_meta_attr(field))
        target[insert_pos, :] = 0.0
        width = min(int(values.shape[0]), int(target.shape[1]))
        if width > 0:
            target[insert_pos, :width] = np.asarray(values[:width], dtype=np.float32)


def _patch_offpolicy_agent_to_store_effective_actions(agent: "BaseAlgorithm") -> "BaseAlgorithm":
    if getattr(agent, "_safe_effective_action_buffer_patch", False):
        return agent

    agent._store_transition_original = type(agent)._store_transition
    agent._store_transition = types.MethodType(_store_transition_with_effective_actions, agent)
    agent._safe_effective_action_buffer_patch = True
    return agent


def _store_transition_with_effective_actions(self, replay_buffer, buffer_action, new_obs, reward, dones, infos):
    insert_pos = int(getattr(replay_buffer, "pos", 0)) if replay_buffer is not None else 0
    patched_buffer_action = _extract_effective_buffer_action_from_infos(
        infos=infos,
        fallback_buffer_action=np.asarray(buffer_action, dtype=np.float32),
    )
    replay_metadata = _extract_safe_replay_metadata_from_infos(
        infos=infos,
        row_count=int(np.asarray(patched_buffer_action).reshape(-1, np.asarray(patched_buffer_action).shape[-1]).shape[0]),
    )
    original_store_transition = getattr(self, "_store_transition_original", None)
    if original_store_transition is None:
        original_store_transition = type(self)._store_transition
    result = original_store_transition(self, replay_buffer, patched_buffer_action, new_obs, reward, dones, infos)
    if replay_buffer is not None:
        _write_safe_replay_metadata(replay_buffer, insert_pos=insert_pos, metadata=replay_metadata)
    return result


def load_agent(agent_name: str, model_path: str, env=None, device: str = "auto") -> "BaseAlgorithm | Any":
    model_cls = model_class_for(agent_name)
    return model_cls.load(model_path, env=env, device=device)


def create_agent(
    agent_name: str,
    env,
    total_steps: int,
    seed: int,
    device: str,
    sac_ent_coef: str = "auto",
    sac_target_entropy_scale: float = 1.0,
    agent_hyperparams: dict | None = None,
    tensorboard_log: str | None = None,
) -> BaseAlgorithm:
    agent = canonicalize_agent_name(agent_name)
    hyperparams = dict(agent_hyperparams or {})
    replay_buffer_size = replay_buffer_size_for(agent, total_steps)
    action_space = env.action_space
    if isinstance(action_space, spaces.Discrete):
        action_dim = 1
    else:
        action_dim = int(np.prod(getattr(action_space, "shape", (0,)) or (0,)))

    hidden_sizes = [max(int(size), 1) for size in hyperparams.get("net_arch", (256, 128, 64)) if int(size) > 0] or [256, 128, 64]
    learning_rate = float(hyperparams.get("learning_rate", 3e-4))
    learning_starts = max(int(hyperparams.get("learning_starts", 1000)), 0)
    off_policy_batch_size = max(int(hyperparams.get("off_policy_batch_size", 384)), 1)
    on_policy_batch_size = max(int(hyperparams.get("ppo_batch_size", 128)), 1)
    ppo_n_steps = max(int(hyperparams.get("ppo_n_steps", 2048)), 1)
    trpo_n_steps = max(int(hyperparams.get("trpo_n_steps", ppo_n_steps)), 1)
    ppo_batch_size = min(on_policy_batch_size, ppo_n_steps)
    trpo_batch_size = min(on_policy_batch_size, trpo_n_steps)
    dqn_batch_size = max(int(hyperparams.get("dqn_batch_size", 256)), 1)
    gamma = float(hyperparams.get("gamma", 0.985))
    tau = float(hyperparams.get("tau", 0.003))
    td3_action_noise_sigma = max(float(hyperparams.get("td3_action_noise_sigma", 0.10)), 0.0)
    ddpg_action_noise_sigma = max(float(hyperparams.get("ddpg_action_noise_sigma", td3_action_noise_sigma)), 0.0)
    td3_policy_delay = max(int(hyperparams.get("td3_policy_delay", 2)), 1)
    td3_target_policy_noise = max(float(hyperparams.get("td3_target_policy_noise", 0.2)), 0.0)
    td3_target_noise_clip = max(float(hyperparams.get("td3_target_noise_clip", 0.5)), 0.0)
    dqn_exploration_fraction = float(hyperparams.get("dqn_exploration_fraction", 0.25))
    dqn_exploration_final_eps = float(hyperparams.get("dqn_exploration_final_eps", 0.05))
    tqc_top_quantiles_to_drop = max(int(hyperparams.get("tqc_top_quantiles_to_drop", 2)), 0)
    trpo_target_kl = max(float(hyperparams.get("trpo_target_kl", 0.01)), 1e-6)
    trpo_cg_damping = max(float(hyperparams.get("trpo_cg_damping", 0.1)), 0.0)
    sac_ent_coef = str(hyperparams.get("sac_ent_coef", sac_ent_coef))
    sac_target_entropy_scale = float(hyperparams.get("sac_target_entropy_scale", sac_target_entropy_scale))

    actor_critic_kwargs = dict(net_arch=dict(pi=hidden_sizes, qf=hidden_sizes))
    if agent == "sac":
        sac_cls = _load_sb3_algorithm(agent)
        model = sac_cls(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            buffer_size=replay_buffer_size,
            learning_starts=learning_starts,
            batch_size=off_policy_batch_size,
            tau=tau,
            gamma=gamma,
            ent_coef=sac_ent_coef,
            target_entropy=-float(max(action_dim, 1)) * float(sac_target_entropy_scale),
            verbose=1,
            seed=seed,
            device=device,
            policy_kwargs=actor_critic_kwargs,
            tensorboard_log=tensorboard_log,
        )
        return _patch_offpolicy_agent_to_store_effective_actions(model)
    if agent == "tqc":
        tqc_cls = _load_sb3_algorithm(agent)
        model = tqc_cls(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            buffer_size=replay_buffer_size,
            learning_starts=learning_starts,
            batch_size=off_policy_batch_size,
            tau=tau,
            gamma=gamma,
            ent_coef=sac_ent_coef,
            target_entropy=-float(max(action_dim, 1)) * float(sac_target_entropy_scale),
            top_quantiles_to_drop_per_net=tqc_top_quantiles_to_drop,
            verbose=1,
            seed=seed,
            device=device,
            policy_kwargs=actor_critic_kwargs,
            tensorboard_log=tensorboard_log,
        )
        return _patch_offpolicy_agent_to_store_effective_actions(model)
    if agent == "ppo":
        ppo_cls = _load_sb3_algorithm(agent)
        return ppo_cls(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=ppo_n_steps,
            batch_size=ppo_batch_size,
            n_epochs=10,
            gamma=gamma,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            verbose=1,
            seed=seed,
            device=device,
            policy_kwargs=dict(net_arch=dict(pi=hidden_sizes, vf=hidden_sizes)),
            tensorboard_log=tensorboard_log,
        )
    if agent == "trpo":
        trpo_cls = _load_sb3_algorithm(agent)
        return trpo_cls(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=trpo_n_steps,
            batch_size=trpo_batch_size,
            gamma=gamma,
            gae_lambda=0.95,
            cg_damping=trpo_cg_damping,
            target_kl=trpo_target_kl,
            verbose=1,
            seed=seed,
            device=device,
            policy_kwargs=dict(net_arch=dict(pi=hidden_sizes, vf=hidden_sizes)),
            tensorboard_log=tensorboard_log,
        )
    if agent == "td3":
        noise_sigma = td3_action_noise_sigma * np.ones(max(action_dim, 1), dtype=float)
        from stable_baselines3.common.noise import NormalActionNoise

        td3_cls = _load_sb3_algorithm(agent)
        model = td3_cls(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            buffer_size=replay_buffer_size,
            learning_starts=learning_starts,
            batch_size=off_policy_batch_size,
            tau=tau,
            gamma=gamma,
            policy_delay=td3_policy_delay,
            target_policy_noise=td3_target_policy_noise,
            target_noise_clip=td3_target_noise_clip,
            action_noise=NormalActionNoise(mean=np.zeros_like(noise_sigma), sigma=noise_sigma),
            verbose=1,
            seed=seed,
            device=device,
            policy_kwargs=actor_critic_kwargs,
            tensorboard_log=tensorboard_log,
        )
        return _patch_offpolicy_agent_to_store_effective_actions(model)
    if agent == "ddpg":
        noise_sigma = ddpg_action_noise_sigma * np.ones(max(action_dim, 1), dtype=float)
        from stable_baselines3.common.noise import NormalActionNoise

        ddpg_cls = _load_sb3_algorithm(agent)
        model = ddpg_cls(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            buffer_size=replay_buffer_size,
            learning_starts=learning_starts,
            batch_size=off_policy_batch_size,
            tau=tau,
            gamma=gamma,
            action_noise=NormalActionNoise(mean=np.zeros_like(noise_sigma), sigma=noise_sigma),
            verbose=1,
            seed=seed,
            device=device,
            policy_kwargs=actor_critic_kwargs,
            tensorboard_log=tensorboard_log,
        )
        return _patch_offpolicy_agent_to_store_effective_actions(model)
    if agent == "d4pg":
        d4pg_agent_cls = _load_d4pg_agent_class()
        return d4pg_agent_cls(
            env=env,
            total_steps=total_steps,
            seed=seed,
            device=device,
            agent_hyperparams=hyperparams,
        )
    if not isinstance(action_space, spaces.Discrete):
        raise TypeError("DQN requires a discrete action space.")
    dqn_cls = _load_sb3_algorithm(agent)
    return dqn_cls(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        buffer_size=replay_buffer_size,
        learning_starts=learning_starts,
        batch_size=dqn_batch_size,
        gamma=gamma,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=dqn_exploration_fraction,
        exploration_initial_eps=1.0,
        exploration_final_eps=dqn_exploration_final_eps,
        verbose=1,
        seed=seed,
        device=device,
        policy_kwargs=dict(net_arch=hidden_sizes),
        tensorboard_log=tensorboard_log,
    )
