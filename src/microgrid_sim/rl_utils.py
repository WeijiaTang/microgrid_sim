"""Shared SB3 algorithm helpers for focused experiment entrypoints."""

from __future__ import annotations

import numpy as np
from gymnasium import spaces
from stable_baselines3 import DQN, PPO, SAC, TD3
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.noise import NormalActionNoise

SUPPORTED_AGENT_NAMES = ("sac", "ppo", "td3", "dqn")
OFF_POLICY_AGENT_NAMES = frozenset({"sac", "td3", "dqn"})


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
    if agent == "sac":
        return SAC
    if agent == "ppo":
        return PPO
    if agent == "td3":
        return TD3
    return DQN


def load_agent(agent_name: str, model_path: str, env=None, device: str = "auto") -> BaseAlgorithm:
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
    ppo_batch_size = max(int(hyperparams.get("ppo_batch_size", 128)), 1)
    dqn_batch_size = max(int(hyperparams.get("dqn_batch_size", 256)), 1)
    gamma = float(hyperparams.get("gamma", 0.985))
    tau = float(hyperparams.get("tau", 0.003))
    td3_action_noise_sigma = max(float(hyperparams.get("td3_action_noise_sigma", 0.10)), 0.0)
    td3_policy_delay = max(int(hyperparams.get("td3_policy_delay", 2)), 1)
    td3_target_policy_noise = max(float(hyperparams.get("td3_target_policy_noise", 0.2)), 0.0)
    td3_target_noise_clip = max(float(hyperparams.get("td3_target_noise_clip", 0.5)), 0.0)
    dqn_exploration_fraction = float(hyperparams.get("dqn_exploration_fraction", 0.25))
    dqn_exploration_final_eps = float(hyperparams.get("dqn_exploration_final_eps", 0.05))
    sac_ent_coef = str(hyperparams.get("sac_ent_coef", sac_ent_coef))
    sac_target_entropy_scale = float(hyperparams.get("sac_target_entropy_scale", sac_target_entropy_scale))

    actor_critic_kwargs = dict(net_arch=dict(pi=hidden_sizes, qf=hidden_sizes))
    if agent == "sac":
        return SAC(
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
        )
    if agent == "ppo":
        return PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=2048,
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
        )
    if agent == "td3":
        noise_sigma = td3_action_noise_sigma * np.ones(max(action_dim, 1), dtype=float)
        return TD3(
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
        )
    if not isinstance(action_space, spaces.Discrete):
        raise TypeError("DQN requires a discrete action space.")
    return DQN(
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
    )
