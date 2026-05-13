# Ref: docs/spec/task.md (Task-ID: SPEC-SAFE-OFFLINE-RL-001)

from __future__ import annotations

import json
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import pytest
import torch as th
from stable_baselines3 import SAC

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from microgrid_sim.training.bc_warmstart import (
    _extract_replay_buffer_observation_action_arrays,
    _prepare_safe_distillation_targets,
    apply_bc_warmstart,
    distill_sac_actor_from_replay_buffer,
    evaluate_actor_behavior_cloning_mse,
    load_offline_dataset,
    offline_dataset_to_transition_batch,
    seed_replay_buffer_from_offline_dataset,
)


def _make_offline_dataset(action_scale: float = 1.0) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for step, obs in enumerate(
        (
            np.array([0.1, -0.2, 0.3], dtype=np.float32),
            np.array([0.4, 0.0, -0.1], dtype=np.float32),
            np.array([-0.2, 0.5, 0.2], dtype=np.float32),
            np.array([0.0, -0.4, 0.6], dtype=np.float32),
        )
    ):
        next_obs = obs + 0.05
        action = float(np.clip((obs[0] - 0.5 * obs[1] + 0.25 * obs[2]) * action_scale, -1.0, 1.0))
        rows.append(
            {
                "case": "ieee33" if step < 3 else "cigre",
                "regime": "network_stress" if step < 3 else "base",
                "battery_model": "simple",
                "controller_source": "heuristic" if step < 2 else "oracle",
                "step": step,
                "obs": json.dumps(obs.tolist()),
                "next_obs": json.dumps(next_obs.tolist()),
                "action": action,
                "reward": float(1.0 - step),
                "done": int(step == 3),
            }
        )
    return pd.DataFrame(rows)


def test_load_offline_dataset_filters_rows():
    dataset = _make_offline_dataset()

    filtered = load_offline_dataset(
        dataset,
        controller_sources=["oracle"],
        cases=["ieee33"],
        regimes=["network_stress"],
    )

    assert len(filtered) == 1
    assert filtered.loc[0, "controller_source"] == "oracle"
    assert filtered.loc[0, "case"] == "ieee33"


def test_seed_replay_buffer_from_offline_dataset_preserves_transition_payload():
    class DummyReplayBuffer:
        def __init__(self):
            self.calls = []
            self.pos = 0
            self.buffer_size = 8
            self.n_envs = 1
            self.observations = np.zeros((8, 1, 3), dtype=np.float32)
            self.actions = np.zeros((8, 1, 1), dtype=np.float32)

        def add(self, obs, next_obs, action, rewards, dones, infos):
            self.calls.append(
                {
                    "obs": np.array(obs, copy=True),
                    "next_obs": np.array(next_obs, copy=True),
                    "action": np.array(action, copy=True),
                    "rewards": np.array(rewards, copy=True),
                    "dones": np.array(dones, copy=True),
                    "infos": infos,
                }
            )
            self.pos += 1

    class DummyPolicy:
        @staticmethod
        def scale_action(action):
            return np.asarray(action, dtype=np.float32)

    class DummyAgent:
        def __init__(self):
            self.replay_buffer = DummyReplayBuffer()
            self.policy = DummyPolicy()
            self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    dataset = load_offline_dataset(_make_offline_dataset(), cases=["ieee33"], regimes=["network_stress"])
    batch = offline_dataset_to_transition_batch(dataset)
    assert batch.observations.shape == (3, 3)

    agent = DummyAgent()
    seeded = seed_replay_buffer_from_offline_dataset(agent, dataset, max_transitions=2)

    assert seeded == 2
    assert len(agent.replay_buffer.calls) == 2
    first = agent.replay_buffer.calls[0]
    assert first["obs"].shape == (1, 3)
    assert first["next_obs"].shape == (1, 3)
    assert first["action"].shape == (1, 1)
    assert first["infos"][0]["offline_dataset"] is True
    assert first["infos"][0]["case"] == "ieee33"
    assert float(agent.replay_buffer._safe_replay_meta_offline_dataset[0, 0]) == pytest.approx(1.0)


def test_apply_bc_warmstart_reduces_sac_actor_mse():
    env = gym.make("Pendulum-v1")
    try:
        agent = SAC(
            "MlpPolicy",
            env,
            learning_starts=1,
            buffer_size=256,
            batch_size=16,
            seed=42,
        )

        obs_rows = []
        low = env.observation_space.low.astype(np.float32)
        high = env.observation_space.high.astype(np.float32)
        rng = np.random.default_rng(42)
        for step in range(128):
            obs = rng.uniform(low=low, high=high).astype(np.float32)
            next_obs = np.clip(obs + rng.normal(0.0, 0.05, size=obs.shape).astype(np.float32), low, high)
            action = float(np.clip(0.6 * obs[0] - 0.2 * obs[1] + 0.05 * obs[2], -2.0, 2.0))
            obs_rows.append(
                {
                    "case": "cigre",
                    "regime": "base",
                    "battery_model": "simple",
                    "controller_source": "oracle",
                    "step": step,
                    "obs": json.dumps(obs.tolist()),
                    "next_obs": json.dumps(next_obs.tolist()),
                    "action": action,
                    "reward": float(-abs(action)),
                    "done": 0,
                }
            )
        dataset = pd.DataFrame(obs_rows)

        initial_mse = evaluate_actor_behavior_cloning_mse(agent, dataset)
        report = apply_bc_warmstart(
            agent,
            dataset,
            replay_seed_limit=64,
            actor_prefit_gradient_steps=200,
            actor_prefit_batch_size=32,
            actor_prefit_learning_rate=1e-3,
            shuffle_seed=42,
        )
        final_mse = evaluate_actor_behavior_cloning_mse(agent, dataset)

        assert report.dataset_rows == 128
        assert report.replay_seeded_transitions == 64
        assert report.actor_gradient_steps == 200
        assert final_mse < initial_mse
        assert report.final_actor_mse < report.initial_actor_mse
    finally:
        env.close()


def test_replay_buffer_safe_distillation_reduces_actor_mse():
    env = gym.make("Pendulum-v1")
    try:
        agent = SAC(
            "MlpPolicy",
            env,
            learning_starts=1,
            buffer_size=256,
            batch_size=16,
            seed=123,
        )

        low = env.observation_space.low.astype(np.float32)
        high = env.observation_space.high.astype(np.float32)
        rng = np.random.default_rng(123)
        obs_rows = []
        intervention_rows = 0
        teacher_rows = 0
        for idx in range(96):
            obs = rng.uniform(low=low, high=high).astype(np.float32)
            next_obs = np.clip(obs + rng.normal(0.0, 0.05, size=obs.shape).astype(np.float32), low, high)
            action = float(np.clip(0.4 * obs[0] - 0.1 * obs[1] + 0.03 * obs[2], -1.0, 1.0))
            obs_rows.append((obs, next_obs, action))
            agent.replay_buffer.add(
                obs.reshape(1, -1),
                next_obs.reshape(1, -1),
                np.asarray([[action]], dtype=np.float32),
                np.asarray([0.0], dtype=np.float32),
                np.asarray([0.0], dtype=np.float32),
                [{}],
            )
            slot = idx
            if idx == 0:
                for field in (
                    "_safe_replay_meta_shield_applied",
                    "_safe_replay_meta_shield_delta_abs",
                    "_safe_replay_meta_shield_boundary_active",
                    "_safe_replay_meta_shield_terminal_active",
                    "_safe_replay_meta_shield_reserve_active",
                    "_safe_replay_meta_inventory_teacher_action",
                    "_safe_replay_meta_inventory_teacher_active",
                    "_safe_replay_meta_inventory_teacher_boundary_active",
                    "_safe_replay_meta_inventory_teacher_terminal_active",
                    "_safe_replay_meta_inventory_teacher_reserve_active",
                    "_safe_replay_meta_inventory_teacher_weight",
                    "_safe_replay_meta_battery_action_raw",
                    "_safe_replay_meta_battery_action_applied_pre_shield",
                    "_safe_replay_meta_soc",
                    "_safe_replay_meta_terminal_soc_deviation",
                    "_safe_replay_meta_peak_reserve_shortfall",
                ):
                    setattr(agent.replay_buffer, field, np.zeros((agent.replay_buffer.buffer_size, agent.replay_buffer.n_envs), dtype=np.float32))
            if idx % 5 == 0:
                agent.replay_buffer._safe_replay_meta_shield_applied[slot, 0] = 1.0
                agent.replay_buffer._safe_replay_meta_shield_delta_abs[slot, 0] = 0.6
                agent.replay_buffer._safe_replay_meta_shield_boundary_active[slot, 0] = 1.0
                intervention_rows += 1
            if idx % 7 == 0:
                teacher_action = float(np.clip(action + 0.15, -1.0, 1.0))
                agent.replay_buffer._safe_replay_meta_inventory_teacher_action[slot, 0] = teacher_action
                agent.replay_buffer._safe_replay_meta_inventory_teacher_active[slot, 0] = 1.0
                agent.replay_buffer._safe_replay_meta_inventory_teacher_terminal_active[slot, 0] = 1.0
                agent.replay_buffer._safe_replay_meta_inventory_teacher_reserve_active[slot, 0] = 1.0
                agent.replay_buffer._safe_replay_meta_inventory_teacher_weight[slot, 0] = 1.8
                agent.replay_buffer._safe_replay_meta_terminal_soc_deviation[slot, 0] = 0.2
                teacher_rows += 1

        device = getattr(agent, "device", "cpu")
        actor = agent.actor
        observations = np.stack([row[0] for row in obs_rows], axis=0).astype(np.float32)
        targets = np.asarray([[row[2]] for row in obs_rows], dtype=np.float32)
        obs_tensor = th.as_tensor(observations, dtype=th.float32, device=device)
        target_tensor = th.as_tensor(targets, dtype=th.float32, device=device)
        with th.no_grad():
            initial_mse = float(th.mean((actor(obs_tensor, deterministic=True) - target_tensor) ** 2).cpu().item())

        report = distill_sac_actor_from_replay_buffer(
            agent,
            gradient_steps=150,
            batch_size=32,
            learning_rate=1e-3,
            max_samples=96,
            shuffle_seed=123,
        )

        with th.no_grad():
            final_mse = float(th.mean((actor(obs_tensor, deterministic=True) - target_tensor) ** 2).cpu().item())

        assert report.replay_rows == 96
        assert report.actor_gradient_steps == 150
        assert report.intervention_rows == intervention_rows
        assert report.inventory_teacher_rows == teacher_rows
        assert report.mean_sample_weight > 1.0
        assert final_mse < initial_mse
        assert report.final_actor_mse < report.initial_actor_mse
    finally:
        env.close()


def test_prepare_safe_distillation_targets_only_overrides_material_teacher_rows():
    actions = np.asarray([[0.1], [0.2], [0.3], [0.4]], dtype=np.float32)
    metadata = {
        "shield_applied": np.asarray([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
        "inventory_teacher_active": np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        "inventory_teacher_action": np.asarray([0.8, -0.7, -0.9, 0.6], dtype=np.float32),
        "inventory_teacher_boundary_active": np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        "inventory_teacher_terminal_active": np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "inventory_teacher_reserve_active": np.asarray([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
        "inventory_teacher_peak_value_active": np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "inventory_teacher_valley_value_active": np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "inventory_teacher_weight": np.asarray([1.9, 1.8, 1.7, 1.75], dtype=np.float32),
        "shield_boundary_active": np.zeros((4,), dtype=np.float32),
        "shield_terminal_active": np.zeros((4,), dtype=np.float32),
        "shield_reserve_active": np.zeros((4,), dtype=np.float32),
        "shield_delta_abs": np.zeros((4,), dtype=np.float32),
        "terminal_soc_deviation": np.zeros((4,), dtype=np.float32),
    }

    targets, sample_weights, _, diagnostics = _prepare_safe_distillation_targets(
        actions=actions,
        metadata=metadata,
        intervention_priority_coef=4.0,
        boundary_priority_coef=2.0,
        terminal_priority_coef=2.0,
        reserve_priority_coef=1.0,
        teacher_priority_coef=2.0,
        peak_value_priority_coef=0.75,
        valley_value_priority_coef=0.5,
        delta_priority_coef=2.0,
        terminal_deviation_priority_coef=1.0,
    )

    assert targets[:, 0].tolist() == pytest.approx([0.1, -0.7, 0.3, 0.6])
    assert diagnostics["inventory_teacher_rows"] == 3
    assert sample_weights[0] == pytest.approx(1.0)
    assert sample_weights[1] > sample_weights[0]
    assert sample_weights[3] > sample_weights[0]
    assert sample_weights[3] < sample_weights[1]


def test_extract_replay_buffer_observation_action_arrays_can_exclude_offline_seed_rows():
    class DummyReplayBuffer:
        def __init__(self):
            self.buffer_size = 4
            self.n_envs = 1
            self.observations = np.asarray(
                [[[-1.0, 0.0]], [[-0.5, 0.1]], [[0.5, 0.2]], [[1.0, 0.3]]],
                dtype=np.float32,
            )
            self.actions = np.asarray([[[0.1]], [[0.2]], [[0.3]], [[0.4]]], dtype=np.float32)
            for field in (
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
            ):
                setattr(self, f"_safe_replay_meta_{field}", np.zeros((4, 1), dtype=np.float32))
            self._safe_replay_meta_offline_dataset[0, 0] = 1.0
            self._safe_replay_meta_offline_dataset[1, 0] = 1.0

        def size(self):
            return 4

    class DummyAgent:
        def __init__(self):
            self.replay_buffer = DummyReplayBuffer()

    observations, actions, metadata = _extract_replay_buffer_observation_action_arrays(
        DummyAgent(),
        exclude_offline_dataset=True,
    )

    assert observations.shape == (2, 2)
    assert actions.shape == (2, 1)
    assert observations[:, 0].tolist() == pytest.approx([0.5, 1.0])
    assert metadata["offline_dataset"].tolist() == pytest.approx([0.0, 0.0])


def test_prepare_safe_distillation_targets_downweights_pure_value_teacher_rows_vs_structural_rows():
    actions = np.asarray([[0.1], [0.2], [0.3]], dtype=np.float32)
    metadata = {
        "shield_applied": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        "inventory_teacher_active": np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
        "inventory_teacher_action": np.asarray([0.8, -0.7, 0.6], dtype=np.float32),
        "inventory_teacher_boundary_active": np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        "inventory_teacher_terminal_active": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        "inventory_teacher_reserve_active": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        "inventory_teacher_peak_value_active": np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        "inventory_teacher_valley_value_active": np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        "inventory_teacher_weight": np.asarray([1.8, 1.8, 1.8], dtype=np.float32),
        "shield_boundary_active": np.zeros((3,), dtype=np.float32),
        "shield_terminal_active": np.zeros((3,), dtype=np.float32),
        "shield_reserve_active": np.zeros((3,), dtype=np.float32),
        "shield_delta_abs": np.zeros((3,), dtype=np.float32),
        "terminal_soc_deviation": np.zeros((3,), dtype=np.float32),
    }

    _, sample_weights, _, _ = _prepare_safe_distillation_targets(
        actions=actions,
        metadata=metadata,
        intervention_priority_coef=4.0,
        boundary_priority_coef=2.0,
        terminal_priority_coef=2.0,
        reserve_priority_coef=1.0,
        teacher_priority_coef=2.0,
        peak_value_priority_coef=0.75,
        valley_value_priority_coef=0.5,
        delta_priority_coef=2.0,
        terminal_deviation_priority_coef=1.0,
    )

    assert sample_weights[0] > sample_weights[1] > sample_weights[2] > 1.0
