# Ref: docs/spec/task.md (Task-ID: SPEC-SAFE-OFFLINE-RL-001)

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import gymnasium as gym
import pytest
from stable_baselines3 import SAC

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from microgrid_sim.rl_utils import _extract_safe_replay_metadata_from_infos, _patch_offpolicy_agent_to_store_effective_actions


class _DummyAgent:
    def __init__(self):
        self.calls: list[dict[str, object]] = []

    def _store_transition(self, replay_buffer, buffer_action, new_obs, reward, dones, infos):
        del replay_buffer, new_obs, reward, dones
        self.calls.append(
            {
                "buffer_action": np.asarray(buffer_action, dtype=np.float32).copy(),
                "infos": infos,
            }
        )
        return "ok"


def test_offpolicy_patch_stores_shielded_action_instead_of_raw_policy_action():
    agent = _DummyAgent()
    patched = _patch_offpolicy_agent_to_store_effective_actions(agent)

    result = patched._store_transition(
        replay_buffer=None,
        buffer_action=np.asarray([[0.85]], dtype=np.float32),
        new_obs=None,
        reward=np.asarray([0.0], dtype=np.float32),
        dones=np.asarray([False]),
        infos=[{"shield_post_action": 0.15}],
    )

    assert result == "ok"
    assert len(agent.calls) == 1
    assert float(agent.calls[0]["buffer_action"][0, 0]) == pytest.approx(0.15)


def test_offpolicy_patch_falls_back_to_applied_action_when_shield_field_absent():
    agent = _DummyAgent()
    patched = _patch_offpolicy_agent_to_store_effective_actions(agent)

    patched._store_transition(
        replay_buffer=None,
        buffer_action=np.asarray([[-0.6]], dtype=np.float32),
        new_obs=None,
        reward=np.asarray([0.0], dtype=np.float32),
        dones=np.asarray([False]),
        infos=[{"battery_action_applied": -0.2}],
    )

    assert float(agent.calls[0]["buffer_action"][0, 0]) == pytest.approx(-0.2)


def test_offpolicy_patch_preserves_original_buffer_action_without_effective_action_fields():
    agent = _DummyAgent()
    patched = _patch_offpolicy_agent_to_store_effective_actions(agent)

    patched._store_transition(
        replay_buffer=None,
        buffer_action=np.asarray([[0.33]], dtype=np.float32),
        new_obs=None,
        reward=np.asarray([0.0], dtype=np.float32),
        dones=np.asarray([False]),
        infos=[{"offline_dataset": True}],
    )

    assert float(agent.calls[0]["buffer_action"][0, 0]) == pytest.approx(0.33)


def test_offpolicy_patch_records_safe_replay_metadata_on_replay_buffer():
    class DummyReplayBuffer:
        def __init__(self):
            self.buffer_size = 8
            self.n_envs = 1
            self.pos = 0
            self.observations = np.zeros((8, 1, 3), dtype=np.float32)

    agent = _DummyAgent()
    patched = _patch_offpolicy_agent_to_store_effective_actions(agent)
    replay_buffer = DummyReplayBuffer()

    patched._store_transition(
        replay_buffer=replay_buffer,
        buffer_action=np.asarray([[0.75]], dtype=np.float32),
        new_obs=None,
        reward=np.asarray([0.0], dtype=np.float32),
        dones=np.asarray([False]),
        infos=[
            {
                "shield_post_action": 0.1,
                "shield_applied": 1,
                "shield_delta": -0.65,
                "shield_boundary_active": 1,
                "inventory_teacher_action": -0.2,
                "inventory_teacher_active": 1,
                "inventory_teacher_reserve_active": 1,
                "inventory_teacher_peak_value_active": 1,
                "inventory_teacher_weight": 2.5,
                "soc": 0.88,
                "terminal_soc_deviation": 0.12,
                "peak_reserve_shortfall": 0.3,
                "offline_dataset": True,
            }
        ],
    )

    assert float(replay_buffer._safe_replay_meta_shield_applied[0, 0]) == pytest.approx(1.0)
    assert float(replay_buffer._safe_replay_meta_shield_delta_abs[0, 0]) == pytest.approx(0.65)
    assert float(replay_buffer._safe_replay_meta_inventory_teacher_action[0, 0]) == pytest.approx(-0.2)
    assert float(replay_buffer._safe_replay_meta_inventory_teacher_reserve_active[0, 0]) == pytest.approx(1.0)
    assert float(replay_buffer._safe_replay_meta_inventory_teacher_peak_value_active[0, 0]) == pytest.approx(1.0)
    assert float(replay_buffer._safe_replay_meta_inventory_teacher_weight[0, 0]) == pytest.approx(2.5)
    assert float(replay_buffer._safe_replay_meta_soc[0, 0]) == pytest.approx(0.88)
    assert float(replay_buffer._safe_replay_meta_offline_dataset[0, 0]) == pytest.approx(1.0)


def test_offpolicy_patch_does_not_break_agent_save(tmp_path: Path):
    env = gym.make("Pendulum-v1")
    try:
        agent = SAC("MlpPolicy", env, learning_starts=1, buffer_size=64, batch_size=8, seed=7)
        patched = _patch_offpolicy_agent_to_store_effective_actions(agent)
        save_path = tmp_path / "patched_agent"
        patched.save(str(save_path))
        assert save_path.with_suffix(".zip").exists()
    finally:
        env.close()


def test_safe_replay_metadata_ignores_tiny_terminal_only_adjustments():
    metadata = _extract_safe_replay_metadata_from_infos(
        infos=[
            {
                "shield_applied": 1,
                "shield_delta": 0.005,
                "shield_terminal_active": 1,
                "inventory_teacher_action": 0.104,
                "inventory_teacher_active": 1,
                "inventory_teacher_terminal_active": 1,
                "battery_action_applied_pre_shield": 0.10,
                "battery_action_raw": 0.10,
            }
        ],
        row_count=1,
    )

    assert float(metadata["shield_applied"][0]) == pytest.approx(0.0)
    assert float(metadata["shield_terminal_active"][0]) == pytest.approx(0.0)
    assert float(metadata["inventory_teacher_active"][0]) == pytest.approx(0.0)
    assert float(metadata["inventory_teacher_terminal_active"][0]) == pytest.approx(0.0)
