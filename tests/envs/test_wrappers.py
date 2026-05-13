from __future__ import annotations

import numpy as np
import pytest

from microgrid_sim.cases import IEEE33Config
from microgrid_sim.envs.network_microgrid import NetworkMicrogridEnv
from microgrid_sim.envs.wrappers import ContinuousActionRegularizationWrapper, RuleGuidedActionWrapper, ShieldedActionWrapper


def test_continuous_action_regularization_wrapper_clips_to_soc_feasible_range():
    env = NetworkMicrogridEnv(IEEE33Config(simulation_days=1, seed=42, battery_model="simple", regime="base"))
    wrapped = ContinuousActionRegularizationWrapper(
        env,
        battery_feasibility_aware=True,
        battery_infeasible_penalty=-2.0,
    )
    try:
        wrapped.reset(seed=42)
        env.unwrapped.battery.soc = float(env.unwrapped.config.battery_params.soc_max)
        _, reward, terminated, truncated, info = wrapped.step(np.array([-1.0], dtype=np.float32))
        del reward, terminated, truncated
        assert info["battery_action_raw"] == -1.0
        assert info["battery_action_applied"] == 0.0
        assert info["battery_action_feasible_low"] == 0.0
        assert info["battery_charge_fraction_feasible"] == 0.0
        assert info["battery_action_infeasible_gap"] == 1.0
        assert info["battery_action_infeasible_penalty"] == -2.0
    finally:
        wrapped.close()


def test_continuous_action_regularization_wrapper_exposes_previous_applied_action_in_observation():
    env = NetworkMicrogridEnv(IEEE33Config(simulation_days=1, seed=42, battery_model="simple", regime="base"))
    wrapped = ContinuousActionRegularizationWrapper(env, smoothing_coef=0.0, max_delta=0.0)
    try:
        obs, _ = wrapped.reset(seed=42)
        assert obs.shape[0] == env.observation_space.shape[0] + env.action_space.shape[0]
        assert float(obs[-1]) == 0.0

        next_obs, _, _, _, info = wrapped.step(np.array([0.25], dtype=np.float32))
        assert info["battery_action_applied"] == 0.25
        assert float(next_obs[-1]) == 0.25
    finally:
        wrapped.close()


def test_rule_guided_action_wrapper_blends_with_rule_hint():
    env = NetworkMicrogridEnv(IEEE33Config(simulation_days=1, seed=42, battery_model="simple", regime="base"))
    wrapped = RuleGuidedActionWrapper(
        env,
        guidance_mix=1.0,
        guidance_decay_steps=0,
    )
    try:
        wrapped.reset(seed=42)
        _, _, terminated, truncated, info = wrapped.step(np.array([0.0], dtype=np.float32))
        del terminated, truncated
        assert info["policy_action_pre_guidance"] == 0.0
        assert info["rule_guidance_mix"] == 1.0
        assert info["rule_guided_action"] < 0.0
        assert info["action_after_rule_guidance"] == info["rule_guided_action"]
    finally:
        wrapped.close()


def test_rule_guided_action_wrapper_exposes_mix_in_observation():
    env = NetworkMicrogridEnv(IEEE33Config(simulation_days=1, seed=42, battery_model="simple", regime="base"))
    wrapped = RuleGuidedActionWrapper(
        env,
        guidance_mix=0.6,
        guidance_decay_steps=10,
    )
    try:
        obs, _ = wrapped.reset(seed=42)
        assert obs.shape[0] == env.observation_space.shape[0] + 1
        assert float(obs[-1]) == pytest.approx(0.6)

        next_obs, _, _, _, info = wrapped.step(np.array([0.0], dtype=np.float32))
        assert info["rule_guidance_mix"] == 0.6
        assert float(next_obs[-1]) == pytest.approx(0.6)
    finally:
        wrapped.close()


def test_rule_guided_action_wrapper_can_disable_action_blending_while_preserving_observation_shape():
    env = NetworkMicrogridEnv(IEEE33Config(simulation_days=1, seed=42, battery_model="simple", regime="base"))
    wrapped = RuleGuidedActionWrapper(
        env,
        guidance_mix=0.6,
        guidance_decay_steps=10,
        guidance_enabled=False,
    )
    try:
        obs, _ = wrapped.reset(seed=42)
        assert obs.shape[0] == env.observation_space.shape[0] + 1
        assert float(obs[-1]) == 0.0

        _, _, _, _, info = wrapped.step(np.array([0.2], dtype=np.float32))
        assert info["rule_guidance_mix"] == 0.0
        assert info["action_after_rule_guidance"] == pytest.approx(0.2)
    finally:
        wrapped.close()


def test_rule_guided_action_wrapper_can_reset_guidance_progress():
    env = NetworkMicrogridEnv(IEEE33Config(simulation_days=1, seed=42, battery_model="simple", regime="base"))
    wrapped = RuleGuidedActionWrapper(
        env,
        guidance_mix=0.6,
        guidance_decay_steps=10,
    )
    try:
        wrapped.reset(seed=42)
        _, _, _, _, info_first = wrapped.step(np.array([0.0], dtype=np.float32))
        _, _, _, _, info_second = wrapped.step(np.array([0.0], dtype=np.float32))
        assert info_first["rule_guidance_mix"] == 0.6
        assert info_second["rule_guidance_mix"] < info_first["rule_guidance_mix"]
        wrapped.reset_guidance_progress()
        _, _, _, _, info_reset = wrapped.step(np.array([0.0], dtype=np.float32))
        assert info_reset["rule_guidance_mix"] == info_first["rule_guidance_mix"]
    finally:
        wrapped.close()


def test_shielded_action_wrapper_exposes_post_shield_diagnostics():
    env = NetworkMicrogridEnv(IEEE33Config(simulation_days=1, seed=42, battery_model="simple", regime="base"))
    wrapped = ShieldedActionWrapper(env, hard_pullback_action=0.2)
    try:
        wrapped.reset(seed=42)
        env.unwrapped.battery.soc = float(env.unwrapped.config.battery_params.soc_max)
        _, _, _, _, info = wrapped.step(np.array([-0.8], dtype=np.float32))
        assert info["shield_enabled"] == 1
        assert info["shield_applied"] == 1
        assert info["shield_boundary_active"] == 1
        assert info["shield_pre_action"] == pytest.approx(-0.8)
        assert info["shield_post_action"] >= 0.0
        assert info["battery_action_applied"] == pytest.approx(info["shield_post_action"])
        assert "inventory_teacher_action" in info
        assert info["inventory_teacher_active"] == 1
        assert info["inventory_teacher_boundary_active"] == 1
        assert "inventory_teacher_reserve_active" in info
        assert info["inventory_teacher_weight"] > 1.0
    finally:
        wrapped.close()


def test_shielded_action_wrapper_can_penalize_shield_mismatch():
    env = NetworkMicrogridEnv(IEEE33Config(simulation_days=1, seed=42, battery_model="simple", regime="base"))
    wrapped = ShieldedActionWrapper(env, hard_pullback_action=0.2, shield_delta_penalty_coef=2.0)
    try:
        wrapped.reset(seed=42)
        env.unwrapped.battery.soc = float(env.unwrapped.config.battery_params.soc_max)
        _, reward, _, _, info = wrapped.step(np.array([-0.8], dtype=np.float32))
        assert info["shield_applied"] == 1
        assert info["shield_delta_penalty"] > 0.0
        assert reward < 0.0
    finally:
        wrapped.close()
