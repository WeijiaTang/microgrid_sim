from __future__ import annotations

import numpy as np

from microgrid_sim.cases import IEEE33Config
from microgrid_sim.envs.network_microgrid import NetworkMicrogridEnv
from microgrid_sim.envs.wrappers import ContinuousActionRegularizationWrapper, RuleGuidedActionWrapper


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
