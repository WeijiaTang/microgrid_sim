from __future__ import annotations

import numpy as np

from microgrid_sim.cases import IEEE33ModifiedConfig
from microgrid_sim.envs.network_microgrid import NetworkMicrogridEnv
from microgrid_sim.envs.wrappers import ContinuousActionRegularizationWrapper


def test_continuous_action_regularization_wrapper_clips_to_soc_feasible_range():
    env = NetworkMicrogridEnv(IEEE33ModifiedConfig(simulation_days=1, seed=42, battery_model="simple", regime="base"))
    wrapped = ContinuousActionRegularizationWrapper(
        env,
        battery_feasibility_aware=True,
        battery_infeasible_penalty=2.0,
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
        assert info["battery_action_infeasible_penalty"] == 2.0
    finally:
        wrapped.close()
