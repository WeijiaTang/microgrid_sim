"""Environment wrappers used by alternative DRL agents."""

from __future__ import annotations

from itertools import product

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class DiscreteActionWrapper(gym.ActionWrapper):
    """Map a continuous Box action space onto a fixed discrete action table."""

    def __init__(self, env: gym.Env, action_bins: int = 21, max_actions: int = 2048):
        super().__init__(env)
        if not isinstance(env.action_space, spaces.Box):
            raise TypeError("DiscreteActionWrapper requires a Box action space")

        low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
        self.action_bins = max(int(action_bins), 3)
        action_count = int(self.action_bins ** max(low.size, 1))
        if action_count > int(max_actions):
            raise ValueError(
                f"Discrete action table would contain {action_count} actions; "
                f"reduce action_bins or increase max_actions."
            )

        per_dim_values = [
            np.linspace(float(dim_low), float(dim_high), num=self.action_bins, dtype=np.float32)
            for dim_low, dim_high in zip(low, high)
        ]
        self.action_lookup = np.asarray(list(product(*per_dim_values)), dtype=np.float32)
        if self.action_lookup.ndim == 1:
            self.action_lookup = self.action_lookup.reshape(-1, 1)
        self.action_space = spaces.Discrete(int(len(self.action_lookup)))

    def action(self, action):
        index = int(np.clip(int(action), 0, len(self.action_lookup) - 1))
        return self.action_lookup[index].copy()


class ContinuousActionRegularizationWrapper(gym.Wrapper):
    """Apply smoothing and slew-rate limits to continuous control actions."""

    def __init__(
        self,
        env: gym.Env,
        smoothing_coef: float = 0.0,
        max_delta: float = 0.0,
        rate_penalty: float = 0.0,
        symmetric_battery_action: bool = False,
        battery_feasibility_aware: bool = False,
        battery_infeasible_penalty: float = 0.0,
    ):
        super().__init__(env)
        if not isinstance(env.action_space, spaces.Box):
            raise TypeError("ContinuousActionRegularizationWrapper requires a Box action space")

        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self._action_low = np.asarray(self.action_space.low, dtype=np.float32).reshape(-1)
        self._action_high = np.asarray(self.action_space.high, dtype=np.float32).reshape(-1)
        self.smoothing_coef = float(np.clip(float(smoothing_coef), 0.0, 0.995))
        self.max_delta = max(float(max_delta), 0.0)
        self.rate_penalty = max(float(rate_penalty), 0.0)
        self.symmetric_battery_action = bool(symmetric_battery_action)
        self.battery_feasibility_aware = bool(battery_feasibility_aware)
        self.battery_infeasible_penalty = max(float(battery_infeasible_penalty), 0.0)
        self._battery_positive_scale = self._resolve_battery_positive_scale()
        self._prev_applied_action = np.zeros_like(self._action_low, dtype=np.float32)

    def _resolve_battery_positive_scale(self) -> float:
        if not self.symmetric_battery_action:
            return 1.0
        battery = getattr(self.env.unwrapped, "battery", None)
        params = getattr(battery, "params", None)
        if params is None:
            params = getattr(getattr(self.env.unwrapped, "config", None), "battery_params", None)
        if params is None:
            return 1.0
        discharge_limit = max(float(getattr(params, "p_discharge_max", 0.0)), 0.0)
        charge_limit = max(float(getattr(params, "p_charge_max", 0.0)), 0.0)
        if discharge_limit <= 0.0 or charge_limit <= 0.0:
            return 1.0
        return float(np.clip(min(discharge_limit, charge_limit) / discharge_limit, 0.0, 1.0))

    def _battery_feasible_action_bounds(self) -> tuple[float, float, float, float]:
        battery = getattr(self.env.unwrapped, "battery", None)
        params = getattr(battery, "params", None)
        config = getattr(self.env.unwrapped, "config", None)
        if battery is None or params is None or config is None:
            return -1.0, 1.0, 1.0, 1.0

        soc = float(getattr(battery, "soc", getattr(params, "soc_init", 0.5)))
        soc_min = float(getattr(params, "soc_min", 0.0))
        soc_max = float(getattr(params, "soc_max", 1.0))
        nominal_energy_wh = max(float(getattr(params, "nominal_energy_wh", 0.0)), 1e-9)
        dt = max(float(getattr(config, "dt_seconds", 3600.0)), 1e-9)
        p_charge_max = max(float(getattr(params, "p_charge_max", 0.0)), 1e-9)
        p_discharge_max = max(float(getattr(params, "p_discharge_max", 0.0)), 1e-9)
        eta_charge = max(float(getattr(params, "eta_charge", 1.0)), 1e-9)
        eta_discharge = max(float(getattr(params, "eta_discharge", 1.0)), 0.0)

        available_charge_wh = max((soc_max - soc) * nominal_energy_wh, 0.0)
        available_discharge_wh = max((soc - soc_min) * nominal_energy_wh, 0.0)
        charge_power_limit = available_charge_wh * 3600.0 / dt / eta_charge
        discharge_power_limit = available_discharge_wh * 3600.0 / dt * eta_discharge

        charge_fraction = float(np.clip(charge_power_limit / p_charge_max, 0.0, 1.0))
        discharge_fraction = float(np.clip(discharge_power_limit / p_discharge_max, 0.0, 1.0))
        return -charge_fraction, discharge_fraction, charge_fraction, discharge_fraction

    def _regularize_action(self, action) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
        raw = np.asarray(action, dtype=np.float32).reshape(-1)
        if raw.size != self._action_low.size:
            raw = np.resize(raw, self._action_low.shape)
        raw = np.clip(raw, self._action_low, self._action_high)
        target = raw.copy()
        if self.symmetric_battery_action and target.size:
            target[0] = target[0] * self._battery_positive_scale if target[0] > 0.0 else target[0]
        if self.max_delta > 0.0:
            target = np.clip(
                target,
                self._prev_applied_action - self.max_delta,
                self._prev_applied_action + self.max_delta,
            )
        if self.smoothing_coef > 0.0:
            applied = self.smoothing_coef * self._prev_applied_action + (1.0 - self.smoothing_coef) * target
        else:
            applied = target
        battery_action_feasible_low = float(self._action_low[0]) if self._action_low.size else -1.0
        battery_action_feasible_high = float(self._action_high[0]) if self._action_high.size else 1.0
        battery_charge_fraction_feasible = 1.0
        battery_discharge_fraction_feasible = 1.0
        infeasible_gap = 0.0
        if self.battery_feasibility_aware and applied.size:
            (
                battery_action_feasible_low,
                battery_action_feasible_high,
                battery_charge_fraction_feasible,
                battery_discharge_fraction_feasible,
            ) = self._battery_feasible_action_bounds()
            unclipped_battery_action = float(applied[0])
            applied[0] = np.clip(unclipped_battery_action, battery_action_feasible_low, battery_action_feasible_high)
            infeasible_gap = abs(unclipped_battery_action - float(applied[0]))
        applied = np.clip(applied, self._action_low, self._action_high).astype(np.float32, copy=False)
        delta = applied - self._prev_applied_action
        self._prev_applied_action = applied.copy()
        diagnostics = {
            "battery_action_feasible_low": float(battery_action_feasible_low),
            "battery_action_feasible_high": float(battery_action_feasible_high),
            "battery_charge_fraction_feasible": float(battery_charge_fraction_feasible),
            "battery_discharge_fraction_feasible": float(battery_discharge_fraction_feasible),
            "battery_action_infeasible_gap": float(infeasible_gap),
        }
        return raw, applied, delta, diagnostics

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self._battery_positive_scale = self._resolve_battery_positive_scale()
        self._prev_applied_action = np.zeros_like(self._action_low, dtype=np.float32)
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        raw, applied, delta, diagnostics = self._regularize_action(action)
        obs, reward, terminated, truncated, info = self.env.step(applied)
        regularization_penalty = self.rate_penalty * float(np.mean(np.abs(delta))) if self.rate_penalty > 0.0 else 0.0
        infeasible_penalty = 0.0
        if self.battery_infeasible_penalty > 0.0:
            infeasible_penalty = self.battery_infeasible_penalty * float(diagnostics.get("battery_action_infeasible_gap", 0.0))
        total_penalty = regularization_penalty + infeasible_penalty
        if total_penalty > 0.0:
            reward = float(reward) - total_penalty
        info = dict(info or {})
        info.update(
            {
                "battery_action_raw": float(raw[0]) if raw.size else 0.0,
                "battery_action_applied": float(applied[0]) if applied.size else 0.0,
                "battery_action_delta": float(delta[0]) if delta.size else 0.0,
                "action_rate_penalty": float(regularization_penalty),
                "battery_action_infeasible_penalty": float(infeasible_penalty),
                "symmetric_battery_action_scale": float(self._battery_positive_scale),
            }
        )
        info.update(diagnostics)
        if raw.size >= 2:
            info["generator_action_raw"] = float(raw[1])
            info["generator_action_applied"] = float(applied[1])
            info["generator_action_delta"] = float(delta[1])
        return obs, reward, terminated, truncated, info
