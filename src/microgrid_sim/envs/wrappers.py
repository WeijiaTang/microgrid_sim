"""Environment wrappers used by alternative DRL agents."""

from __future__ import annotations

from itertools import product

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from microgrid_sim.training.shield import BatteryActionShieldConfig, inventory_teacher_action, shield_battery_action


def _normalized_battery_action_from_power(*, power_w: float, charge_limit_w: float, discharge_limit_w: float) -> float:
    power = float(power_w)
    if power >= 0.0:
        scale = max(float(discharge_limit_w), 1e-9)
    else:
        scale = max(float(charge_limit_w), 1e-9)
    return float(np.clip(power / scale, -1.0, 1.0))


def _apply_soc_boundary_bias(
    *,
    desired_power_w: float,
    soc: float,
    soc_min: float,
    soc_max: float,
    charge_limit_w: float,
    discharge_limit_w: float,
    soft_buffer_fraction: float = 0.18,
    hard_buffer_fraction: float = 0.10,
) -> float:
    usable_soc_span = max(float(soc_max) - float(soc_min), 1e-9)
    soft_buffer = min(max(usable_soc_span * float(soft_buffer_fraction), 0.08), usable_soc_span / 2.0)
    hard_buffer = min(max(usable_soc_span * float(hard_buffer_fraction), 0.04), usable_soc_span / 2.0)
    desired = float(desired_power_w)

    if soc <= soc_min + hard_buffer and charge_limit_w > 1e-9:
        return -float(charge_limit_w)
    if soc >= soc_max - hard_buffer and discharge_limit_w > 1e-9:
        return float(discharge_limit_w)

    if soc <= soc_min + soft_buffer and charge_limit_w > 1e-9:
        proximity = np.clip(((soc_min + soft_buffer) - soc) / max(soft_buffer, 1e-9), 0.0, 1.0)
        away_bias_w = float(proximity) * 0.5 * float(charge_limit_w)
        return min(desired, -away_bias_w)
    if soc >= soc_max - soft_buffer and discharge_limit_w > 1e-9:
        proximity = np.clip((soc - (soc_max - soft_buffer)) / max(soft_buffer, 1e-9), 0.0, 1.0)
        away_bias_w = float(proximity) * 0.5 * float(discharge_limit_w)
        return max(desired, away_bias_w)
    return desired


def compute_rule_guidance_action(unwrapped_env, policy_name: str = "rule") -> np.ndarray:
    action_space = getattr(unwrapped_env, "action_space", None)
    action_shape = tuple(getattr(action_space, "shape", tuple()) or (1,))
    action = np.zeros(action_shape, dtype=np.float32)

    config = getattr(unwrapped_env, "config", None)
    battery = getattr(unwrapped_env, "battery", None)
    profiles = getattr(unwrapped_env, "_profiles", None)
    total_steps = int(getattr(unwrapped_env, "total_steps", 0))
    if config is None or battery is None or profiles is None or total_steps <= 0:
        return action

    params = getattr(config, "battery_params", None)
    if params is None:
        return action

    idx = min(int(getattr(unwrapped_env, "current_step", 0)), total_steps - 1)
    price = float(profiles.price[idx])
    load_w = float(profiles.load_w[idx])
    pv_w = float(profiles.pv_w[idx])
    soc = float(getattr(battery, "soc", getattr(params, "soc_init", 0.5)))
    soc_min = float(getattr(params, "soc_min", 0.0))
    soc_max = float(getattr(params, "soc_max", 1.0))
    dt_hours = max(float(getattr(config, "dt_seconds", 3600.0)) / 3600.0, 1e-9)
    current_step = int(getattr(unwrapped_env, "current_step", 0))
    remaining_steps = max(total_steps - current_step, 1)

    rated_charge_limit_w = max(float(getattr(params, "p_charge_max", 0.0)), 0.0)
    rated_discharge_limit_w = max(float(getattr(params, "p_discharge_max", 0.0)), 0.0)
    if hasattr(battery, "power_command_bounds"):
        min_command_w, max_command_w = battery.power_command_bounds(dt=max(float(getattr(config, "dt_seconds", 3600.0)), 1e-9))
        charge_limit_w = max(float(-min_command_w), 0.0)
        discharge_limit_w = max(float(max_command_w), 0.0)
    else:
        charge_limit_w = rated_charge_limit_w
        discharge_limit_w = rated_discharge_limit_w

    reward_cfg = getattr(config, "reward", None)
    valley_price = float(getattr(reward_cfg, "valley_price", 0.39073))
    peak_price = float(getattr(reward_cfg, "peak_price", 0.51373))
    policy_name = str(policy_name).strip().lower() or "rule"

    if policy_name == "rule":
        desired_power_w = 0.0
        if price <= valley_price and soc < min(0.8, soc_max) and charge_limit_w > 1e-9:
            desired_power_w = -charge_limit_w
        elif price >= peak_price and soc > max(0.2, soc_min) and discharge_limit_w > 1e-9:
            desired_power_w = discharge_limit_w
        elif pv_w > 0.0 and soc < soc_max and charge_limit_w > 1e-9:
            desired_power_w = -min(charge_limit_w, pv_w)
    else:
        net_demand_w = float(load_w - pv_w)
        import_limit_w = float(config.grid_import_max) * 1e6 if np.isfinite(float(config.grid_import_max)) else float("inf")
        export_limit_w = float(config.grid_export_max) * 1e6 if np.isfinite(float(config.grid_export_max)) else float("inf")
        desired_power_w = 0.0
        if np.isfinite(import_limit_w) and net_demand_w > import_limit_w and discharge_limit_w > 1e-9:
            desired_power_w = min(net_demand_w - import_limit_w, discharge_limit_w)
        elif np.isfinite(export_limit_w) and net_demand_w < -export_limit_w and charge_limit_w > 1e-9:
            desired_power_w = -min((-net_demand_w) - export_limit_w, charge_limit_w)
        elif price <= valley_price and soc < min(0.75, soc_max) and charge_limit_w > 1e-9:
            desired_power_w = -0.35 * charge_limit_w
        elif price >= peak_price and soc > max(0.25, soc_min) and discharge_limit_w > 1e-9:
            desired_power_w = 0.35 * discharge_limit_w

        if policy_name == "terminal_balanced":
            target_soc = getattr(config, "terminal_soc_target", None)
            if target_soc is None:
                target_soc = getattr(params, "soc_init", 0.5)
            tolerance = max(float(getattr(config, "terminal_soc_tolerance", 0.0)), 0.0)
            target_gap_soc = float(target_soc) - soc
            target_gap_excess = max(abs(target_gap_soc) - tolerance, 0.0)
            remaining_fraction = float(remaining_steps) / max(float(total_steps), 1.0)
            closure_horizon_fraction = 0.35
            if target_gap_excess > 0.0:
                remaining_hours = max(float(remaining_steps) * dt_hours, dt_hours)
                closure_power_w = -(target_gap_soc * float(params.nominal_energy_wh)) / remaining_hours
                closure_power_w = float(np.clip(closure_power_w, -charge_limit_w, discharge_limit_w))
                closure_progress = np.clip(1.0 - remaining_fraction / closure_horizon_fraction, 0.0, 1.0)
                closure_urgency = np.clip(target_gap_excess / 0.20, 0.0, 1.0)
                closure_mix = float(np.clip(max(closure_progress, 0.5 * closure_urgency), 0.0, 1.0))
                desired_power_w = (1.0 - closure_mix) * desired_power_w + closure_mix * closure_power_w
            desired_power_w = _apply_soc_boundary_bias(
                desired_power_w=desired_power_w,
                soc=soc,
                soc_min=soc_min,
                soc_max=soc_max,
                charge_limit_w=charge_limit_w,
                discharge_limit_w=discharge_limit_w,
            )

    if action.size:
        action.reshape(-1)[0] = _normalized_battery_action_from_power(
            power_w=desired_power_w,
            charge_limit_w=max(charge_limit_w, rated_charge_limit_w, 1e-9),
            discharge_limit_w=max(discharge_limit_w, rated_discharge_limit_w, 1e-9),
        )
    return action


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
        battery_infeasible_penalty: float = -1.0,
    ):
        super().__init__(env)
        if not isinstance(env.action_space, spaces.Box):
            raise TypeError("ContinuousActionRegularizationWrapper requires a Box action space")
        if not isinstance(env.observation_space, spaces.Box):
            raise TypeError("ContinuousActionRegularizationWrapper requires a Box observation space")

        self.action_space = env.action_space
        base_low = np.asarray(env.observation_space.low, dtype=np.float32).reshape(-1)
        base_high = np.asarray(env.observation_space.high, dtype=np.float32).reshape(-1)
        self._action_low = np.asarray(self.action_space.low, dtype=np.float32).reshape(-1)
        self._action_high = np.asarray(self.action_space.high, dtype=np.float32).reshape(-1)
        self.observation_space = spaces.Box(
            low=np.concatenate([base_low, self._action_low]).astype(np.float32, copy=False),
            high=np.concatenate([base_high, self._action_high]).astype(np.float32, copy=False),
            dtype=np.float32,
        )
        self.smoothing_coef = float(np.clip(float(smoothing_coef), 0.0, 0.995))
        self.max_delta = max(float(max_delta), 0.0)
        self.rate_penalty = max(float(rate_penalty), 0.0)
        self.symmetric_battery_action = bool(symmetric_battery_action)
        self.battery_feasibility_aware = bool(battery_feasibility_aware)
        self.battery_infeasible_penalty = float(battery_infeasible_penalty)
        self._battery_positive_scale = self._resolve_battery_positive_scale()
        self._prev_applied_action = np.zeros_like(self._action_low, dtype=np.float32)

    def _augment_observation(self, obs) -> np.ndarray:
        obs_array = np.asarray(obs, dtype=np.float32).reshape(-1)
        return np.concatenate([obs_array, self._prev_applied_action.astype(np.float32, copy=False)]).astype(np.float32, copy=False)

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
        min_command_w, max_command_w = battery.power_command_bounds(dt=max(float(getattr(config, "dt_seconds", 3600.0)), 1e-9))
        charge_power_limit = max(float(-min_command_w), 0.0)
        discharge_power_limit = max(float(max_command_w), 0.0)
        p_charge_max = max(float(getattr(params, "p_charge_max", 0.0)), 1e-9)
        p_discharge_max = max(float(getattr(params, "p_discharge_max", 0.0)), 1e-9)
        charge_fraction = float(np.clip(charge_power_limit / p_charge_max, 0.0, 1.0))
        discharge_fraction = float(np.clip(discharge_power_limit / p_discharge_max, 0.0, 1.0))
        action_low = -1.0 if charge_power_limit > 1e-9 else 0.0
        action_high = 1.0 if discharge_power_limit > 1e-9 else 0.0
        return action_low, action_high, charge_fraction, discharge_fraction

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
        config = getattr(self.env.unwrapped, "config", None)
        charge_power_limit = max(float(getattr(getattr(config, "battery_params", None), "p_charge_max", 0.0)), 0.0) if config is not None else 0.0
        discharge_power_limit = max(float(getattr(getattr(config, "battery_params", None), "p_discharge_max", 0.0)), 0.0) if config is not None else 0.0
        infeasible_gap = 0.0
        if self.battery_feasibility_aware and applied.size:
            (
                battery_action_feasible_low,
                battery_action_feasible_high,
                battery_charge_fraction_feasible,
                battery_discharge_fraction_feasible,
            ) = self._battery_feasible_action_bounds()
            charge_power_limit = battery_charge_fraction_feasible * charge_power_limit
            discharge_power_limit = battery_discharge_fraction_feasible * discharge_power_limit
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
            "battery_charge_power_limit": float(charge_power_limit),
            "battery_discharge_power_limit": float(discharge_power_limit),
            "battery_action_infeasible_gap": float(infeasible_gap),
        }
        return raw, applied, delta, diagnostics

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self._battery_positive_scale = self._resolve_battery_positive_scale()
        self._prev_applied_action = np.zeros_like(self._action_low, dtype=np.float32)
        obs, info = self.env.reset(seed=seed, options=options)
        return self._augment_observation(obs), info

    def step(self, action):
        raw, applied, delta, diagnostics = self._regularize_action(action)
        obs, reward, terminated, truncated, info = self.env.step(applied)
        obs = self._augment_observation(obs)
        regularization_penalty = self.rate_penalty * float(np.mean(np.abs(delta))) if self.rate_penalty > 0.0 else 0.0
        reward_adjustment = -regularization_penalty
        infeasible_penalty = 0.0
        if self.battery_infeasible_penalty != 0.0:
            infeasible_penalty = self.battery_infeasible_penalty * float(diagnostics.get("battery_action_infeasible_gap", 0.0))
            reward_adjustment += infeasible_penalty
        if reward_adjustment != 0.0:
            reward = float(reward) + reward_adjustment
        info = dict(info or {})
        shield_post_action = info.get("shield_post_action", None)
        info.update(
            {
                "battery_action_raw": float(raw[0]) if raw.size else 0.0,
                "battery_action_applied": float(shield_post_action) if shield_post_action is not None else (float(applied[0]) if applied.size else 0.0),
                "battery_action_applied_pre_shield": float(applied[0]) if applied.size else 0.0,
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


class ShieldedActionWrapper(gym.Wrapper):
    """Apply a battery-action safety shield before the base environment step."""

    def __init__(
        self,
        env: gym.Env,
        reserve_discharge_min_fraction: float = 0.25,
        soc_soft_buffer_fraction: float = 0.18,
        soc_hard_buffer_fraction: float = 0.10,
        hard_pullback_action: float = 0.25,
        terminal_closure_horizon_fraction: float = 0.35,
        terminal_closure_urgency_soc: float = 0.20,
        shield_delta_penalty_coef: float = 0.0,
        shield_delta_penalty_start: float | None = None,
        shield_delta_penalty_end: float | None = None,
        shield_delta_penalty_warmup_steps: int = 0,
    ):
        super().__init__(env)
        if not isinstance(env.action_space, spaces.Box):
            raise TypeError("ShieldedActionWrapper requires a Box action space")
        if not isinstance(env.observation_space, spaces.Box):
            raise TypeError("ShieldedActionWrapper requires a Box observation space")
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.shield_config = BatteryActionShieldConfig(
            reserve_discharge_min_fraction=float(reserve_discharge_min_fraction),
            soc_soft_buffer_fraction=float(soc_soft_buffer_fraction),
            soc_hard_buffer_fraction=float(soc_hard_buffer_fraction),
            hard_pullback_action=float(hard_pullback_action),
            terminal_closure_horizon_fraction=float(terminal_closure_horizon_fraction),
            terminal_closure_urgency_soc=float(terminal_closure_urgency_soc),
        )
        static_penalty_coef = max(float(shield_delta_penalty_coef), 0.0)
        start_coef = static_penalty_coef if shield_delta_penalty_start is None else float(shield_delta_penalty_start)
        end_coef = static_penalty_coef if shield_delta_penalty_end is None else float(shield_delta_penalty_end)
        self.shield_delta_penalty_start = max(float(start_coef), 0.0)
        self.shield_delta_penalty_end = max(float(end_coef), 0.0)
        self.shield_delta_penalty_warmup_steps = max(int(shield_delta_penalty_warmup_steps), 0)
        self.shield_delta_penalty_coef = self.shield_delta_penalty_end
        self._shield_penalty_step_count = 0

    def _shield_delta_penalty_progress(self) -> float:
        if self.shield_delta_penalty_warmup_steps <= 0:
            return 1.0
        return float(np.clip(self._shield_penalty_step_count / float(self.shield_delta_penalty_warmup_steps), 0.0, 1.0))

    def _current_shield_delta_penalty_coef(self) -> float:
        progress = self._shield_delta_penalty_progress()
        return float(
            (1.0 - progress) * self.shield_delta_penalty_start
            + progress * self.shield_delta_penalty_end
        )

    def reset_shield_penalty_progress(self) -> None:
        """Reset only the curriculum counter; safe execution behavior is unchanged."""
        self._shield_penalty_step_count = 0

    def step(self, action):
        teacher_decision = inventory_teacher_action(
            unwrapped_env=self.env.unwrapped,
            config=self.shield_config,
            action_space_low=np.asarray(self.action_space.low, dtype=np.float32),
            action_space_high=np.asarray(self.action_space.high, dtype=np.float32),
        )
        decision = shield_battery_action(
            action,
            unwrapped_env=self.env.unwrapped,
            config=self.shield_config,
            action_space_low=np.asarray(self.action_space.low, dtype=np.float32),
            action_space_high=np.asarray(self.action_space.high, dtype=np.float32),
        )
        obs, reward, terminated, truncated, info = self.env.step(decision.action)
        shield_delta_penalty_progress = self._shield_delta_penalty_progress()
        shield_delta_penalty_coef_current = self._current_shield_delta_penalty_coef()
        shield_delta_penalty = shield_delta_penalty_coef_current * abs(float(decision.delta))
        if shield_delta_penalty > 0.0:
            reward = float(reward) - float(shield_delta_penalty)
        self._shield_penalty_step_count += 1
        info = dict(info or {})
        info.update(
            {
                "shield_enabled": 1,
                "shield_pre_action": float(decision.pre_action),
                "shield_post_action": float(decision.post_action),
                "shield_delta": float(decision.delta),
                "shield_applied": int(bool(decision.applied)),
                "shield_feasibility_clipped": int(bool(decision.feasibility_clipped)),
                "shield_reserve_active": int(bool(decision.reserve_active)),
                "shield_boundary_active": int(bool(decision.boundary_active)),
                "shield_terminal_active": int(bool(decision.terminal_active)),
                "shield_effective_low": float(decision.feasible_low),
                "shield_effective_high": float(decision.feasible_high),
                "shield_closure_action": float(decision.closure_action),
                "shield_closure_mix": float(decision.closure_mix),
                "shield_delta_penalty": float(shield_delta_penalty),
                "shield_delta_penalty_coef_current": float(shield_delta_penalty_coef_current),
                "shield_delta_penalty_progress": float(shield_delta_penalty_progress),
                "shield_delta_penalty_start": float(self.shield_delta_penalty_start),
                "shield_delta_penalty_end": float(self.shield_delta_penalty_end),
                "shield_delta_penalty_warmup_steps": int(self.shield_delta_penalty_warmup_steps),
                "battery_action_applied": float(decision.post_action),
                "inventory_teacher_action": float(teacher_decision.action.reshape(-1)[0]) if teacher_decision.action.size else 0.0,
                "inventory_teacher_active": int(bool(teacher_decision.active)),
                "inventory_teacher_boundary_active": int(bool(teacher_decision.boundary_active)),
                "inventory_teacher_terminal_active": int(bool(teacher_decision.terminal_active)),
                "inventory_teacher_reserve_active": int(bool(teacher_decision.reserve_active)),
                "inventory_teacher_peak_value_active": int(bool(teacher_decision.peak_value_active)),
                "inventory_teacher_valley_value_active": int(bool(teacher_decision.valley_value_active)),
                "inventory_teacher_weight": float(teacher_decision.weight),
            }
        )
        return obs, reward, terminated, truncated, info


class RuleGuidedActionWrapper(gym.Wrapper):
    """Blend policy actions with a simple rule-based hint during early training."""

    def __init__(
        self,
        env: gym.Env,
        guidance_mix: float = 0.0,
        guidance_decay_steps: int = 0,
        guidance_policy: str = "rule",
        guidance_enabled: bool = True,
    ):
        super().__init__(env)
        if not isinstance(env.action_space, spaces.Box):
            raise TypeError("RuleGuidedActionWrapper requires a Box action space")
        if not isinstance(env.observation_space, spaces.Box):
            raise TypeError("RuleGuidedActionWrapper requires a Box observation space")
        self.action_space = env.action_space
        base_low = np.asarray(env.observation_space.low, dtype=np.float32).reshape(-1)
        base_high = np.asarray(env.observation_space.high, dtype=np.float32).reshape(-1)
        self.observation_space = spaces.Box(
            low=np.concatenate([base_low, np.asarray([0.0], dtype=np.float32)]),
            high=np.concatenate([base_high, np.asarray([1.0], dtype=np.float32)]),
            dtype=np.float32,
        )
        self.guidance_mix = float(np.clip(float(guidance_mix), 0.0, 1.0))
        self.guidance_decay_steps = max(int(guidance_decay_steps), 0)
        self.guidance_policy = str(guidance_policy).strip().lower() or "rule"
        self.guidance_enabled = bool(guidance_enabled)
        self._step_count = 0

    def _augment_observation(self, obs, *, mix: float) -> np.ndarray:
        obs_array = np.asarray(obs, dtype=np.float32).reshape(-1)
        return np.concatenate([obs_array, np.asarray([float(np.clip(mix, 0.0, 1.0))], dtype=np.float32)]).astype(np.float32, copy=False)

    def reset_guidance_progress(self) -> None:
        self._step_count = 0

    def _rule_based_action(self) -> np.ndarray:
        return compute_rule_guidance_action(self.env.unwrapped, self.guidance_policy)

    def _current_mix(self) -> float:
        if not self.guidance_enabled or self.guidance_mix <= 0.0:
            return 0.0
        if self.guidance_decay_steps <= 0:
            return self.guidance_mix
        remaining = max(0.0, 1.0 - float(self._step_count) / float(self.guidance_decay_steps))
        return float(self.guidance_mix * remaining)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        mix = float(self._current_mix())
        obs = self._augment_observation(obs, mix=mix)
        info = dict(info or {})
        info.setdefault("rule_guidance_mix", mix)
        info.setdefault("rule_guided_action", 0.0)
        return obs, info

    def step(self, action):
        raw = np.asarray(action, dtype=np.float32).reshape(self.action_space.shape)
        clipped = np.clip(raw, self.action_space.low, self.action_space.high).astype(np.float32, copy=False)
        rule_action = self._rule_based_action()
        mix = float(self._current_mix())
        if mix > 0.0:
            applied = (1.0 - mix) * clipped + mix * rule_action
        else:
            applied = clipped
        applied = np.clip(applied, self.action_space.low, self.action_space.high).astype(np.float32, copy=False)
        obs, reward, terminated, truncated, info = self.env.step(applied)
        if self.guidance_enabled:
            self._step_count += 1
        obs = self._augment_observation(obs, mix=mix)
        info = dict(info or {})
        info.update(
            {
                "policy_action_pre_guidance": float(clipped.reshape(-1)[0]) if clipped.size else 0.0,
                "rule_guided_action": float(rule_action.reshape(-1)[0]) if rule_action.size else 0.0,
                "rule_guidance_mix": float(mix),
                "action_after_rule_guidance": float(applied.reshape(-1)[0]) if applied.size else 0.0,
                "rule_based_action_hint": float(rule_action.reshape(-1)[0]) if rule_action.size else 0.0,
            }
        )
        return obs, reward, terminated, truncated, info
