# Ref: docs/spec/task.md (Task-ID: SPEC-SAFE-OFFLINE-RL-001)

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BatteryActionShieldConfig:
    reserve_discharge_min_fraction: float = 0.25
    soc_soft_buffer_fraction: float = 0.18
    soc_hard_buffer_fraction: float = 0.10
    hard_pullback_action: float = 0.25
    terminal_closure_horizon_fraction: float = 0.35
    terminal_closure_urgency_soc: float = 0.20


@dataclass(frozen=True)
class BatteryActionShieldDecision:
    action: np.ndarray
    feasible_low: float
    feasible_high: float
    pre_action: float
    post_action: float
    delta: float
    applied: bool
    feasibility_clipped: bool
    reserve_active: bool
    boundary_active: bool
    terminal_active: bool
    closure_action: float
    closure_mix: float


@dataclass(frozen=True)
class InventoryTeacherDecision:
    action: np.ndarray
    active: bool
    boundary_active: bool
    terminal_active: bool
    reserve_active: bool
    peak_value_active: bool
    valley_value_active: bool
    weight: float


def _normalized_battery_action_from_power(*, power_w: float, charge_limit_w: float, discharge_limit_w: float) -> float:
    power = float(power_w)
    if power >= 0.0:
        scale = max(float(discharge_limit_w), 1e-9)
    else:
        scale = max(float(charge_limit_w), 1e-9)
    return float(np.clip(power / scale, -1.0, 1.0))


def _resolve_terminal_target_soc(unwrapped_env, soc: float) -> tuple[float, float]:
    config_obj = getattr(unwrapped_env, "config", None)
    battery = getattr(unwrapped_env, "battery", None)
    params = getattr(battery, "params", None)
    if params is None and config_obj is not None:
        params = getattr(config_obj, "battery_params", None)
    if config_obj is None or params is None:
        return float(soc), 0.0
    target_soc = getattr(config_obj, "terminal_soc_target", None)
    if target_soc is None:
        target_soc = getattr(params, "soc_init", soc)
    tolerance = max(float(getattr(config_obj, "terminal_soc_tolerance", 0.0)), 0.0)
    return float(target_soc), float(tolerance)


def _apply_inventory_boundary_bias(
    *,
    desired_power_w: float,
    soc: float,
    soc_min: float,
    soc_max: float,
    charge_limit_w: float,
    discharge_limit_w: float,
    soft_buffer: float,
    hard_buffer: float,
) -> float:
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


def _battery_feasible_action_bounds(unwrapped_env) -> tuple[float, float, float, float, float, float]:
    battery = getattr(unwrapped_env, "battery", None)
    params = getattr(battery, "params", None)
    config = getattr(unwrapped_env, "config", None)
    if battery is None or params is None or config is None:
        return -1.0, 1.0, 1.0, 1.0, 0.0, 0.0
    min_command_w, max_command_w = battery.power_command_bounds(dt=max(float(getattr(config, "dt_seconds", 3600.0)), 1e-9))
    charge_power_limit = max(float(-min_command_w), 0.0)
    discharge_power_limit = max(float(max_command_w), 0.0)
    p_charge_max = max(float(getattr(params, "p_charge_max", 0.0)), 1e-9)
    p_discharge_max = max(float(getattr(params, "p_discharge_max", 0.0)), 1e-9)
    charge_fraction = float(np.clip(charge_power_limit / p_charge_max, 0.0, 1.0))
    discharge_fraction = float(np.clip(discharge_power_limit / p_discharge_max, 0.0, 1.0))
    action_low = -1.0 if charge_power_limit > 1e-9 else 0.0
    action_high = 1.0 if discharge_power_limit > 1e-9 else 0.0
    return action_low, action_high, charge_fraction, discharge_fraction, charge_power_limit, discharge_power_limit


def shield_battery_action(
    action,
    *,
    unwrapped_env,
    config: BatteryActionShieldConfig,
    action_space_low: np.ndarray,
    action_space_high: np.ndarray,
) -> BatteryActionShieldDecision:
    raw = np.asarray(action, dtype=np.float32).reshape(-1)
    if raw.size == 0:
        return BatteryActionShieldDecision(
            action=np.asarray([], dtype=np.float32),
            feasible_low=-1.0,
            feasible_high=1.0,
            pre_action=0.0,
            post_action=0.0,
            delta=0.0,
            applied=False,
            feasibility_clipped=False,
            reserve_active=False,
            boundary_active=False,
            terminal_active=False,
            closure_action=0.0,
            closure_mix=0.0,
        )

    adjusted = np.clip(raw.copy(), np.asarray(action_space_low, dtype=np.float32).reshape(-1), np.asarray(action_space_high, dtype=np.float32).reshape(-1))
    pre_action = float(adjusted[0])

    feasible_low, feasible_high, _, discharge_fraction, charge_limit_w, discharge_limit_w = _battery_feasible_action_bounds(unwrapped_env)
    allowed_low = float(max(feasible_low, float(np.asarray(action_space_low).reshape(-1)[0])))
    allowed_high = float(min(feasible_high, float(np.asarray(action_space_high).reshape(-1)[0])))

    feasibility_clipped = False
    reserve_active = False
    boundary_active = False
    terminal_active = False
    closure_action = float(pre_action)
    closure_mix = 0.0

    config_obj = getattr(unwrapped_env, "config", None)
    battery = getattr(unwrapped_env, "battery", None)
    profiles = getattr(unwrapped_env, "_profiles", None)
    total_steps = max(int(getattr(unwrapped_env, "total_steps", 0)), 1)
    current_step = min(int(getattr(unwrapped_env, "current_step", 0)), total_steps - 1)
    params = getattr(config_obj, "battery_params", None)
    reward_cfg = getattr(config_obj, "reward", None)
    if config_obj is not None and battery is not None and params is not None and profiles is not None:
        price = float(profiles.price[current_step])
        peak_price = float(getattr(reward_cfg, "peak_price", 0.51373))
        soc = float(getattr(battery, "soc", getattr(params, "soc_init", 0.5)))
        soc_min = float(getattr(params, "soc_min", 0.0))
        soc_max = float(getattr(params, "soc_max", 1.0))
        usable_soc_span = max(soc_max - soc_min, 1e-9)

        adjusted[0] = float(np.clip(adjusted[0], allowed_low, allowed_high))
        feasibility_clipped = bool(abs(adjusted[0] - pre_action) > 1e-9)

        reserve_floor = float(np.clip(float(config.reserve_discharge_min_fraction), 0.0, 1.0))
        if price >= peak_price and discharge_fraction <= reserve_floor + 1e-9 and adjusted[0] > 0.0:
            adjusted[0] = 0.0
            allowed_high = min(allowed_high, 0.0)
            reserve_active = True

        soft_buffer = min(max(usable_soc_span * float(config.soc_soft_buffer_fraction), 0.0), usable_soc_span / 2.0)
        hard_buffer = min(max(usable_soc_span * float(config.soc_hard_buffer_fraction), 0.0), usable_soc_span / 2.0)
        hard_pullback_action = float(np.clip(float(config.hard_pullback_action), 0.0, 1.0))

        if soc <= soc_min + soft_buffer and adjusted[0] > 0.0:
            adjusted[0] = min(adjusted[0], 0.0)
            allowed_high = min(allowed_high, 0.0)
            boundary_active = True
        if soc >= soc_max - soft_buffer and adjusted[0] < 0.0:
            adjusted[0] = max(adjusted[0], 0.0)
            allowed_low = max(allowed_low, 0.0)
            boundary_active = True
        if hard_pullback_action > 0.0 and soc <= soc_min + hard_buffer:
            adjusted[0] = min(adjusted[0], -hard_pullback_action if allowed_low < 0.0 else 0.0)
            boundary_active = True
        if hard_pullback_action > 0.0 and soc >= soc_max - hard_buffer:
            adjusted[0] = max(adjusted[0], hard_pullback_action if allowed_high > 0.0 else 0.0)
            boundary_active = True

        target_soc = getattr(config_obj, "terminal_soc_target", None)
        if target_soc is None:
            target_soc = getattr(params, "soc_init", soc)
        tolerance = max(float(getattr(config_obj, "terminal_soc_tolerance", 0.0)), 0.0)
        target_gap_soc = float(target_soc) - soc
        target_gap_excess = max(abs(target_gap_soc) - tolerance, 0.0)
        remaining_steps = max(total_steps - current_step, 1)
        remaining_fraction = float(remaining_steps) / float(max(total_steps, 1))
        closure_horizon_fraction = float(np.clip(float(config.terminal_closure_horizon_fraction), 0.0, 1.0))
        within_closure_horizon = remaining_fraction <= closure_horizon_fraction + 1e-9
        if closure_horizon_fraction > 0.0 and target_gap_excess > 0.0 and within_closure_horizon:
            dt_hours = max(float(getattr(config_obj, "dt_seconds", 3600.0)) / 3600.0, 1e-9)
            remaining_hours = max(float(remaining_steps) * dt_hours, dt_hours)
            closure_power_w = -(target_gap_soc * float(getattr(params, "nominal_energy_wh", 0.0))) / remaining_hours
            closure_power_w = float(np.clip(closure_power_w, -charge_limit_w, discharge_limit_w))
            closure_action = _normalized_battery_action_from_power(
                power_w=closure_power_w,
                charge_limit_w=max(charge_limit_w, 1e-9),
                discharge_limit_w=max(discharge_limit_w, 1e-9),
            )
            closure_action = float(np.clip(closure_action, allowed_low, allowed_high))
            closure_progress = np.clip(1.0 - remaining_fraction / max(closure_horizon_fraction, 1e-9), 0.0, 1.0)
            closure_urgency = np.clip(target_gap_excess / max(float(config.terminal_closure_urgency_soc), 1e-9), 0.0, 1.0)
            closure_mix = float(np.clip(max(closure_progress, 0.5 * closure_urgency), 0.0, 1.0))
            if closure_mix > 0.0:
                adjusted[0] = float(np.clip((1.0 - closure_mix) * adjusted[0] + closure_mix * closure_action, allowed_low, allowed_high))
                terminal_active = True

    adjusted[0] = float(np.clip(adjusted[0], allowed_low, allowed_high))
    post_action = float(adjusted[0])
    delta = float(post_action - pre_action)
    return BatteryActionShieldDecision(
        action=adjusted.astype(np.float32, copy=False),
        feasible_low=float(allowed_low),
        feasible_high=float(allowed_high),
        pre_action=pre_action,
        post_action=post_action,
        delta=delta,
        applied=bool(abs(delta) > 1e-9),
        feasibility_clipped=bool(feasibility_clipped),
        reserve_active=bool(reserve_active),
        boundary_active=bool(boundary_active),
        terminal_active=bool(terminal_active),
        closure_action=float(closure_action),
        closure_mix=float(closure_mix),
    )


def inventory_teacher_action(
    *,
    unwrapped_env,
    config: BatteryActionShieldConfig,
    action_space_low: np.ndarray,
    action_space_high: np.ndarray,
) -> InventoryTeacherDecision:
    feasible_low, feasible_high, _, discharge_fraction, charge_limit_w, discharge_limit_w = _battery_feasible_action_bounds(
        unwrapped_env
    )
    action = np.zeros((1,), dtype=np.float32)
    config_obj = getattr(unwrapped_env, "config", None)
    battery = getattr(unwrapped_env, "battery", None)
    profiles = getattr(unwrapped_env, "_profiles", None)
    params = getattr(config_obj, "battery_params", None) if config_obj is not None else None
    total_steps = max(int(getattr(unwrapped_env, "total_steps", 0)), 1)
    current_step = min(int(getattr(unwrapped_env, "current_step", 0)), total_steps - 1)
    if config_obj is None or battery is None or params is None or profiles is None:
        return InventoryTeacherDecision(
            action=action,
            active=False,
            boundary_active=False,
            terminal_active=False,
            reserve_active=False,
            peak_value_active=False,
            valley_value_active=False,
            weight=0.0,
        )

    reward_cfg = getattr(config_obj, "reward", None)
    price = float(profiles.price[current_step])
    valley_price = float(getattr(reward_cfg, "valley_price", 0.39073))
    peak_price = float(getattr(reward_cfg, "peak_price", 0.51373))
    soc = float(getattr(battery, "soc", getattr(params, "soc_init", 0.5)))
    soc_min = float(getattr(params, "soc_min", 0.0))
    soc_max = float(getattr(params, "soc_max", 1.0))
    usable_soc_span = max(soc_max - soc_min, 1e-9)
    soft_buffer = min(max(usable_soc_span * float(config.soc_soft_buffer_fraction), 0.0), usable_soc_span / 2.0)
    hard_buffer = min(max(usable_soc_span * float(config.soc_hard_buffer_fraction), 0.0), usable_soc_span / 2.0)
    target_soc, tolerance = _resolve_terminal_target_soc(unwrapped_env, soc)

    midband_half_width = min(max(0.18 * usable_soc_span, tolerance + 0.04), usable_soc_span / 2.0)
    midband_low = float(np.clip(target_soc - midband_half_width, soc_min + hard_buffer, soc_max - hard_buffer))
    midband_high = float(np.clip(target_soc + midband_half_width, soc_min + hard_buffer, soc_max - hard_buffer))
    charge_inventory_target = float(np.clip(max(target_soc, soc_min + 0.62 * usable_soc_span), midband_low, soc_max - hard_buffer))
    discharge_inventory_target = float(
        np.clip(min(target_soc, soc_min + 0.42 * usable_soc_span), soc_min + hard_buffer, midband_high)
    )

    boundary_active = False
    terminal_active = False
    reserve_active = False
    peak_value_active = False
    valley_value_active = False
    desired_power_w = 0.0
    value_activation_margin = min(max(0.08 * usable_soc_span, tolerance + 0.02), usable_soc_span / 3.0)
    peak_value_material_soc = float(np.clip(target_soc + value_activation_margin, soc_min + hard_buffer, soc_max))
    valley_value_material_soc = float(np.clip(target_soc - value_activation_margin, soc_min, soc_max - hard_buffer))

    if charge_limit_w > 1e-9 and soc <= soc_min + hard_buffer:
        desired_power_w = -charge_limit_w
        boundary_active = True
    elif discharge_limit_w > 1e-9 and soc >= soc_max - hard_buffer:
        desired_power_w = discharge_limit_w
        boundary_active = True
    elif price <= valley_price and charge_limit_w > 1e-9 and soc < charge_inventory_target:
        inventory_gap = np.clip((charge_inventory_target - soc) / usable_soc_span, 0.0, 1.0)
        desired_power_w = -charge_limit_w * float(np.clip(max(inventory_gap, 0.25), 0.0, 1.0))
        valley_value_active = bool(desired_power_w < -1e-9 and soc < valley_value_material_soc - 1e-9)
    elif price >= peak_price and discharge_limit_w > 1e-9 and soc > discharge_inventory_target:
        inventory_gap = np.clip((soc - discharge_inventory_target) / usable_soc_span, 0.0, 1.0)
        desired_power_w = discharge_limit_w * float(np.clip(max(inventory_gap, 0.25), 0.0, 1.0))
        peak_value_active = bool(desired_power_w > 1e-9 and soc > peak_value_material_soc + 1e-9)
    elif charge_limit_w > 1e-9 and soc < midband_low:
        inventory_gap = np.clip((midband_low - soc) / usable_soc_span, 0.0, 1.0)
        desired_power_w = -charge_limit_w * float(np.clip(max(inventory_gap, 0.15), 0.0, 0.6))
    elif discharge_limit_w > 1e-9 and soc > midband_high:
        inventory_gap = np.clip((soc - midband_high) / usable_soc_span, 0.0, 1.0)
        desired_power_w = discharge_limit_w * float(np.clip(max(inventory_gap, 0.15), 0.0, 0.6))

    reserve_floor = float(np.clip(float(config.reserve_discharge_min_fraction), 0.0, 1.0))
    if price >= peak_price and discharge_fraction <= reserve_floor + 1e-9:
        reserve_active = True
        reserve_target_soc = float(np.clip(midband_low, soc_min + hard_buffer, soc_max - hard_buffer))
        if charge_limit_w > 1e-9 and soc < reserve_target_soc:
            reserve_gap = np.clip((reserve_target_soc - soc) / usable_soc_span, 0.0, 1.0)
            desired_power_w = min(desired_power_w, -charge_limit_w * float(np.clip(max(reserve_gap, 0.4), 0.0, 1.0)))
        else:
            desired_power_w = min(desired_power_w, 0.0)

    target_gap_soc = float(target_soc) - soc
    target_gap_excess = max(abs(target_gap_soc) - tolerance, 0.0)
    remaining_steps = max(total_steps - current_step, 1)
    remaining_fraction = float(remaining_steps) / float(max(total_steps, 1))
    closure_horizon_fraction = float(np.clip(float(config.terminal_closure_horizon_fraction), 0.0, 1.0))
    within_closure_horizon = remaining_fraction <= closure_horizon_fraction + 1e-9
    if closure_horizon_fraction > 0.0 and target_gap_excess > 0.0 and within_closure_horizon:
        dt_hours = max(float(getattr(config_obj, "dt_seconds", 3600.0)) / 3600.0, 1e-9)
        remaining_hours = max(float(remaining_steps) * dt_hours, dt_hours)
        closure_power_w = -(target_gap_soc * float(getattr(params, "nominal_energy_wh", 0.0))) / remaining_hours
        closure_power_w = float(np.clip(closure_power_w, -charge_limit_w, discharge_limit_w))
        closure_progress = np.clip(1.0 - remaining_fraction / max(closure_horizon_fraction, 1e-9), 0.0, 1.0)
        closure_urgency = np.clip(target_gap_excess / max(float(config.terminal_closure_urgency_soc), 1e-9), 0.0, 1.0)
        closure_mix = float(np.clip(max(closure_progress, 0.5 * closure_urgency), 0.0, 1.0))
        if closure_mix > 0.0:
            desired_power_w = (1.0 - closure_mix) * desired_power_w + closure_mix * closure_power_w
            terminal_active = True

    desired_power_w = _apply_inventory_boundary_bias(
        desired_power_w=desired_power_w,
        soc=soc,
        soc_min=soc_min,
        soc_max=soc_max,
        charge_limit_w=charge_limit_w,
        discharge_limit_w=discharge_limit_w,
        soft_buffer=soft_buffer,
        hard_buffer=hard_buffer,
    )
    normalized_action = _normalized_battery_action_from_power(
        power_w=desired_power_w,
        charge_limit_w=max(charge_limit_w, 1e-9),
        discharge_limit_w=max(discharge_limit_w, 1e-9),
    )
    normalized_action = float(np.clip(normalized_action, feasible_low, feasible_high))
    normalized_action = float(
        np.clip(
            normalized_action,
            float(np.asarray(action_space_low, dtype=np.float32).reshape(-1)[0]),
            float(np.asarray(action_space_high, dtype=np.float32).reshape(-1)[0]),
        )
    )
    action[0] = normalized_action

    off_midband = bool(soc < midband_low - 1e-9 or soc > midband_high + 1e-9)
    strong_inventory_push = bool(abs(normalized_action) >= 0.10)
    economically_material_push = bool((peak_value_active or valley_value_active) and strong_inventory_push)
    active = bool(
        boundary_active
        or terminal_active
        or reserve_active
        or (off_midband and strong_inventory_push)
        or economically_material_push
    )
    weight = 0.0
    if active:
        weight = 1.0 + abs(float(normalized_action))
        if boundary_active:
            weight += 0.75
        if terminal_active:
            weight += 0.75
        if reserve_active:
            weight += 0.5
        if off_midband:
            weight += 0.25
        if peak_value_active:
            weight += 0.5
        if valley_value_active:
            weight += 0.25
    return InventoryTeacherDecision(
        action=action.copy(),
        active=bool(active),
        boundary_active=bool(boundary_active),
        terminal_active=bool(terminal_active),
        reserve_active=bool(reserve_active),
        peak_value_active=bool(peak_value_active),
        valley_value_active=bool(valley_value_active),
        weight=float(weight),
    )
