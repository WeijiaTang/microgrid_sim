"""CIGRE component decomposition with startup/shutdown logic and wind availability."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DispatchableUnitParams:
    name: str
    p_min_w: float
    p_max_w: float
    ramp_up_w: float
    ramp_down_w: float
    min_up_steps: int
    min_down_steps: int
    start_threshold_w: float
    cost_a0: float = 0.0
    cost_a1: float = 0.0
    cost_a2: float = 0.0
    startup_cost: float = 0.0
    low_load_threshold_fraction: float = 0.0
    low_load_cost_scale: float = 0.0
    low_load_cost_exponent: float = 2.0


class DispatchableUnit:
    def __init__(self, params: DispatchableUnitParams):
        self.params = params
        self.online = False
        self.power_w = 0.0
        self.up_steps = 0
        self.down_steps = params.min_down_steps
        self.start_count = 0

    def reset(self) -> None:
        self.online = False
        self.power_w = 0.0
        self.up_steps = 0
        self.down_steps = self.params.min_down_steps
        self.start_count = 0

    def _load_ratio(self) -> float:
        if self.params.p_max_w <= 0.0:
            return 0.0
        return float(np.clip(self.power_w / self.params.p_max_w, 0.0, 1.0))

    def _low_load_multiplier(self) -> float:
        threshold = float(np.clip(self.params.low_load_threshold_fraction, 0.0, 1.0))
        if self.power_w <= 0.0 or threshold <= 0.0:
            return 1.0
        load_ratio = self._load_ratio()
        if load_ratio >= threshold:
            return 1.0
        normalized_gap = (threshold - load_ratio) / max(threshold, 1e-9)
        scale = max(float(self.params.low_load_cost_scale), 0.0)
        exponent = max(float(self.params.low_load_cost_exponent), 1.0)
        return float(1.0 + scale * (normalized_gap**exponent))

    def step(self, target_w: float) -> dict[str, float | bool]:
        params = self.params
        startup = False
        shutdown = False
        target_w = max(float(target_w), 0.0)
        prev_power = self.power_w

        if self.online:
            must_stay_on = self.up_steps < params.min_up_steps
            if target_w < params.p_min_w and not must_stay_on:
                self.online = False
                self.power_w = 0.0
                self.up_steps = 0
                self.down_steps = 1
                shutdown = True
            else:
                desired = min(max(target_w, params.p_min_w), params.p_max_w)
                lower = max(params.p_min_w, prev_power - params.ramp_down_w)
                upper = min(params.p_max_w, prev_power + params.ramp_up_w)
                self.power_w = float(np.clip(desired, lower, upper))
                self.up_steps += 1
                self.down_steps = 0
        else:
            can_start = self.down_steps >= params.min_down_steps and target_w >= params.start_threshold_w
            if can_start:
                self.online = True
                self.start_count += 1
                startup = True
                desired = min(max(target_w, params.p_min_w), params.p_max_w)
                self.power_w = float(min(desired, params.ramp_up_w))
                self.up_steps = 1
                self.down_steps = 0
            else:
                self.power_w = 0.0
                self.down_steps += 1

        power_kw = self.power_w / 1000.0
        load_ratio = self._load_ratio()
        low_load_multiplier = 1.0
        low_load_penalty = 0.0
        operating_cost = 0.0
        if self.online and self.power_w > 0.0:
            operating_cost = params.cost_a0 + params.cost_a1 * power_kw + params.cost_a2 * power_kw * power_kw
            low_load_multiplier = self._low_load_multiplier()
            low_load_penalty = operating_cost * max(low_load_multiplier - 1.0, 0.0)
        total_cost = operating_cost + low_load_penalty + (params.startup_cost if startup else 0.0)

        return {
            f"{params.name}_online": self.online,
            f"{params.name}_power": float(self.power_w),
            f"{params.name}_startup": startup,
            f"{params.name}_shutdown": shutdown,
            f"{params.name}_starts_total": int(self.start_count),
            f"{params.name}_load_ratio": float(load_ratio),
            f"{params.name}_low_load_region": bool(self.online and self.power_w > 0.0 and low_load_multiplier > 1.0),
            f"{params.name}_low_load_multiplier": float(low_load_multiplier),
            f"{params.name}_low_load_penalty": float(low_load_penalty),
            f"{params.name}_operating_cost": float(operating_cost),
            f"{params.name}_total_cost": float(total_cost),
        }


class CIGREComponentPortfolio:
    """Decompose aggregated ``P_other`` into WT/FC/MT plus residual."""

    def __init__(self, wt_max_power_w: float = 10_000.0):
        self.fc = DispatchableUnit(
            DispatchableUnitParams(
                name="fc",
                p_min_w=0.0,
                p_max_w=10_000.0,
                ramp_up_w=5_000.0,
                ramp_down_w=5_000.0,
                min_up_steps=1,
                min_down_steps=1,
                start_threshold_w=2_000.0,
                cost_a0=0.0,
                cost_a1=0.2,
                cost_a2=0.0,
            )
        )
        self.mt = DispatchableUnit(
            DispatchableUnitParams(
                name="mt",
                p_min_w=3_000.0,
                p_max_w=30_000.0,
                ramp_up_w=10_000.0,
                ramp_down_w=10_000.0,
                min_up_steps=2,
                min_down_steps=2,
                start_threshold_w=8_000.0,
                cost_a0=0.4710,
                cost_a1=0.1080,
                cost_a2=0.0103,
            )
        )
        self.wt_max_power_w = float(wt_max_power_w)

    def reset(self) -> None:
        self.fc.reset()
        self.mt.reset()

    def _synthetic_wind_available_power(self, step: int) -> float:
        hour = step % 24
        day = (step // 24) % 7
        diurnal = 0.55 + 0.25 * np.sin(2.0 * np.pi * (hour - 3.0) / 24.0)
        weekly = 0.95 + 0.05 * np.cos(2.0 * np.pi * day / 7.0)
        availability = float(np.clip(diurnal * weekly, 0.15, 1.0))
        return self.wt_max_power_w * availability

    def _resolve_wind_available_power(self, step: int, wind_available_w: float | None = None) -> float:
        if wind_available_w is None:
            return float(self._synthetic_wind_available_power(step))
        return float(np.clip(wind_available_w, 0.0, self.wt_max_power_w))

    def step(self, target_other_w: float, step: int, wind_available_w: float | None = None) -> dict[str, float | bool]:
        target_other_w = float(target_other_w)
        target_generation_w = max(target_other_w, 0.0)
        fixed_other_w = min(target_other_w, 0.0)

        wt_available_w = self._resolve_wind_available_power(step, wind_available_w=wind_available_w)
        wt_power_w = min(target_generation_w, wt_available_w)
        wt_curtailment_w = max(wt_available_w - wt_power_w, 0.0)

        remaining_after_wt = max(target_generation_w - wt_power_w, 0.0)
        fc_info = self.fc.step(remaining_after_wt)
        fc_power_w = float(fc_info["fc_power"])

        remaining_after_fc = max(remaining_after_wt - fc_power_w, 0.0)
        mt_info = self.mt.step(remaining_after_fc)
        mt_power_w = float(mt_info["mt_power"])

        dispatchable_generation_w = wt_power_w + fc_power_w + mt_power_w
        unmet_generation_w = max(target_generation_w - dispatchable_generation_w, 0.0)
        actual_other_power_w = fixed_other_w + dispatchable_generation_w
        tracking_error_w = actual_other_power_w - target_other_w
        total_component_cost = float(fc_info["fc_total_cost"]) + float(mt_info["mt_total_cost"])

        return {
            "wt_available_power": float(wt_available_w),
            "wt_power": float(wt_power_w),
            "wt_curtailment_power": float(wt_curtailment_w),
            **fc_info,
            **mt_info,
            "target_other_power": float(target_other_w),
            "dispatchable_generation": float(dispatchable_generation_w),
            "fixed_other_power": float(fixed_other_w),
            "unmet_generation_power": float(unmet_generation_w),
            "tracking_error_power": float(tracking_error_w),
            "component_operating_cost": float(total_component_cost),
            "component_other_power": float(actual_other_power_w),
            "residual_other_power": float(fixed_other_w),
        }
