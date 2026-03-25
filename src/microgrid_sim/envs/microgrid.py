"""Battery-oriented PCC environment for DRL training and evaluation."""

from __future__ import annotations

from collections import deque
from typing import Any

import calendar
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..cases import MicrogridConfig
from ..data import load_case_data
from ..models import CIGREComponentPortfolio, DispatchableUnit, SimpleBattery, TheveninBattery


class MicrogridEnv(gym.Env):
    """State and reward structure for battery scheduling."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: MicrogridConfig | None = None, render_mode: str | None = None):
        super().__init__()
        self.config = config or MicrogridConfig()
        self.render_mode = render_mode
        self.total_steps = self.config.simulation_days * 24
        self.battery = self._build_battery()
        self.residential_generator = self._build_residential_generator()
        self.component_portfolio = self._build_component_portfolio()
        self._load_data()
        base_obs_dim = 19
        self.observation_stack_steps = max(int(getattr(self.config, "observation_stack_steps", 1)), 1)
        self.single_obs_dim = base_obs_dim + (4 if self._generator_enabled() else 0)
        obs_dim = self.single_obs_dim * self.observation_stack_steps
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        if self._generator_enabled():
            self.action_space = spaces.Box(
                low=np.array([-1.0, 0.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.current_step = 0
        self.cumulative_cost = 0.0
        self.episode_rewards: list[float] = []
        self._obs_history: deque[np.ndarray] = deque(maxlen=self.observation_stack_steps)

    def _build_battery(self):
        if self.config.battery_model == "simple":
            return SimpleBattery(self.config.battery_params)
        return TheveninBattery(self.config.battery_params)

    def _build_component_portfolio(self):
        if self.config.case_key == "mg_cigre" and self.config.component_commitment_enabled:
            return CIGREComponentPortfolio()
        return None

    def _build_residential_generator(self):
        if self._generator_enabled():
            return DispatchableUnit(self.config.generator_params)
        return None

    def _generator_enabled(self) -> bool:
        return (
            self.config.case_key == "mg_res"
            and bool(getattr(self.config, "generator_enabled", False))
            and getattr(self.config, "generator_params", None) is not None
        )

    def _base_other_power(self, idx: int) -> float:
        other = float(self.other_power_target[idx])
        if self.config.case_key == "mg_res" and hasattr(self, "wind_power"):
            other += float(self.wind_power[idx])
        return other

    def _battery_temperature_obs(self) -> float:
        params = self.config.battery_params
        temperature_c = float(getattr(self.battery, "temperature_c", params.temperature_init_c))
        ambient = float(getattr(params, "ambient_temperature_c", params.temperature_init_c))
        scale = max(0.25 * (float(params.temperature_max_c) - float(params.temperature_min_c)), 8.0)
        normalized = (temperature_c - ambient) / scale
        return float(np.clip(normalized, -1.0, 1.0))

    def _battery_hysteresis_obs(self) -> float:
        params = self.config.battery_params
        charge_curve = np.asarray(getattr(params, "ocv_charge_values", params.ocv_values), dtype=float)
        discharge_curve = np.asarray(getattr(params, "ocv_discharge_values", params.ocv_values), dtype=float)
        scale = float(np.max(np.abs(charge_curve - discharge_curve))) * float(params.num_cells_series)
        if scale <= 1e-9:
            return 0.0
        value = float(getattr(self.battery, "ocv_hysteresis_offset_v", 0.0))
        return float(np.clip(value / scale, -1.0, 1.0))

    def _battery_polarization_obs(self) -> float:
        params = self.config.battery_params
        value = float(getattr(self.battery, "v_rc_1", 0.0)) + float(getattr(self.battery, "v_rc_2", 0.0))
        rc1 = np.asarray(getattr(params, "rc_branch_1_resistance_values", np.zeros(1, dtype=float)), dtype=float)
        rc2 = np.asarray(getattr(params, "rc_branch_2_resistance_values", np.zeros(1, dtype=float)), dtype=float)
        rc1 = rc1[np.isfinite(rc1)]
        rc2 = rc2[np.isfinite(rc2)]
        if rc1.size and rc2.size:
            rc_sum_cell = float(np.max(rc1 + rc2))
        elif rc1.size:
            rc_sum_cell = float(np.max(rc1))
        elif rc2.size:
            rc_sum_cell = float(np.max(rc2))
        else:
            rc_sum_cell = 0.0
        if hasattr(self.battery, "get_ocv"):
            nominal_voltage = float(self.battery.get_ocv(float(self.battery.soc))) * float(params.num_cells_series)
        else:
            ocv_values = np.asarray(getattr(params, "ocv_values", np.zeros(1, dtype=float)), dtype=float)
            nominal_voltage = float(np.max(ocv_values)) * float(params.num_cells_series)
        nominal_voltage = max(nominal_voltage, 1e-6)
        reference_current = 0.05 * float(params.cell_capacity_ah) * max(float(params.num_cells_parallel), 1.0)
        reference_current = max(reference_current, 0.05 * max(float(params.p_charge_max), float(params.p_discharge_max)) / nominal_voltage)
        scale = max(rc_sum_cell * float(params.num_cells_series) * reference_current, 0.05)
        return float(np.clip(value / scale, -1.0, 1.0))

    def _battery_power_limit_obs(self) -> float:
        params = self.config.battery_params
        rated_power = max(float(params.p_discharge_max), float(params.p_charge_max), 1e-6)
        if not hasattr(self.battery, "get_r_int"):
            return 1.0
        try:
            soc = float(getattr(self.battery, "soc", params.soc_init))
            if hasattr(self.battery, "_effective_ocv"):
                v_ocv = float(self.battery._effective_ocv(soc)[0])
            else:
                v_ocv = float(self.battery.get_ocv(soc)) * float(params.num_cells_series)
            r_int = float(self.battery.get_r_int(soc)) * float(params.num_cells_series) / max(float(params.num_cells_parallel), 1.0)
            if hasattr(self.battery, "_temperature_r_int_factor"):
                r_int *= float(self.battery._temperature_r_int_factor())
            if hasattr(self.battery, "_soc_r_int_factor"):
                r_int *= float(self.battery._soc_r_int_factor(soc))
            if hasattr(self.battery, "_paper_soc_r_int_factor"):
                r_int *= float(self.battery._paper_soc_r_int_factor(soc, 0.0))
            polarization_voltage = float(getattr(self.battery, "v_rc_1", 0.0)) + float(getattr(self.battery, "v_rc_2", 0.0))
            v_headroom = max(v_ocv - polarization_voltage, 1e-6)
            p_max = rated_power if r_int <= 0.0 else (v_headroom**2) / (4.0 * r_int)
            return float(np.clip(p_max / rated_power, 0.0, 2.0))
        except Exception:
            return 1.0

    def _battery_resistance_obs(self) -> float:
        params = self.config.battery_params
        if not hasattr(self.battery, "get_r_int"):
            return 0.0
        try:
            soc = float(getattr(self.battery, "soc", params.soc_init))
            current_r_int = float(self.battery.get_r_int(soc)) * float(params.num_cells_series) / max(float(params.num_cells_parallel), 1.0)
            if hasattr(self.battery, "_temperature_r_int_factor"):
                current_r_int *= float(self.battery._temperature_r_int_factor())
            if hasattr(self.battery, "_soc_r_int_factor"):
                current_r_int *= float(self.battery._soc_r_int_factor(soc))
            if hasattr(self.battery, "_paper_soc_r_int_factor"):
                current_r_int *= float(self.battery._paper_soc_r_int_factor(soc, 0.0))
            base_values = np.asarray(getattr(params, "r_int_values", np.zeros(1, dtype=float)), dtype=float)
            mean_r_int = float(np.mean(base_values)) * float(params.num_cells_series) / max(float(params.num_cells_parallel), 1.0)
            if mean_r_int <= 1e-9 or current_r_int <= 1e-9:
                return 0.0
            return float(np.tanh(np.log(current_r_int / mean_r_int)))
        except Exception:
            return 0.0

    def _current_monthly_peak_billed_kw(self, idx: int) -> float:
        if idx >= len(self.month_index):
            return 0.0
        month_id = int(self.month_index[idx])
        peak_state = getattr(self, "monthly_peak_billed_kw", {})
        return float(peak_state.get(month_id, 0.0))

    def _monthly_peak_billed_obs(self, idx: int) -> float:
        threshold_kw = float(self.config.monthly_demand_charge_threshold_w) / 1000.0
        if float(self.config.monthly_demand_charge_per_kw) <= 0.0 or not np.isfinite(threshold_kw) or threshold_kw <= 0.0:
            return 0.0
        scale_kw = max(threshold_kw, float(self.config.load_max_power) / 1000.0, 1e-6)
        return float(np.clip(self._current_monthly_peak_billed_kw(idx) / scale_kw, 0.0, 2.0))

    def _monthly_peak_headroom_obs(self, idx: int) -> float:
        threshold_kw = float(self.config.monthly_demand_charge_threshold_w) / 1000.0
        if float(self.config.monthly_demand_charge_per_kw) <= 0.0 or not np.isfinite(threshold_kw) or threshold_kw <= 0.0:
            return 0.0
        current_peak_kw = self._current_monthly_peak_billed_kw(idx)
        headroom_kw = threshold_kw - current_peak_kw
        return float(np.clip(headroom_kw / threshold_kw, -1.0, 1.0))

    def _generator_low_load_state(self, generator_power: float) -> tuple[float, float, bool]:
        if not self._generator_enabled() or self.config.generator_params is None or generator_power <= 0.0:
            return 0.0, 1.0, False
        params = self.config.generator_params
        load_ratio = float(np.clip(generator_power / max(float(params.p_max_w), 1e-9), 0.0, 1.0))
        threshold = float(np.clip(getattr(params, "low_load_threshold_fraction", 0.0), 0.0, 1.0))
        if threshold <= 0.0 or load_ratio >= threshold:
            return load_ratio, 1.0, False
        normalized_gap = (threshold - load_ratio) / max(threshold, 1e-9)
        scale = max(float(getattr(params, "low_load_cost_scale", 0.0)), 0.0)
        exponent = max(float(getattr(params, "low_load_cost_exponent", 2.0)), 1.0)
        multiplier = float(1.0 + scale * (normalized_gap**exponent))
        return load_ratio, multiplier, multiplier > 1.0 + 1e-9

    def _residential_generator_snapshot(self, idx: int, generator_setpoint: float | None = None, generator_cost: float = 0.0, generator_startup: bool = False, generator_shutdown: bool = False, generator_low_load_penalty: float = 0.0) -> dict[str, Any]:
        renewable_other = float(self.wind_power[idx]) if self.config.case_key == "mg_res" and hasattr(self, "wind_power") else 0.0
        fixed_other = float(self.other_power_target[idx])
        generator_power = float(self.residential_generator.power_w) if self.residential_generator is not None else 0.0
        generator_online = bool(self.residential_generator.online) if self.residential_generator is not None else False
        starts_total = int(self.residential_generator.start_count) if self.residential_generator is not None else 0
        up_steps = int(self.residential_generator.up_steps) if self.residential_generator is not None else 0
        down_steps = int(self.residential_generator.down_steps) if self.residential_generator is not None else 0
        total_other = fixed_other + renewable_other + generator_power
        setpoint = generator_power if generator_setpoint is None else float(generator_setpoint)
        generator_load_ratio, generator_low_load_multiplier, generator_low_load_region = self._generator_low_load_state(generator_power)
        return {
            "target_other_power": float(fixed_other + renewable_other),
            "component_other_power": float(total_other),
            "component_operating_cost": float(generator_cost),
            "renewable_other_power": float(renewable_other),
            "dispatchable_generation": float(generator_power),
            "fixed_other_power": float(fixed_other),
            "generator_power": float(generator_power),
            "generator_setpoint": float(setpoint),
            "generator_online": generator_online,
            "generator_startup": bool(generator_startup),
            "generator_shutdown": bool(generator_shutdown),
            "generator_starts_total": starts_total,
            "generator_up_steps": up_steps,
            "generator_down_steps": down_steps,
            "generator_cost": float(generator_cost),
            "generator_load_ratio": float(generator_load_ratio),
            "generator_low_load_multiplier": float(generator_low_load_multiplier),
            "generator_low_load_region": bool(generator_low_load_region),
            "generator_low_load_penalty": float(generator_low_load_penalty),
            "wt_available_power": float(renewable_other),
            "wt_power": float(renewable_other),
        }

    def _step_residential_generator(self, idx: int, action: np.ndarray) -> dict[str, Any]:
        if self.residential_generator is None:
            return self._static_component_info(idx)
        generator_setpoint = self._generator_action_to_power(action)
        generator_info = self.residential_generator.step(generator_setpoint)
        total_cost = float(generator_info.get("gen_total_cost", 0.0))
        snapshot = self._residential_generator_snapshot(
            idx,
            generator_setpoint=generator_setpoint,
            generator_cost=total_cost,
            generator_startup=bool(generator_info.get("gen_startup", False)),
            generator_shutdown=bool(generator_info.get("gen_shutdown", False)),
            generator_low_load_penalty=float(generator_info.get("gen_low_load_penalty", 0.0)),
        )
        snapshot.update(
            {
                "component_operating_cost": total_cost,
                "dispatchable_generation": float(generator_info.get("gen_power", snapshot["dispatchable_generation"])),
                "generator_power": float(generator_info.get("gen_power", snapshot["generator_power"])),
                "generator_online": bool(generator_info.get("gen_online", snapshot["generator_online"])),
                "generator_operating_cost": float(generator_info.get("gen_operating_cost", total_cost)),
                "generator_cost": total_cost,
                "generator_load_ratio": float(generator_info.get("gen_load_ratio", snapshot["generator_load_ratio"])),
                "generator_low_load_multiplier": float(generator_info.get("gen_low_load_multiplier", snapshot["generator_low_load_multiplier"])),
                "generator_low_load_region": bool(generator_info.get("gen_low_load_region", snapshot["generator_low_load_region"])),
                "generator_low_load_penalty": float(generator_info.get("gen_low_load_penalty", snapshot["generator_low_load_penalty"])),
                "generator_startup": bool(generator_info.get("gen_startup", False)),
                "generator_shutdown": bool(generator_info.get("gen_shutdown", False)),
                "generator_starts_total": int(generator_info.get("gen_starts_total", snapshot["generator_starts_total"])),
            }
        )
        snapshot["component_other_power"] = snapshot["fixed_other_power"] + snapshot["renewable_other_power"] + snapshot["generator_power"]
        return snapshot

    def _static_component_info(
        self,
        idx: int,
        generator_power: float = 0.0,
        generator_setpoint: float = 0.0,
        generator_cost: float = 0.0,
    ) -> dict[str, Any]:
        renewable_other = 0.0
        if self.config.case_key == "mg_res" and hasattr(self, "wind_power"):
            renewable_other = float(self.wind_power[idx])
        fixed_other = float(self.other_power_target[idx])
        total_other = fixed_other + renewable_other + float(generator_power)
        generator_load_ratio, generator_low_load_multiplier, generator_low_load_region = self._generator_low_load_state(float(generator_power))
        return {
            "target_other_power": float(fixed_other + renewable_other),
            "component_other_power": float(total_other),
            "component_operating_cost": float(generator_cost),
            "renewable_other_power": float(renewable_other),
            "dispatchable_generation": float(generator_power),
            "fixed_other_power": float(fixed_other),
            "generator_power": float(generator_power),
            "generator_setpoint": float(generator_setpoint),
            "generator_online": bool(generator_power > 1e-9),
            "generator_cost": float(generator_cost),
            "generator_load_ratio": float(generator_load_ratio),
            "generator_low_load_multiplier": float(generator_low_load_multiplier),
            "generator_low_load_region": bool(generator_low_load_region),
            "generator_low_load_penalty": 0.0,
            "wt_available_power": float(renewable_other),
            "wt_power": float(renewable_other),
        }

    def _transform_price_series(self, price: np.ndarray) -> np.ndarray:
        series = np.asarray(price, dtype=float)
        if series.size == 0:
            return series
        multiplier = max(float(self.config.tou_price_spread_multiplier), 0.0)
        if abs(multiplier - 1.0) < 1e-9:
            return series.copy()
        center = 0.5 * (float(np.min(series)) + float(np.max(series)))
        transformed = center + (series - center) * multiplier
        return np.clip(transformed, 0.0, None)

    def _hour_window_mask(self, length: int, start_hour: int, end_hour: int) -> np.ndarray:
        if length <= 0:
            return np.zeros(0, dtype=bool)
        hours = np.arange(length) % 24
        start = int(start_hour) % 24
        end = int(end_hour) % 24
        if start <= end:
            return (hours >= start) & (hours <= end)
        return (hours >= start) | (hours <= end)

    def _boost_series_by_hour(self, series: np.ndarray, multiplier: float, start_hour: int, end_hour: int) -> np.ndarray:
        values = np.asarray(series, dtype=float).copy()
        factor = max(float(multiplier), 0.0)
        if values.size == 0 or abs(factor - 1.0) < 1e-9:
            return values
        mask = self._hour_window_mask(len(values), start_hour, end_hour)
        values[mask] *= factor
        return values

    def _rolling_window_mean(self, values: np.ndarray, window: int) -> np.ndarray:
        samples = np.asarray(values, dtype=float)
        if samples.size == 0 or window <= 0 or samples.size < window:
            return np.zeros(0, dtype=float)
        cumulative = np.cumsum(np.concatenate(([0.0], samples)))
        return (cumulative[window:] - cumulative[:-window]) / float(window)

    def _effective_peak_threshold_w(self) -> float:
        thresholds: list[float] = []
        if float(self.config.peak_import_penalty_per_kw) > 0.0:
            thresholds.append(float(self.config.peak_import_penalty_threshold_w))
        if float(self.config.monthly_demand_charge_per_kw) > 0.0:
            thresholds.append(float(self.config.monthly_demand_charge_threshold_w))
        finite = [value for value in thresholds if np.isfinite(value)]
        return min(finite) if finite else float("inf")

    def _build_source_month_index(self, hours: int) -> np.ndarray:
        if hours <= 0:
            return np.zeros(0, dtype=int)
        year = int(self.config.data_year or 2023)
        month_lengths = [31, 29 if calendar.isleap(year) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        month_hours = np.asarray(month_lengths, dtype=int) * 24
        cumulative = np.cumsum(month_hours)
        hour_index = np.arange(hours) % int(cumulative[-1])
        month_index = np.searchsorted(cumulative, hour_index, side="right")
        return month_index.astype(int)

    def _build_episode_start_probability(self) -> np.ndarray | None:
        if not self.config.stress_episode_sampling or self.source_hours <= self.total_steps:
            return None
        net_import = np.clip(self.source_load_power - self.source_pv_power - self.source_other_power, 0.0, None)
        price_norm = self.source_price / max(float(np.max(self.source_price)), 1e-9)
        net_norm = net_import / max(float(np.max(net_import)), 1e-9)
        threshold_w = self._effective_peak_threshold_w()
        peak_excess = np.clip(net_import - threshold_w, 0.0, None) if np.isfinite(threshold_w) else np.zeros_like(net_import)
        peak_norm = peak_excess / max(float(np.max(peak_excess)), 1e-9) if np.max(peak_excess) > 0 else np.zeros_like(net_import)
        hourly_stress = 0.35 * price_norm + 0.45 * net_norm + 0.20 * peak_norm
        window_stress = self._rolling_window_mean(hourly_stress, self.total_steps)
        if window_stress.size == 0:
            return None
        span = float(np.max(window_stress) - np.min(window_stress))
        normalized = (window_stress - float(np.min(window_stress))) / max(span, 1e-9)
        strength = max(float(self.config.stress_sampling_strength), 0.0)
        weights = 1.0 + strength * np.square(normalized)
        total = float(np.sum(weights))
        if total <= 0.0 or not np.isfinite(total):
            return None
        return weights / total

    def _load_data(self):
        data = load_case_data(config=self.config, total_hours=None, data_dir=self.config.data_dir)
        self.source_pv_power = np.asarray(data["pv"], dtype=float)
        self.source_load_power = np.asarray(data["load"], dtype=float)
        self.source_other_power = np.asarray(data.get("other", np.zeros_like(self.source_load_power)), dtype=float)
        self.source_wind_power = np.asarray(data.get("wind", np.zeros_like(self.source_load_power)), dtype=float)
        self.source_pv_power = self._boost_series_by_hour(
            self.source_pv_power,
            self.config.midday_pv_boost_multiplier,
            self.config.midday_pv_boost_start_hour,
            self.config.midday_pv_boost_end_hour,
        )
        self.source_load_power = self._boost_series_by_hour(
            self.source_load_power,
            self.config.evening_load_boost_multiplier,
            self.config.evening_load_boost_start_hour,
            self.config.evening_load_boost_end_hour,
        )
        self.source_price_raw = np.asarray(data["price"], dtype=float)
        self.source_price = self._transform_price_series(self.source_price_raw)
        self.source_hours = len(self.source_load_power)
        self.source_month_index = self._build_source_month_index(self.source_hours)
        self.data_source = data.get("source", "unknown")
        self.price_obs_max = max(float(np.max(self.source_price)) if self.source_price.size else self.config.price_max, 1e-9)
        self.episode_start_probability = self._build_episode_start_probability()
        self._bind_episode_window(self.config.episode_start_hour)

    def _take_window(self, values: np.ndarray, start_hour: int, length: int) -> np.ndarray:
        samples = np.asarray(values, dtype=float)
        if samples.size == 0 or length <= 0:
            return np.zeros(0, dtype=float)
        if samples.size < length:
            repeat_count = int(np.ceil(length / samples.size))
            return np.tile(samples, repeat_count)[:length]
        if start_hour < 0:
            start_hour = int(start_hour % samples.size)
        if start_hour + length <= samples.size:
            return samples[start_hour:start_hour + length]
        indices = (np.arange(length, dtype=int) + int(start_hour)) % samples.size
        return samples[indices]

    def _bind_episode_window(self, start_hour: int) -> None:
        if self.source_hours <= 0:
            raise ValueError("Loaded dataset has zero length")

        repeat_count = int(np.ceil(self.total_steps / self.source_hours)) if self.source_hours < self.total_steps else 1
        if self.source_hours < self.total_steps:
            self.episode_start_hour = 0
            self.pv_power = np.tile(self.source_pv_power, repeat_count)[: self.total_steps]
            self.load_power = np.tile(self.source_load_power, repeat_count)[: self.total_steps]
            self.other_power_target = np.tile(self.source_other_power, repeat_count)[: self.total_steps]
            self.wind_power = np.tile(self.source_wind_power, repeat_count)[: self.total_steps]
            self.tou_price_raw = np.tile(self.source_price_raw, repeat_count)[: self.total_steps]
            self.tou_price = np.tile(self.source_price, repeat_count)[: self.total_steps]
            self.month_index = np.tile(self.source_month_index, repeat_count)[: self.total_steps]
            return

        if self.source_hours == self.total_steps:
            self.episode_start_hour = int(start_hour % self.source_hours)
            self.pv_power = self._take_window(self.source_pv_power, self.episode_start_hour, self.total_steps)
            self.load_power = self._take_window(self.source_load_power, self.episode_start_hour, self.total_steps)
            self.other_power_target = self._take_window(self.source_other_power, self.episode_start_hour, self.total_steps)
            self.wind_power = self._take_window(self.source_wind_power, self.episode_start_hour, self.total_steps)
            self.tou_price_raw = self._take_window(self.source_price_raw, self.episode_start_hour, self.total_steps)
            self.tou_price = self._take_window(self.source_price, self.episode_start_hour, self.total_steps)
            self.month_index = self._take_window(self.source_month_index, self.episode_start_hour, self.total_steps).astype(int, copy=False)
            return

        max_start = max(self.source_hours - self.total_steps, 0)
        self.episode_start_hour = int(np.clip(start_hour, 0, max_start))
        end_hour = self.episode_start_hour + self.total_steps
        self.pv_power = self.source_pv_power[self.episode_start_hour:end_hour]
        self.load_power = self.source_load_power[self.episode_start_hour:end_hour]
        self.other_power_target = self.source_other_power[self.episode_start_hour:end_hour]
        self.wind_power = self.source_wind_power[self.episode_start_hour:end_hour]
        self.tou_price_raw = self.source_price_raw[self.episode_start_hour:end_hour]
        self.tou_price = self.source_price[self.episode_start_hour:end_hour]
        self.month_index = self.source_month_index[self.episode_start_hour:end_hour]

    def _resolve_component_state(self, idx: int) -> dict[str, Any]:
        if self.residential_generator is not None:
            return self._residential_generator_snapshot(idx)
        if self.component_portfolio is None:
            return self._static_component_info(idx)
        return self.component_portfolio.step(
            float(self.other_power_target[idx]),
            idx,
            wind_available_w=float(self.wind_power[idx]),
        )

    def _single_scaled_obs(self) -> np.ndarray:
        idx = min(self.current_step, self.total_steps - 1)
        year = int(self.config.data_year or 2023)
        year_days = 366 if calendar.isleap(year) else 365
        hour = (self.episode_start_hour + idx) % 24
        abs_day = (self.episode_start_hour + idx) // 24
        day = abs_day % 7
        day_of_year = abs_day % max(year_days, 1)
        pv = self.pv_power[idx]
        load = self.load_power[idx]
        component_info = dict(getattr(self, "current_component_info", {}) or {})
        other = float(getattr(self, "current_other_power", self._base_other_power(idx)))
        net = load - pv - other
        price = self.tou_price[idx]
        obs = [
            np.sin(2 * np.pi * hour / 24.0),
            np.cos(2 * np.pi * hour / 24.0),
            np.sin(2 * np.pi * day / 7.0),
            np.cos(2 * np.pi * day / 7.0),
            np.sin(2 * np.pi * day_of_year / max(float(year_days), 1.0)),
            np.cos(2 * np.pi * day_of_year / max(float(year_days), 1.0)),
            self.battery.soc,
            self.battery.soh,
            pv / max(self.config.pv_max_power, 1.0),
            load / max(self.config.load_max_power, 1.0),
            net / max(self.config.load_max_power, 1.0),
            price / self.price_obs_max,
            self._battery_temperature_obs(),
            self._battery_hysteresis_obs(),
            self._battery_polarization_obs(),
            self._monthly_peak_billed_obs(idx),
            self._monthly_peak_headroom_obs(idx),
            self._battery_power_limit_obs(),
            self._battery_resistance_obs(),
        ]
        if self._generator_enabled():
            generator_params = self.config.generator_params
            min_up_steps = max(int(getattr(generator_params, "min_up_steps", 1)), 1)
            min_down_steps = max(int(getattr(generator_params, "min_down_steps", 1)), 1)
            generator_power = float(component_info.get("generator_power", 0.0))
            generator_online = 1.0 if bool(component_info.get("generator_online", False)) else 0.0
            generator_power_norm = generator_power / max(float(self.config.generator_params.p_max_w), 1.0)
            generator_up_steps = float(component_info.get("generator_up_steps", 0.0))
            generator_down_steps = float(component_info.get("generator_down_steps", 0.0))
            generator_up_norm = float(np.clip(generator_up_steps / float(min_up_steps), 0.0, 1.0))
            generator_down_norm = float(np.clip(generator_down_steps / float(min_down_steps), 0.0, 1.0))
            obs.extend([
                generator_online,
                generator_power_norm,
                generator_up_norm,
                generator_down_norm,
            ])
        return np.asarray(obs, dtype=np.float32)

    def _scaled_obs(self) -> np.ndarray:
        current_obs = self._single_scaled_obs()
        if not self._obs_history:
            self._obs_history.extend(current_obs.copy() for _ in range(self.observation_stack_steps))
        return np.concatenate(list(self._obs_history)).astype(np.float32, copy=False)

    def _action_to_power(self, action: np.ndarray) -> float:
        value = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
        params = self.battery.params
        if value >= 0:
            return value * params.p_discharge_max
        return value * params.p_charge_max

    def _generator_action_to_power(self, action: np.ndarray) -> float:
        if not self._generator_enabled():
            return 0.0
        values = np.asarray(action, dtype=float).reshape(-1)
        if values.size < 2:
            return 0.0
        params = self.config.generator_params
        normalized = float(np.clip(values[1], 0.0, 1.0))
        requested_power = normalized * float(params.p_max_w)
        if requested_power < float(params.start_threshold_w):
            return 0.0
        return float(np.clip(max(requested_power, float(params.p_min_w)), float(params.p_min_w), float(params.p_max_w)))

    def _generator_operating_cost(self, generator_power_w: float) -> float:
        if not self._generator_enabled() or generator_power_w <= 0.0:
            return 0.0
        params = self.config.generator_params
        power_kw = float(generator_power_w) / 1000.0
        return float(params.cost_a0 + params.cost_a1 * power_kw + params.cost_a2 * power_kw * power_kw)

    def _apply_grid_bounds(self, net: float, p_cmd: float) -> tuple[float, dict[str, float]]:
        lower = net - self.config.grid_import_max
        upper = net + self.config.grid_export_max
        p_feasible = float(np.clip(p_cmd, lower, upper))
        return p_feasible, {
            "battery_power_min": float(lower),
            "battery_power_max": float(upper),
            "p_cmd_clipped_by_grid": float(p_feasible),
        }

    def _price_factor(self, price: float) -> float:
        reward = self.config.reward
        if price <= reward.valley_price:
            return -1.0
        if price >= reward.peak_price:
            return 1.0
        span = reward.peak_price - reward.valley_price
        return 2.0 * (price - reward.valley_price) / span - 1.0

    def _peak_import_penalty(self, p_grid: float) -> float:
        threshold_w = float(self.config.peak_import_penalty_threshold_w)
        penalty_per_kw = float(self.config.peak_import_penalty_per_kw)
        if penalty_per_kw <= 0.0 or not np.isfinite(threshold_w):
            return 0.0
        excess_kw = max(float(p_grid) - threshold_w, 0.0) / 1000.0
        if excess_kw <= 0.0:
            return 0.0
        dt_hours = self.config.dt_seconds / 3600.0
        return penalty_per_kw * excess_kw * dt_hours

    def _monthly_peak_increment_kw(self, idx: int, p_grid: float) -> tuple[int | None, float, float]:
        threshold_w = float(self.config.monthly_demand_charge_threshold_w)
        if idx >= len(self.month_index):
            return None, 0.0, 0.0
        billed_kw = max(float(p_grid) - threshold_w, 0.0) / 1000.0
        if billed_kw <= 0.0:
            return int(self.month_index[idx]), 0.0, billed_kw
        month_id = int(self.month_index[idx])
        previous_peak_kw = float(self.monthly_peak_billed_kw.get(month_id, 0.0))
        increment_kw = max(billed_kw - previous_peak_kw, 0.0)
        return month_id, increment_kw, billed_kw

    def _monthly_demand_charge(self, month_id: int | None, monthly_peak_increment_kw: float, billed_kw: float) -> float:
        charge_per_kw = float(self.config.monthly_demand_charge_per_kw)
        if charge_per_kw <= 0.0 or month_id is None or monthly_peak_increment_kw <= 0.0:
            return 0.0
        self.monthly_peak_billed_kw[int(month_id)] = float(billed_kw)
        return charge_per_kw * monthly_peak_increment_kw

    def _monthly_peak_increment_penalty(self, monthly_peak_increment_kw: float) -> float:
        penalty_per_kw = float(self.config.monthly_peak_increment_penalty_per_kw)
        if penalty_per_kw <= 0.0 or monthly_peak_increment_kw <= 0.0:
            return 0.0
        return penalty_per_kw * monthly_peak_increment_kw

    def _battery_throughput_penalty(self, battery_info: dict) -> float:
        penalty_per_kwh = float(self.config.battery_throughput_penalty_per_kwh)
        if penalty_per_kwh <= 0.0:
            return 0.0
        effective_power = abs(float(battery_info.get("effective_power", 0.0)))
        dt_hours = self.config.dt_seconds / 3600.0
        throughput_kwh = effective_power * dt_hours / 1000.0
        return penalty_per_kwh * throughput_kwh

    def _battery_loss_penalty(self, battery_info: dict) -> float:
        penalty_per_kwh = float(self.config.battery_loss_penalty_per_kwh)
        if penalty_per_kwh <= 0.0:
            return 0.0
        power_loss = max(float(battery_info.get("power_loss", 0.0)), 0.0)
        dt_hours = self.config.dt_seconds / 3600.0
        loss_kwh = power_loss * dt_hours / 1000.0
        return penalty_per_kwh * loss_kwh

    def _battery_stress_penalty(self, battery_info: dict) -> float:
        penalty_per_kwh = float(self.config.battery_stress_penalty_per_kwh)
        if penalty_per_kwh <= 0.0:
            return 0.0
        effective_power = abs(float(battery_info.get("effective_power", 0.0)))
        dt_hours = self.config.dt_seconds / 3600.0
        throughput_kwh = effective_power * dt_hours / 1000.0
        soc_excess = max(float(battery_info.get("r_int_soc_factor", 1.0)) - 1.0, 0.0)
        power_excess = max(float(battery_info.get("r_int_power_factor", 1.0)) - 1.0, 0.0)
        temp_excess = max(float(battery_info.get("r_int_temp_factor", 1.0)) - 1.0, 0.0)
        stress_multiplier = soc_excess + 0.75 * power_excess + 0.50 * temp_excess
        if stress_multiplier <= 0.0 or throughput_kwh <= 0.0:
            return 0.0
        return penalty_per_kwh * throughput_kwh * stress_multiplier

    def _calculate_reward(
        self,
        cost: float,
        battery_info: dict,
        p_grid: float,
        price: float,
        monthly_peak_increment_penalty: float = 0.0,
    ) -> float:
        reward_cfg = self.config.reward
        if self.config.reward_mode == "cost":
            del p_grid, price
            reward = -float(cost)
            reward -= float(monthly_peak_increment_penalty)
            reward -= 100.0 * reward_cfg.w_soc_violation * float(battery_info.get("soc_violation", 0.0))
            return float(np.clip(reward, reward_cfg.reward_min, reward_cfg.reward_max))

        p_buy_kw = max(p_grid, 0.0) / 1000.0
        p_sell_kw = max(-p_grid, 0.0) / 1000.0
        price_factor = self._price_factor(price)
        if price_factor < 0:
            r_buy = 20.0 * (1.0 - np.exp(-p_buy_kw / 5.0)) * (-price_factor)
        else:
            r_buy = -30.0 * np.tanh(p_buy_kw / 10.0) * price_factor
        r_sell = 15.0 * np.tanh(p_sell_kw / 5.0) * max(0.0, price_factor)
        r_economic = r_buy + r_sell
        soc = float(battery_info["soc"])
        if reward_cfg.soc_band_min <= soc <= reward_cfg.soc_band_max:
            r_band = reward_cfg.w_band * np.exp(-((soc - reward_cfg.soc_center) ** 2) / (2.0 * reward_cfg.soc_sigma**2))
        else:
            r_band = -reward_cfg.w_edge * abs(soc - reward_cfg.soc_center)
        r_soc = r_band - 100.0 * reward_cfg.w_soc_violation * float(battery_info["soc_violation"])
        r_soh = reward_cfg.w_soh * float(battery_info["soh"])
        r_cost = -reward_cfg.w_cost * cost
        reward = r_economic + r_soc + r_soh + r_cost - float(monthly_peak_increment_penalty)
        return float(np.clip(reward, reward_cfg.reward_min, reward_cfg.reward_max))

    def _get_info(
        self,
        idx: int | None = None,
        other_override: float | None = None,
        component_info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        idx = min(self.current_step if idx is None else idx, self.total_steps - 1)
        if other_override is None:
            other_power = float(getattr(self, "current_other_power", self._base_other_power(idx)))
        else:
            other_power = float(other_override)
        info = {
            "step": self.current_step,
            "hour": (self.episode_start_hour + idx) % 24,
            "episode_start_hour": int(self.episode_start_hour),
            "soc": self.battery.soc,
            "soh": self.battery.soh,
            "cumulative_cost": self.cumulative_cost,
            "price": float(self.tou_price[idx]),
            "price_raw": float(self.tou_price_raw[idx]),
            "pv_power": float(self.pv_power[idx]),
            "load_power": float(self.load_power[idx]),
            "other_power": other_power,
            "net_load": float(self.load_power[idx] - self.pv_power[idx] - other_power),
            "data_source": self.data_source,
            "battery_temperature_c": float(getattr(self.battery, "temperature_c", self.config.battery_params.temperature_init_c)),
            "battery_power_limit_ratio": float(self._battery_power_limit_obs()),
            "battery_resistance_state": float(self._battery_resistance_obs()),
            "current_monthly_peak_billed_kw": float(self._current_monthly_peak_billed_kw(idx)),
            "remaining_monthly_peak_headroom_kw": float(
                max(float(self.config.monthly_demand_charge_threshold_w) / 1000.0 - self._current_monthly_peak_billed_kw(idx), 0.0)
            ) if np.isfinite(float(self.config.monthly_demand_charge_threshold_w)) else float("inf"),
            "price_spread_multiplier": float(self.config.tou_price_spread_multiplier),
            "peak_import_penalty_per_kw": float(self.config.peak_import_penalty_per_kw),
            "peak_import_penalty_threshold_w": float(self.config.peak_import_penalty_threshold_w),
            "monthly_demand_charge_per_kw": float(self.config.monthly_demand_charge_per_kw),
            "monthly_demand_charge_threshold_w": float(self.config.monthly_demand_charge_threshold_w),
            "monthly_peak_increment_penalty_per_kw": float(self.config.monthly_peak_increment_penalty_per_kw),
            "midday_pv_boost_multiplier": float(self.config.midday_pv_boost_multiplier),
            "evening_load_boost_multiplier": float(self.config.evening_load_boost_multiplier),
        }
        if component_info:
            info.update(component_info)
        return info

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        del options
        if self.config.random_episode_start and self.source_hours >= self.total_steps:
            if self.source_hours == self.total_steps:
                configured_starts = tuple(int(value) for value in getattr(self.config, "full_year_random_start_hours", ()) or ())
                if configured_starts:
                    candidate_starts = np.unique(np.asarray([value % self.source_hours for value in configured_starts], dtype=int))
                else:
                    stride_hours = max(int(getattr(self.config, "full_year_random_start_stride_hours", 1)), 1)
                    candidate_starts = np.arange(0, self.source_hours, stride_hours, dtype=int)
                random_start = int(self.np_random.choice(candidate_starts)) if candidate_starts.size else 0
            else:
                max_start = self.source_hours - self.total_steps
                if self.episode_start_probability is not None and len(self.episode_start_probability) == max_start + 1:
                    random_start = int(self.np_random.choice(np.arange(max_start + 1), p=self.episode_start_probability))
                else:
                    random_start = int(self.np_random.integers(0, max_start + 1))
            self._bind_episode_window(random_start)
        else:
            self._bind_episode_window(self.config.episode_start_hour)
        self.current_step = 0
        self.cumulative_cost = 0.0
        self.episode_rewards = []
        self.monthly_peak_billed_kw: dict[int, float] = {}
        initial_soc = None
        if self.config.random_initial_soc:
            soc_low = max(float(self.config.initial_soc_min), float(self.battery.params.soc_min))
            soc_high = min(float(self.config.initial_soc_max), float(self.battery.params.soc_max))
            if soc_high > soc_low:
                initial_soc = float(self.np_random.uniform(soc_low, soc_high))
        self.battery.reset(soc=initial_soc)
        if self.residential_generator is not None:
            self.residential_generator.reset()
        if self.component_portfolio is not None:
            self.component_portfolio.reset()
        self.current_component_info = self._resolve_component_state(0)
        self.current_other_power = float(self.current_component_info.get("component_other_power", self.other_power_target[0]))
        initial_obs = self._single_scaled_obs()
        self._obs_history.clear()
        self._obs_history.extend(initial_obs.copy() for _ in range(self.observation_stack_steps))
        return self._scaled_obs(), self._get_info(idx=0, other_override=self.current_other_power, component_info=self.current_component_info)

    def step(self, action: np.ndarray):
        idx = self.current_step
        pv = float(self.pv_power[idx])
        load = float(self.load_power[idx])
        action_values = np.asarray(action, dtype=float).reshape(-1)
        battery_action = float(np.clip(action_values[0], -1.0, 1.0)) if action_values.size else 0.0
        generator_action = float(np.clip(action_values[1], 0.0, 1.0)) if action_values.size >= 2 else 0.0

        if self.residential_generator is not None:
            component_info = self._step_residential_generator(idx, action_values)
        else:
            component_info = dict(getattr(self, "current_component_info", {}))

        other = float(component_info.get("component_other_power", getattr(self, "current_other_power", self._base_other_power(idx))))
        net = load - pv - other
        price = float(self.tou_price[idx])
        p_cmd_raw = self._action_to_power(action_values)
        grid_slack_enabled = bool(getattr(self.config, "grid_slack_enabled", True))
        if grid_slack_enabled:
            p_cmd, feasibility = self._apply_grid_bounds(net, p_cmd_raw)
        else:
            p_cmd = float(p_cmd_raw)
            feasibility = {
                "battery_power_min": float("-inf"),
                "battery_power_max": float("inf"),
                "p_cmd_clipped_by_grid": float(p_cmd),
            }
        p_actual, _, battery_info = self.battery.step(p_cmd, self.config.dt_seconds)
        p_grid = net - p_actual
        dt_hours = self.config.dt_seconds / 3600.0
        import_cost = 0.0
        export_credit = 0.0
        peak_import_penalty = 0.0
        month_id = None
        monthly_peak_increment_kw = 0.0
        monthly_billed_kw = 0.0
        monthly_demand_charge = 0.0
        monthly_peak_increment_penalty = 0.0
        nse_power = 0.0
        curt_power = 0.0
        nse_kwh = 0.0
        curt_kwh = 0.0
        nse_cost = 0.0
        curt_cost = 0.0
        if grid_slack_enabled:
            import_cost = max(p_grid, 0.0) * price * dt_hours / 1000.0
            export_credit = max(-p_grid, 0.0) * self.config.feed_in_tariff * dt_hours / 1000.0
            peak_import_penalty = self._peak_import_penalty(p_grid)
            month_id, monthly_peak_increment_kw, monthly_billed_kw = self._monthly_peak_increment_kw(idx, p_grid)
            monthly_demand_charge = self._monthly_demand_charge(month_id, monthly_peak_increment_kw, monthly_billed_kw)
            monthly_peak_increment_penalty = self._monthly_peak_increment_penalty(monthly_peak_increment_kw)
        else:
            nse_power = max(p_grid, 0.0)
            curt_power = max(-p_grid, 0.0)
            nse_kwh = nse_power * dt_hours / 1000.0
            curt_kwh = curt_power * dt_hours / 1000.0
            nse_cost = max(float(getattr(self.config, "nse_penalty_per_kwh", 0.0)), 0.0) * nse_kwh
            curt_cost = max(float(getattr(self.config, "curtailment_penalty_per_kwh", 0.0)), 0.0) * curt_kwh
        battery_throughput_penalty = self._battery_throughput_penalty(battery_info)
        battery_loss_penalty = self._battery_loss_penalty(battery_info)
        battery_stress_penalty = self._battery_stress_penalty(battery_info)
        component_cost_raw = float(component_info.get("component_operating_cost", 0.0))
        component_cost_objective = component_cost_raw if self.config.include_component_cost_in_objective else 0.0
        cost = (
            import_cost
            - export_credit
            + peak_import_penalty
            + monthly_demand_charge
            + nse_cost
            + curt_cost
            + battery_throughput_penalty
            + battery_loss_penalty
            + battery_stress_penalty
            + component_cost_objective
        )
        self.cumulative_cost += cost
        reward = self._calculate_reward(
            cost,
            battery_info,
            p_grid,
            price,
            monthly_peak_increment_penalty=monthly_peak_increment_penalty,
        )
        self.episode_rewards.append(reward)
        info = self._get_info(idx=idx, other_override=other, component_info=component_info)
        info.update(
            {
                "step": idx,
                "action": battery_action,
                "action_vector": action_values.astype(float).tolist(),
                "battery_action": battery_action,
                "generator_action": generator_action,
                "p_cmd_raw": p_cmd_raw,
                "p_cmd": p_cmd,
                "p_actual": p_actual,
                "p_grid": p_grid,
                "import_cost": float(import_cost),
                "export_credit": float(export_credit),
                "peak_import_penalty": float(peak_import_penalty),
                "monthly_demand_charge": float(monthly_demand_charge),
                "monthly_peak_increment_kw": float(monthly_peak_increment_kw),
                "monthly_peak_increment_penalty": float(monthly_peak_increment_penalty),
                "nse_power": float(nse_power),
                "curtailment_power": float(curt_power),
                "nse_kwh": float(nse_kwh),
                "curtailment_kwh": float(curt_kwh),
                "nse_cost": float(nse_cost),
                "curtailment_cost": float(curt_cost),
                "battery_throughput_penalty": float(battery_throughput_penalty),
                "battery_loss_penalty": float(battery_loss_penalty),
                "battery_stress_penalty": float(battery_stress_penalty),
                "cost": cost,
                "reward": reward,
                "battery_info": battery_info,
                "component_cost_raw": component_cost_raw,
                "component_cost_objective": component_cost_objective,
                **feasibility,
            }
        )
        self.current_step += 1
        terminated = self.current_step >= self.total_steps
        if not terminated:
            self.current_component_info = self._resolve_component_state(self.current_step)
            self.current_other_power = float(self.current_component_info.get("component_other_power", self._base_other_power(self.current_step)))
        self._obs_history.append(self._single_scaled_obs().copy())
        return self._scaled_obs(), reward, terminated, False, info

    def render(self):
        if self.render_mode == "human":
            info = self._get_info()
            print(
                f"{self.config.case_name} | step={info['step']:4d} | hour={info['hour']:2d} | "
                f"SOC={info['soc']:.2%} | SOH={info['soh']:.4f} | cost=${info['cumulative_cost']:.2f}"
            )


class MicrogridEnvThevenin(MicrogridEnv):
    def __init__(self, config: MicrogridConfig | None = None, **kwargs):
        cfg = config or MicrogridConfig()
        cfg.battery_model = "thevenin"
        super().__init__(cfg, **kwargs)


class MicrogridEnvSimple(MicrogridEnv):
    def __init__(self, config: MicrogridConfig | None = None, **kwargs):
        cfg = config or MicrogridConfig()
        cfg.battery_model = "simple"
        super().__init__(cfg, **kwargs)
