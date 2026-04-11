"""Pandapower-backed network microgrid environment."""

from __future__ import annotations

from dataclasses import replace

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from ..cases import CIGREEuropeanLVConfig, NetworkCaseConfig
from ..data.network_profiles import NetworkProfiles, describe_network_regime, load_network_profiles, normalize_network_regime
from ..models.battery import SimpleBattery, TheveninBattery
from ..network.adapters.injection_mapper import apply_power_injections, initialize_injection_state
from ..network.adapters.pandapower_runner import run_power_flow
from ..network.builders.cigre_lv import build_cigre_european_lv_network
from ..network.builders.ieee33 import build_ieee33_network
from ..network.metrics import extract_network_metrics
from ..time_utils import hours_to_steps, simulation_steps, steps_per_hour
from .observation_builder import OBSERVATION_SIZE, build_network_observation
from .reward_builder import build_network_reward


class NetworkMicrogridEnv(gym.Env):
    """Network-level microgrid environment using pandapower."""

    metadata = {"render_modes": []}

    def __init__(self, config: NetworkCaseConfig | None = None):
        super().__init__()
        self.config = config or CIGREEuropeanLVConfig()
        self.total_steps = simulation_steps(self.config.simulation_days, self.config.dt_seconds)
        self._full_profiles = load_network_profiles(self.config)
        if len(self._full_profiles.load_w) < self.total_steps:
            raise ValueError(
                f"Profile length {len(self._full_profiles.load_w)} is shorter than required episode length {self.total_steps}"
            )
        self._episode_start_step = 0
        self._profiles = self._slice_profiles(0)
        self.net = self._build_network()
        self.injection_state = initialize_injection_state(self.net)
        self.battery = self._build_battery()
        self.current_step = 0
        self.cumulative_cost = 0.0
        self.cumulative_terminal_soc_penalty_cost = 0.0
        self.cumulative_objective_cost = 0.0
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(OBSERVATION_SIZE,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def _slice_profiles(self, start_step: int) -> NetworkProfiles:
        end_step = int(start_step) + self.total_steps
        profiles = self._full_profiles
        return NetworkProfiles(
            load_w=np.asarray(profiles.load_w[start_step:end_step], dtype=float),
            pv_w=np.asarray(profiles.pv_w[start_step:end_step], dtype=float),
            price=np.asarray(profiles.price[start_step:end_step], dtype=float),
            timestamps=pd.DatetimeIndex(profiles.timestamps[start_step:end_step]),
        )

    def _resolve_episode_start_step(self) -> int:
        max_start = max(len(self._full_profiles.load_w) - self.total_steps, 0)
        if not bool(getattr(self.config, "random_episode_start", False)):
            requested = max(hours_to_steps(getattr(self.config, "episode_start_hour", 0), self.config.dt_seconds), 0)
            return min(requested, max_start)

        explicit_hours = tuple(
            int(value) for value in getattr(self.config, "full_year_random_start_hours", tuple()) if int(value) >= 0
        )
        if explicit_hours:
            candidates = [min(hours_to_steps(hour, self.config.dt_seconds), max_start) for hour in explicit_hours]
        else:
            stride_hours = max(int(getattr(self.config, "full_year_random_start_stride_hours", 1)), 1)
            stride_steps = max(hours_to_steps(stride_hours, self.config.dt_seconds), 1)
            candidates = list(range(0, max_start + 1, stride_steps))
        if not candidates:
            return 0
        return int(self.np_random.choice(np.asarray(candidates, dtype=int)))

    def _build_network(self):
        if self.config.case_key == "ieee33_network":
            return build_ieee33_network()
        if self.config.case_key == "cigre_eu_lv_network":
            return build_cigre_european_lv_network()
        raise ValueError(f"Unsupported network case_key: {self.config.case_key}")

    def _build_battery(self):
        if self.config.battery_model == "none":
            return SimpleBattery(self.config.battery_params)
        if self.config.battery_model == "simple":
            return SimpleBattery(self.config.battery_params)
        if self.config.battery_model in {"thevenin", "thevenin_loss_only"}:
            return TheveninBattery(self.config.battery_params)
        raise ValueError(f"Unsupported battery_model: {self.config.battery_model}")

    def _reset_soc_for_regime(self) -> float | None:
        regime = normalize_network_regime(getattr(self.config, "regime", "base"))
        if regime == "tight_soc":
            lower = max(float(self.config.battery_params.soc_min), 0.15)
            upper = min(float(self.config.battery_params.soc_max), 0.30)
            return float(self.np_random.uniform(lower, max(lower, upper)))
        if self.config.random_initial_soc:
            return float(self.np_random.uniform(float(self.config.initial_soc_min), float(self.config.initial_soc_max)))
        return None

    def _battery_power_command(self, action: np.ndarray) -> float:
        action_array = np.asarray(action, dtype=float).reshape(-1)
        if action_array.size == 0:
            raise ValueError("Action must contain at least one scalar battery command")
        scalar = float(np.clip(action_array[0], -1.0, 1.0))
        if scalar >= 0.0:
            return scalar * float(self.config.battery_params.p_discharge_max)
        return scalar * float(self.config.battery_params.p_charge_max)

    def _default_metrics(self) -> dict[str, float]:
        return {
            "min_bus_voltage_pu": 0.0,
            "max_bus_voltage_pu": 0.0,
            "max_line_loading_pct": 0.0,
            "max_line_current_ka": 0.0,
            "mean_line_loading_pct": 0.0,
            "max_transformer_loading_pct": 0.0,
            "slack_active_power_mw": 0.0,
        }

    def _grid_exchange_summary(self, slack_mw: float, price: float) -> dict[str, float]:
        dt_hours = float(self.config.dt_seconds) / 3600.0
        import_mw = max(float(slack_mw), 0.0)
        export_mw = max(-float(slack_mw), 0.0)
        import_cost = import_mw * 1000.0 * float(price) * dt_hours
        export_revenue = export_mw * 1000.0 * float(self.config.feed_in_tariff) * dt_hours
        import_violation_mw = (
            max(import_mw - max(float(self.config.grid_import_max), 0.0), 0.0)
            if np.isfinite(self.config.grid_import_max)
            else 0.0
        )
        export_violation_mw = (
            max(export_mw - max(float(self.config.grid_export_max), 0.0), 0.0)
            if np.isfinite(self.config.grid_export_max)
            else 0.0
        )
        grid_limit_penalty_cost = (
            (import_violation_mw + export_violation_mw)
            * 1000.0
            * float(self.config.grid_limit_violation_penalty_per_kwh)
            * dt_hours
        )
        net_energy_cost = import_cost - export_revenue
        return {
            "grid_import_mw": float(import_mw),
            "grid_export_mw": float(export_mw),
            "import_cost": float(import_cost),
            "export_revenue": float(export_revenue),
            "net_energy_cost": float(net_energy_cost),
            "grid_import_limit_violation_mw": float(import_violation_mw),
            "grid_export_limit_violation_mw": float(export_violation_mw),
            "grid_limit_penalty_cost": float(grid_limit_penalty_cost),
            "total_grid_cost": float(net_energy_cost + grid_limit_penalty_cost),
        }

    def _info(
        self,
        load_w: float,
        pv_w: float,
        price: float,
        battery_power_w: float,
        battery_info: dict,
        metrics: dict,
        penalties: dict[str, float],
        timestamp,
        grid_summary: dict[str, float] | None = None,
        power_flow_result: dict | None = None,
    ) -> dict:
        power_flow_result = dict(power_flow_result or {})
        grid_summary = dict(grid_summary or self._grid_exchange_summary(float(metrics.get("slack_active_power_mw", 0.0)), price))
        return {
            "step": int(self.current_step),
            "timestamp": str(pd.Timestamp(timestamp)),
            "case_name": self.config.case_name,
            "regime": str(getattr(self.config, "regime", "base")),
            "regime_description": describe_network_regime(getattr(self.config, "regime", "base")),
            "episode_start_step": int(self._episode_start_step),
            "episode_start_hour": int(round(self._episode_start_step / max(steps_per_hour(self.config.dt_seconds), 1))),
            "load_w": float(load_w),
            "pv_w": float(pv_w),
            "price": float(price),
            "battery_power_w": float(battery_power_w),
            "battery_power_mw": float(battery_power_w) / 1_000_000.0,
            "cumulative_cost": float(self.cumulative_cost),
            "cumulative_terminal_soc_penalty_cost": float(self.cumulative_terminal_soc_penalty_cost),
            "cumulative_objective_cost": float(self.cumulative_objective_cost),
            "power_flow_converged": bool(power_flow_result.get("converged", True)),
            "power_flow_failed": bool(power_flow_result.get("failed", False)),
            "power_flow_failure_reason": str(power_flow_result.get("failure_reason", "")),
            **grid_summary,
            **battery_info,
            **metrics,
            **penalties,
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        del options
        self.current_step = 0
        self.cumulative_cost = 0.0
        self.cumulative_terminal_soc_penalty_cost = 0.0
        self.cumulative_objective_cost = 0.0
        soc = self._reset_soc_for_regime()
        self.battery.reset(soc=soc)
        self.net = self._build_network()
        self.injection_state = initialize_injection_state(self.net)
        self._episode_start_step = self._resolve_episode_start_step()
        self._profiles = self._slice_profiles(self._episode_start_step)
        load_w = float(self._profiles.load_w[self.current_step])
        pv_w = float(self._profiles.pv_w[self.current_step])
        price = float(self._profiles.price[self.current_step])
        timestamp = self._profiles.timestamps[self.current_step]
        apply_power_injections(self.net, self.injection_state, load_w=load_w, pv_w=pv_w, battery_power_w=0.0)
        power_flow_result = run_power_flow(self.net)
        metrics = extract_network_metrics(self.net) if power_flow_result.get("converged", False) else self._default_metrics()
        battery_info = {
            "soc": float(self.battery.soc),
            "soc_violation": 0.0,
            "current": 0.0,
            "power_loss": 0.0,
            "effective_power": 0.0,
            "actual_power": 0.0,
            "p_max": float(self.config.battery_params.p_discharge_max),
            "r_int": 0.0,
            "r_int_power_factor": 1.0,
            "temperature_c": float(getattr(self.battery, "temperature_c", self.config.battery_params.temperature_init_c)),
        }
        penalties = {
            "undervoltage": 0.0,
            "overvoltage": 0.0,
            "line_overload_pct": 0.0,
            "transformer_overload_pct": 0.0,
            "grid_import_limit_violation_mw": 0.0,
            "grid_export_limit_violation_mw": 0.0,
            "grid_limit_penalty_cost": 0.0,
            "battery_throughput_kwh": 0.0,
            "battery_loss_kwh": 0.0,
            "battery_stress_kwh": 0.0,
            "terminal_soc_target": float(self.config.terminal_soc_target if self.config.terminal_soc_target is not None else self.config.battery_params.soc_init),
            "terminal_soc_tolerance": float(getattr(self.config, "terminal_soc_tolerance", 0.0)),
            "terminal_soc_deviation": 0.0,
            "terminal_soc_excess": 0.0,
            "terminal_soc_penalty": 0.0,
            "power_flow_failure_penalty": 0.0,
        }
        grid_summary = self._grid_exchange_summary(float(metrics.get("slack_active_power_mw", 0.0)), price)
        obs = build_network_observation(
            self.config,
            self.battery,
            load_w,
            pv_w,
            price,
            self.current_step,
            self.total_steps,
            metrics,
            battery_info=battery_info,
            timestamp=timestamp,
        )
        return obs, self._info(
            load_w,
            pv_w,
            price,
            0.0,
            battery_info,
            metrics,
            penalties,
            timestamp=timestamp,
            grid_summary=grid_summary,
            power_flow_result=power_flow_result,
        )

    def step(self, action: np.ndarray):
        idx = min(self.current_step, self.total_steps - 1)
        load_w = float(self._profiles.load_w[idx])
        pv_w = float(self._profiles.pv_w[idx])
        price = float(self._profiles.price[idx])
        timestamp = self._profiles.timestamps[idx]
        battery_command_w = self._battery_power_command(action)
        battery_power_w, _, battery_info = self.battery.step(battery_command_w, self.config.dt_seconds)
        apply_power_injections(self.net, self.injection_state, load_w=load_w, pv_w=pv_w, battery_power_w=battery_power_w)
        power_flow_result = run_power_flow(self.net)
        metrics = extract_network_metrics(self.net) if power_flow_result.get("converged", False) else self._default_metrics()
        grid_summary = self._grid_exchange_summary(float(metrics.get("slack_active_power_mw", 0.0)), price)
        total_grid_cost = float(grid_summary["total_grid_cost"])
        self.cumulative_cost += total_grid_cost
        horizon_reached = (self.current_step + 1) >= self.total_steps
        reward, penalties = build_network_reward(
            self.config,
            battery_info,
            metrics,
            total_grid_cost,
            power_flow_result=power_flow_result,
            is_terminal=horizon_reached,
        )
        terminal_soc_penalty_cost = float(penalties.get("terminal_soc_penalty", 0.0))
        self.cumulative_terminal_soc_penalty_cost += terminal_soc_penalty_cost
        self.cumulative_objective_cost = self.cumulative_cost + self.cumulative_terminal_soc_penalty_cost
        penalties.update(
            {
                "grid_import_limit_violation_mw": float(grid_summary["grid_import_limit_violation_mw"]),
                "grid_export_limit_violation_mw": float(grid_summary["grid_export_limit_violation_mw"]),
                "grid_limit_penalty_cost": float(grid_summary["grid_limit_penalty_cost"]),
            }
        )
        self.current_step += 1
        power_flow_failed = bool(power_flow_result.get("failed", False) or not power_flow_result.get("converged", True))
        terminated = bool(power_flow_failed)
        truncated = bool(horizon_reached)
        next_idx = min(self.current_step, self.total_steps - 1)
        obs = build_network_observation(
            self.config,
            self.battery,
            float(self._profiles.load_w[next_idx]),
            float(self._profiles.pv_w[next_idx]),
            float(self._profiles.price[next_idx]),
            next_idx,
            self.total_steps,
            metrics,
            battery_info=battery_info,
            timestamp=self._profiles.timestamps[next_idx],
        )
        info = self._info(
            load_w,
            pv_w,
            price,
            battery_power_w,
            battery_info,
            metrics,
            penalties,
            timestamp=timestamp,
            grid_summary=grid_summary,
            power_flow_result=power_flow_result,
        )
        return obs, reward, terminated, truncated, info


class NetworkMicrogridEnvThevenin(NetworkMicrogridEnv):
    def __init__(self, config: NetworkCaseConfig | None = None):
        cfg = replace(config, battery_model="thevenin") if config is not None else CIGREEuropeanLVConfig(battery_model="thevenin")
        super().__init__(cfg)


class NetworkMicrogridEnvSimple(NetworkMicrogridEnv):
    def __init__(self, config: NetworkCaseConfig | None = None):
        cfg = replace(config, battery_model="simple") if config is not None else CIGREEuropeanLVConfig(battery_model="simple")
        super().__init__(cfg)
