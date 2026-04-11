"""Paper-relevant non-RL baselines."""

from __future__ import annotations

from typing import Dict, Mapping, Tuple

import numpy as np
from scipy import sparse
from scipy.optimize import linprog
from tqdm import tqdm

from ..models import BatteryParams
from ..time_utils import dt_hours, month_index_from_timestamps, simulation_steps, steps_per_day


class MILPOptimizer:
    """LP relaxation used as the paper's perfect-foresight benchmark."""

    def __init__(
        self,
        battery_params: BatteryParams | None = None,
        horizon: int = 24,
        efficiency_model: str = "simple",
        feed_in_tariff: float = 0.0,
        grid_import_max: float = float("inf"),
        grid_export_max: float = float("inf"),
        peak_import_penalty_per_kw: float = 0.0,
        peak_import_penalty_threshold_w: float = float("inf"),
        monthly_demand_charge_per_kw: float = 0.0,
        monthly_demand_charge_threshold_w: float = float("inf"),
        battery_throughput_penalty_per_kwh: float = 0.0,
        dt_seconds: float = 3600.0,
    ):
        self.params = battery_params or BatteryParams()
        self.horizon = horizon
        self.efficiency_model = efficiency_model
        self.feed_in_tariff = float(feed_in_tariff)
        self.grid_import_max = float(grid_import_max)
        self.grid_export_max = float(grid_export_max)
        self.peak_import_penalty_per_kw = float(peak_import_penalty_per_kw)
        self.peak_import_penalty_threshold_w = float(peak_import_penalty_threshold_w)
        self.monthly_demand_charge_per_kw = float(monthly_demand_charge_per_kw)
        self.monthly_demand_charge_threshold_w = float(monthly_demand_charge_threshold_w)
        self.battery_throughput_penalty_per_kwh = float(battery_throughput_penalty_per_kwh)
        self.dt_hours = dt_hours(dt_seconds)

    def _efficiency_pair(self) -> tuple[float, float]:
        if self.efficiency_model == "realistic":
            eta_charge = float(np.clip(self.params.eta_charge, 1e-6, 1.0))
            eta_discharge = float(np.clip(self.params.eta_discharge, 1e-6, 1.0))
            return eta_charge, eta_discharge
        return 1.0, 1.0

    def solve(
        self,
        pv_forecast: np.ndarray,
        load_forecast: np.ndarray,
        price_forecast: np.ndarray,
        initial_soc: float,
        other_forecast: np.ndarray | None = None,
        month_index: np.ndarray | None = None,
        initial_monthly_peak_billed_kw: Mapping[int, float] | None = None,
    ) -> Tuple[np.ndarray, float]:
        pv_forecast = np.asarray(pv_forecast, dtype=float)
        load_forecast = np.asarray(load_forecast, dtype=float)
        price_forecast = np.asarray(price_forecast, dtype=float)
        other_forecast = (
            np.zeros_like(load_forecast, dtype=float)
            if other_forecast is None
            else np.asarray(other_forecast, dtype=float)
        )
        horizon = int(len(pv_forecast))
        if horizon <= 0:
            return np.zeros(0, dtype=float), 0.0

        params = self.params
        dt_hours = float(self.dt_hours)
        eta_charge, eta_discharge = self._efficiency_pair()
        energy_cap_wh = float(params.nominal_energy_wh)
        energy_min_wh = float(params.soc_min) * energy_cap_wh
        energy_max_wh = float(params.soc_max) * energy_cap_wh
        initial_energy_wh = float(np.clip(initial_soc, params.soc_min, params.soc_max)) * energy_cap_wh
        net_forecast = load_forecast - pv_forecast - other_forecast

        charge_offset = 0
        discharge_offset = charge_offset + horizon
        import_offset = discharge_offset + horizon
        export_offset = import_offset + horizon
        energy_offset = export_offset + horizon
        next_offset = energy_offset + horizon

        use_peak_penalty = (
            self.peak_import_penalty_per_kw > 0.0
            and np.isfinite(self.peak_import_penalty_threshold_w)
        )
        peak_offset = next_offset
        if use_peak_penalty:
            next_offset += horizon

        month_peak_ids: list[int] = []
        month_peak_offset_map: dict[int, int] = {}
        if (
            self.monthly_demand_charge_per_kw > 0.0
            and np.isfinite(self.monthly_demand_charge_threshold_w)
            and month_index is not None
        ):
            raw_month_index = np.asarray(month_index, dtype=int).reshape(-1)
            if raw_month_index.size != horizon:
                raise ValueError("MILP baseline: month_index length must match the forecast horizon")
            month_peak_ids = [int(month_id) for month_id in np.unique(raw_month_index)]
            month_peak_offset_map = {
                int(month_id): next_offset + idx for idx, month_id in enumerate(month_peak_ids)
            }
            next_offset += len(month_peak_ids)
        else:
            raw_month_index = np.zeros(horizon, dtype=int)

        n_vars = next_offset
        objective = np.zeros(n_vars, dtype=float)
        objective[import_offset:export_offset] = price_forecast * dt_hours / 1000.0
        objective[export_offset:energy_offset] = -self.feed_in_tariff * dt_hours / 1000.0
        if self.battery_throughput_penalty_per_kwh > 0.0:
            throughput_cost = self.battery_throughput_penalty_per_kwh * dt_hours / 1000.0
            objective[charge_offset:discharge_offset] += throughput_cost
            objective[discharge_offset:import_offset] += throughput_cost
        if use_peak_penalty:
            objective[peak_offset:peak_offset + horizon] = self.peak_import_penalty_per_kw * dt_hours
        for month_id, col in month_peak_offset_map.items():
            del month_id
            objective[col] = self.monthly_demand_charge_per_kw

        bounds: list[tuple[float | None, float | None]] = []
        bounds.extend([(0.0, float(params.p_charge_max))] * horizon)
        bounds.extend([(0.0, float(params.p_discharge_max))] * horizon)
        import_bound = None if not np.isfinite(self.grid_import_max) else max(self.grid_import_max * 1_000_000.0, 0.0)
        export_bound = None if not np.isfinite(self.grid_export_max) else max(self.grid_export_max * 1_000_000.0, 0.0)
        bounds.extend([(0.0, import_bound)] * horizon)
        bounds.extend([(0.0, export_bound)] * horizon)
        bounds.extend([(energy_min_wh, energy_max_wh)] * horizon)
        if use_peak_penalty:
            bounds.extend([(0.0, None)] * horizon)
        baseline_constant_cost = 0.0
        baseline_peaks = {
            int(month_id): float(value)
            for month_id, value in (initial_monthly_peak_billed_kw or {}).items()
        }
        for month_id in month_peak_ids:
            lower_kw = max(float(baseline_peaks.get(int(month_id), 0.0)), 0.0)
            bounds.append((lower_kw, None))
            baseline_constant_cost += self.monthly_demand_charge_per_kw * lower_kw

        a_eq = sparse.lil_matrix((2 * horizon, n_vars), dtype=float)
        b_eq = np.zeros(2 * horizon, dtype=float)
        for step in range(horizon):
            power_row = step
            a_eq[power_row, charge_offset + step] = -1.0
            a_eq[power_row, discharge_offset + step] = 1.0
            a_eq[power_row, import_offset + step] = 1.0
            a_eq[power_row, export_offset + step] = -1.0
            b_eq[power_row] = float(net_forecast[step])

            energy_row = horizon + step
            a_eq[energy_row, energy_offset + step] = 1.0
            a_eq[energy_row, charge_offset + step] = -eta_charge * dt_hours
            a_eq[energy_row, discharge_offset + step] = dt_hours / eta_discharge
            if step > 0:
                a_eq[energy_row, energy_offset + step - 1] = -1.0
                b_eq[energy_row] = 0.0
            else:
                b_eq[energy_row] = initial_energy_wh

        ub_rows = 0
        if use_peak_penalty:
            ub_rows += horizon
        if month_peak_offset_map:
            ub_rows += horizon
        a_ub = sparse.lil_matrix((ub_rows, n_vars), dtype=float)
        b_ub = np.zeros(ub_rows, dtype=float)

        ub_row = 0
        if use_peak_penalty:
            for step in range(horizon):
                a_ub[ub_row, import_offset + step] = 1.0
                a_ub[ub_row, peak_offset + step] = -1000.0
                b_ub[ub_row] = self.peak_import_penalty_threshold_w
                ub_row += 1
        if month_peak_offset_map:
            for step in range(horizon):
                month_id = int(raw_month_index[step])
                a_ub[ub_row, import_offset + step] = 1.0
                a_ub[ub_row, month_peak_offset_map[month_id]] = -1000.0
                b_ub[ub_row] = self.monthly_demand_charge_threshold_w
                ub_row += 1

        result = linprog(
            objective,
            A_ub=a_ub.tocsr() if ub_rows > 0 else None,
            b_ub=b_ub if ub_rows > 0 else None,
            A_eq=a_eq.tocsr(),
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )
        if not result.success:
            raise RuntimeError(f"MILP baseline failed: {result.message}")

        charge_schedule = np.asarray(result.x[charge_offset:discharge_offset], dtype=float)
        discharge_schedule = np.asarray(result.x[discharge_offset:import_offset], dtype=float)
        battery_schedule = discharge_schedule - charge_schedule
        objective_value = float(result.fun - baseline_constant_cost)
        return battery_schedule, objective_value


class RuleBasedController:
    """TOU-aware charging/discharging heuristic."""

    def __init__(self, battery_params: BatteryParams | None = None, valley_price: float = 0.39073, peak_price: float = 0.51373):
        self.params = battery_params or BatteryParams()
        self.valley_price = valley_price
        self.peak_price = peak_price

    def get_action(self, pv_power: float, load_power: float, price: float, soc: float) -> float:
        del load_power
        if price <= self.valley_price and soc < min(0.8, self.params.soc_max):
            return -self.params.p_charge_max
        if price >= self.peak_price and soc > max(0.2, self.params.soc_min):
            return self.params.p_discharge_max
        if pv_power > 0 and soc < self.params.soc_max:
            return -min(self.params.p_charge_max, pv_power)
        return 0.0


def _extract_network_env_forecasts(env, total_steps: int) -> dict[str, np.ndarray]:
    if not hasattr(env, "_profiles"):
        raise TypeError("Dispatch baseline expects a network environment with '_profiles' forecasts")
    if total_steps > int(getattr(env, "total_steps", total_steps)):
        raise ValueError("Requested baseline horizon exceeds environment total_steps")
    profiles = env._profiles
    load_w = np.asarray(profiles.load_w[:total_steps], dtype=float)
    pv_w = np.asarray(profiles.pv_w[:total_steps], dtype=float)
    price = np.asarray(profiles.price[:total_steps], dtype=float)
    if load_w.shape != pv_w.shape or load_w.shape != price.shape:
        raise ValueError("Network forecast arrays must share the same shape")
    timestamps = getattr(profiles, "timestamps", None)
    if timestamps is None or len(timestamps) < total_steps:
        raise ValueError("Network baseline expects timestamped profiles with at least total_steps entries")
    return {
        "load_w": load_w,
        "pv_w": pv_w,
        "price": price,
        "other_w": np.zeros_like(load_w, dtype=float),
        "month_index": month_index_from_timestamps(timestamps[:total_steps]),
        "timestamps": np.asarray(timestamps[:total_steps]),
    }


def _battery_command_to_action(env, power_w: float) -> np.ndarray:
    params = env.battery.params
    if power_w >= 0.0:
        scale = max(float(params.p_discharge_max), 1e-9)
    else:
        scale = max(float(params.p_charge_max), 1e-9)
    battery_action = float(np.clip(power_w / scale if scale > 0.0 else 0.0, -1.0, 1.0))
    if int(np.prod(env.action_space.shape)) > 1:
        return np.array([battery_action, 0.0], dtype=float)
    return np.array([battery_action], dtype=float)


def _collect_history(env, total_steps: int, controller_name: str, action_fn, post_step_fn=None) -> Dict:
    env.reset()
    steps = []
    soc_history = []
    cost_history = []
    pv_history = []
    load_history = []
    grid_history = []
    power_history = []
    price_history = []
    current_history = []
    voltage_history = []
    efficiency_history = []
    loss_history = []
    r_int_history = []
    v_ocv_history = []

    for step in tqdm(range(total_steps), desc=controller_name):
        idx = min(env.current_step, env.total_steps - 1)
        power = action_fn(idx)
        action = _battery_command_to_action(env, power)
        _, _, terminated, truncated, info = env.step(action)
        battery_info = {
            "soc": float(info.get("soc", getattr(env.battery, "soc", 0.0))),
            "current": float(info.get("current", 0.0)),
            "voltage": float(info.get("voltage", 0.0)),
            "efficiency": float(info.get("efficiency", 1.0)),
            "power_loss": float(info.get("power_loss", 0.0)),
            "r_int": float(info.get("r_int", 0.0)),
            "v_ocv": float(info.get("v_ocv", 0.0)),
        }
        if post_step_fn is not None:
            post_step_fn(step, info)
        steps.append(step)
        soc_history.append(info["soc"])
        cost_history.append(info["cumulative_cost"])
        pv_history.append(info["pv_w"])
        load_history.append(info["load_w"])
        grid_history.append(float(info.get("slack_active_power_mw", 0.0)) * 1_000_000.0)
        power_history.append(info["battery_power_w"])
        price_history.append(info["price"])
        current_history.append(battery_info.get("current", 0.0))
        voltage_history.append(battery_info.get("voltage", 0.0))
        efficiency_history.append(battery_info.get("efficiency", 1.0))
        loss_history.append(battery_info.get("power_loss", 0.0))
        r_int_history.append(battery_info.get("r_int", 0.0))
        v_ocv_history.append(battery_info.get("v_ocv", 0.0))
        if terminated or truncated:
            break

    final_cost = cost_history[-1] if cost_history else 0.0
    return {
        "name": controller_name,
        "total_cost": final_cost,
        "steps": steps,
        "soc": soc_history,
        "cost": cost_history,
        "pv": pv_history,
        "load": load_history,
        "grid": grid_history,
        "battery_power": power_history,
        "price": price_history,
        "current": current_history,
        "voltage": voltage_history,
        "efficiency": efficiency_history,
        "power_loss": loss_history,
        "r_int": r_int_history,
        "v_ocv": v_ocv_history,
    }


def run_milp_baseline(
    env,
    simulation_days: int = 365,
    horizon: int = 24,
    efficiency_model: str = "simple",
    name: str = "MILP Oracle",
    chunk_days: int | None = None,
) -> Dict:
    if not bool(getattr(env.config, "grid_slack_enabled", True)):
        raise ValueError("MILP baseline currently supports only grid-connected cases with grid_slack_enabled=True")
    env_steps_per_day = steps_per_day(getattr(env.config, "dt_seconds", 3600.0))
    if chunk_days is not None:
        horizon_days = int(chunk_days)
        horizon = total_horizon = simulation_steps(simulation_days, env.config.dt_seconds) if horizon_days <= 0 else max(horizon_days, 1) * env_steps_per_day
    else:
        total_horizon = simulation_steps(simulation_days, env.config.dt_seconds)
        horizon = int(horizon) * max(env_steps_per_day // 24, 1)
    forecasts = _extract_network_env_forecasts(env, total_horizon)
    optimizer = MILPOptimizer(
        env.battery.params,
        horizon=int(horizon),
        efficiency_model=efficiency_model,
        feed_in_tariff=float(getattr(env.config, "feed_in_tariff", 0.0)),
        grid_import_max=float(getattr(env.config, "grid_import_max", float("inf"))),
        grid_export_max=float(getattr(env.config, "grid_export_max", float("inf"))),
        peak_import_penalty_per_kw=float(getattr(env.config, "peak_import_penalty_per_kw", 0.0)),
        peak_import_penalty_threshold_w=float(getattr(env.config, "peak_import_penalty_threshold_w", float("inf"))),
        monthly_demand_charge_per_kw=float(getattr(env.config, "monthly_demand_charge_per_kw", 0.0)),
        monthly_demand_charge_threshold_w=float(getattr(env.config, "monthly_demand_charge_threshold_w", float("inf"))),
        battery_throughput_penalty_per_kwh=float(getattr(env.config, "battery_throughput_penalty_per_kwh", 0.0)),
        dt_seconds=float(getattr(env.config, "dt_seconds", 3600.0)),
    )
    total_steps = simulation_steps(simulation_days, env.config.dt_seconds)
    realized_monthly_peak_billed_kw: dict[int, float] = {}

    if int(horizon) >= total_steps:
        schedule, _ = optimizer.solve(
            forecasts["pv_w"][:total_steps],
            forecasts["load_w"][:total_steps],
            forecasts["price"][:total_steps],
            env.battery.soc,
            other_forecast=forecasts["other_w"][:total_steps],
            month_index=forecasts["month_index"][:total_steps],
            initial_monthly_peak_billed_kw={},
        )

        def action_fn(step: int) -> float:
            return float(schedule[step])

    else:

        def action_fn(step: int) -> float:
            lookahead = min(int(horizon), total_steps - step)
            schedule, _ = optimizer.solve(
                forecasts["pv_w"][step : step + lookahead],
                forecasts["load_w"][step : step + lookahead],
                forecasts["price"][step : step + lookahead],
                env.battery.soc,
                other_forecast=forecasts["other_w"][step : step + lookahead],
                month_index=forecasts["month_index"][step : step + lookahead],
                initial_monthly_peak_billed_kw=realized_monthly_peak_billed_kw,
            )
            return float(schedule[0])

    def post_step_fn(step: int, info: dict) -> None:
        month_id = int(forecasts["month_index"][step])
        import_kw = max(float(info.get("grid_import_mw", 0.0)) * 1000.0, 0.0)
        realized_monthly_peak_billed_kw[month_id] = max(realized_monthly_peak_billed_kw.get(month_id, 0.0), import_kw)

    return _collect_history(env, total_steps, name, action_fn, post_step_fn=post_step_fn)


def run_rule_based_baseline(env, simulation_days: int = 365, name: str = "Rule Based") -> Dict:
    controller = RuleBasedController(env.battery.params)
    total_steps = simulation_steps(simulation_days, env.config.dt_seconds)
    forecasts = _extract_network_env_forecasts(env, total_steps)

    def action_fn(step: int) -> float:
        return controller.get_action(
            pv_power=float(forecasts["pv_w"][step]),
            load_power=float(forecasts["load_w"][step]),
            price=float(forecasts["price"][step]),
            soc=float(env.battery.soc),
        )

    return _collect_history(env, total_steps, name, action_fn)
