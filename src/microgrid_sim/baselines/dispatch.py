"""Paper-relevant non-RL baselines."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.optimize import linprog
from tqdm import tqdm

from ..models import BatteryParams


class MILPOptimizer:
    """LP relaxation used as the paper's perfect-foresight benchmark."""

    def __init__(self, battery_params: BatteryParams | None = None, horizon: int = 24, efficiency_model: str = "simple"):
        self.params = battery_params or BatteryParams()
        self.horizon = horizon
        self.efficiency_model = efficiency_model

    def solve(self, pv_forecast: np.ndarray, load_forecast: np.ndarray, price_forecast: np.ndarray, initial_soc: float) -> Tuple[np.ndarray, float]:
        horizon = len(pv_forecast)
        params = self.params
        dt_hours = 1.0
        energy_cap = params.nominal_energy_wh
        eta = 0.95 if self.efficiency_model == "realistic" else 1.0
        n_vars = 2 * horizon
        objective = np.zeros(n_vars)
        objective[horizon:] = price_forecast * dt_hours / 1000.0
        a_ub: list[np.ndarray] = []
        b_ub: list[float] = []

        for step in range(horizon):
            row = np.zeros(n_vars)
            row[step] = -1.0
            row[horizon + step] = -1.0
            a_ub.append(row)
            b_ub.append(pv_forecast[step] - load_forecast[step])

        for step in range(horizon):
            coeff = np.zeros(n_vars)
            coeff[: step + 1] = dt_hours / (energy_cap * eta)
            a_ub.append(coeff)
            b_ub.append(initial_soc - params.soc_min)
            a_ub.append(-coeff)
            b_ub.append(params.soc_max - initial_soc)

        bounds = [(-params.p_charge_max, params.p_discharge_max)] * horizon + [(0.0, None)] * horizon
        result = linprog(objective, A_ub=np.array(a_ub), b_ub=np.array(b_ub), bounds=bounds, method="highs")
        if not result.success:
            raise RuntimeError(f"MILP baseline failed: {result.message}")
        return result.x[:horizon], float(result.fun)


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


def _collect_history(env, total_steps: int, controller_name: str, action_fn) -> Dict:
    env.reset()
    steps = []
    soc_history = []
    soh_history = []
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
        battery_action = power / env.battery.params.p_discharge_max if power >= 0 else power / env.battery.params.p_charge_max
        if int(np.prod(env.action_space.shape)) > 1:
            action = np.array([battery_action, 0.0], dtype=float)
        else:
            action = np.array([battery_action], dtype=float)
        _, _, terminated, truncated, info = env.step(action)
        battery_info = info["battery_info"]
        steps.append(step)
        soc_history.append(info["soc"])
        soh_history.append(info["soh"])
        cost_history.append(info["cumulative_cost"])
        pv_history.append(info["pv_power"])
        load_history.append(info["load_power"])
        grid_history.append(info["p_grid"])
        power_history.append(info["p_actual"])
        price_history.append(info["price"])
        current_history.append(battery_info.get("current", 0.0))
        voltage_history.append(battery_info.get("voltage", 0.0))
        efficiency_history.append(battery_info.get("efficiency", 1.0))
        loss_history.append(battery_info.get("power_loss", 0.0))
        r_int_history.append(battery_info.get("r_int", 0.0))
        v_ocv_history.append(battery_info.get("v_ocv", 0.0))
        if terminated or truncated:
            break

    final_soh = soh_history[-1] if soh_history else 1.0
    final_cost = cost_history[-1] if cost_history else 0.0
    return {
        "name": controller_name,
        "total_cost": final_cost,
        "final_soh": final_soh,
        "soh_degradation": (1.0 - final_soh) * 100.0,
        "steps": steps,
        "soc": soc_history,
        "soh": soh_history,
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
    if chunk_days is not None:
        horizon = int(max(int(chunk_days), 1) * 24)
    optimizer = MILPOptimizer(env.battery.params, horizon=int(horizon), efficiency_model=efficiency_model)
    total_steps = simulation_days * 24

    def action_fn(step: int) -> float:
        lookahead = min(int(horizon), len(env.pv_power) - step)
        schedule, _ = optimizer.solve(
            env.pv_power[step : step + lookahead],
            env.load_power[step : step + lookahead],
            env.tou_price[step : step + lookahead],
            env.battery.soc,
        )
        return float(schedule[0])

    return _collect_history(env, total_steps, name, action_fn)


def run_rule_based_baseline(env, simulation_days: int = 365, name: str = "Rule Based") -> Dict:
    controller = RuleBasedController(env.battery.params)
    total_steps = simulation_days * 24

    def action_fn(step: int) -> float:
        return controller.get_action(
            pv_power=float(env.pv_power[step]),
            load_power=float(env.load_power[step]),
            price=float(env.tou_price[step]),
            soc=float(env.battery.soc),
        )

    return _collect_history(env, total_steps, name, action_fn)
