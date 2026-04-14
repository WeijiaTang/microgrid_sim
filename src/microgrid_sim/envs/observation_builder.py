"""Observation construction for the network microgrid environment."""

from __future__ import annotations

import numpy as np
import pandas as pd


OBSERVATION_SIZE = 30


def _normalized_battery_action(power_w: float, battery_params) -> float:
    power = float(power_w)
    if power >= 0.0:
        scale = max(float(getattr(battery_params, "p_discharge_max", 0.0)), 1e-9)
    else:
        scale = max(float(getattr(battery_params, "p_charge_max", 0.0)), 1e-9)
    return float(np.clip(power / scale, -1.0, 1.0))


def _soc_feasible_action_bounds(config, battery) -> tuple[float, float, float, float]:
    params = getattr(config, "battery_params", None)
    if params is None:
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


def _rule_based_action_hint(config, battery, load_w: float, pv_w: float, price: float) -> float:
    params = getattr(config, "battery_params", None)
    if params is None:
        return 0.0
    soc = float(getattr(battery, "soc", getattr(params, "soc_init", 0.5)))
    soc_max = float(getattr(params, "soc_max", 1.0))
    soc_min = float(getattr(params, "soc_min", 0.0))
    valley_price = 0.39073
    peak_price = 0.51373
    if float(price) <= valley_price and soc < min(0.8, soc_max):
        return -1.0
    if float(price) >= peak_price and soc > max(0.2, soc_min):
        return 1.0
    if float(pv_w) > 0.0 and soc < soc_max:
        return _normalized_battery_action(-min(float(getattr(params, "p_charge_max", 0.0)), float(pv_w)), params)
    del load_w
    return 0.0


def _year_phase_features(step: int, total_steps: int, timestamp) -> tuple[float, float]:
    if timestamp is None:
        progress = float(step) / max(float(total_steps - 1), 1.0)
        return float(np.sin(2.0 * np.pi * progress)), float(np.cos(2.0 * np.pi * progress))
    ts = pd.Timestamp(timestamp)
    year_days = 366.0 if bool(ts.is_leap_year) else 365.0
    hour = float(ts.hour) + float(ts.minute) / 60.0 + float(ts.second) / 3600.0
    phase = ((float(ts.dayofyear) - 1.0) + hour / 24.0) / year_days
    return float(np.sin(2.0 * np.pi * phase)), float(np.cos(2.0 * np.pi * phase))


def build_network_observation(
    config,
    battery,
    load_w: float,
    pv_w: float,
    price: float,
    step: int,
    total_steps: int,
    metrics: dict[str, float],
    battery_info: dict[str, float] | None = None,
    timestamp=None,
) -> np.ndarray:
    battery_info = dict(battery_info or {})
    if timestamp is None:
        hour = step % 24
        timestamp_obj = None
    else:
        ts = pd.Timestamp(timestamp)
        hour = float(ts.hour) + float(ts.minute) / 60.0
        timestamp_obj = ts
    progress = float(step) / max(float(total_steps - 1), 1.0)
    battery_power_w = float(battery_info.get("actual_power", battery_info.get("effective_power", 0.0)))
    current_a = float(battery_info.get("current", 0.0))
    power_loss_w = float(battery_info.get("power_loss", 0.0))
    p_max_w = max(float(battery_info.get("p_max", config.battery_params.p_discharge_max)), 1e-9)
    r_int = float(battery_info.get("r_int", 0.0))
    line_limit_pct = max(float(config.network_line_loading_limit_pct), 1e-9)
    soc = float(getattr(battery, "soc", 0.5))
    soc_min = float(getattr(config.battery_params, "soc_min", 0.0))
    soc_max = float(getattr(config.battery_params, "soc_max", 1.0))
    terminal_soc_target = getattr(config, "terminal_soc_target", None)
    if terminal_soc_target is None:
        terminal_soc_target = getattr(config.battery_params, "soc_init", soc)
    soc_distance_to_target = float(np.clip(soc - float(terminal_soc_target), -1.0, 1.0))
    available_charge_room = float(np.clip(soc_max - soc, 0.0, 1.0))
    available_discharge_room = float(np.clip(soc - soc_min, 0.0, 1.0))
    year_sin, year_cos = _year_phase_features(step=step, total_steps=total_steps, timestamp=timestamp_obj)
    battery_action_feasible_low = float(battery_info.get("battery_action_feasible_low", 0.0))
    battery_action_feasible_high = float(battery_info.get("battery_action_feasible_high", 0.0))
    battery_charge_fraction_feasible = float(battery_info.get("battery_charge_fraction_feasible", -1.0))
    battery_discharge_fraction_feasible = float(battery_info.get("battery_discharge_fraction_feasible", -1.0))
    if battery_charge_fraction_feasible < 0.0 or battery_discharge_fraction_feasible < 0.0:
        (
            battery_action_feasible_low,
            battery_action_feasible_high,
            battery_charge_fraction_feasible,
            battery_discharge_fraction_feasible,
        ) = _soc_feasible_action_bounds(config=config, battery=battery)
    rule_based_action_hint = float(
        np.clip(
            battery_info.get(
                "rule_based_action_hint",
                _rule_based_action_hint(config=config, battery=battery, load_w=load_w, pv_w=pv_w, price=price),
            ),
            -1.0,
            1.0,
        )
    )
    current_action_norm = _normalized_battery_action(battery_power_w, config.battery_params)
    rule_based_action_gap = float(np.clip(rule_based_action_hint - current_action_norm, -2.0, 2.0))
    return np.array(
        [
            soc,
            float(getattr(battery, "temperature_c", config.battery_params.temperature_init_c)) / 100.0,
            float(load_w) / max(float(config.load_max_power), 1e-9),
            float(pv_w) / max(float(config.pv_max_power), 1e-9),
            float(price) / max(float(config.price_max), 1e-9),
            np.sin(2.0 * np.pi * hour / 24.0),
            np.cos(2.0 * np.pi * hour / 24.0),
            progress,
            float(metrics.get("min_bus_voltage_pu", 1.0)),
            float(metrics.get("max_bus_voltage_pu", 1.0)),
            float(metrics.get("max_line_loading_pct", 0.0)) / 100.0,
            float(metrics.get("max_transformer_loading_pct", 0.0)) / 100.0,
            float(metrics.get("slack_active_power_mw", 0.0)),
            battery_power_w / max(float(config.battery_params.p_discharge_max), 1e-9),
            current_a / max(float(config.battery_params.cell_capacity_ah * config.battery_params.num_cells_parallel), 1e-9),
            power_loss_w / max(float(config.battery_params.p_discharge_max), 1e-9),
            np.clip(abs(battery_power_w) / p_max_w, 0.0, 1.5),
            r_int,
            float(metrics.get("max_line_loading_pct", 0.0)) / line_limit_pct,
            soc_distance_to_target,
            available_charge_room,
            available_discharge_room,
            year_sin,
            year_cos,
            battery_action_feasible_low,
            battery_action_feasible_high,
            battery_charge_fraction_feasible,
            battery_discharge_fraction_feasible,
            rule_based_action_hint,
            rule_based_action_gap,
        ],
        dtype=np.float32,
    )
