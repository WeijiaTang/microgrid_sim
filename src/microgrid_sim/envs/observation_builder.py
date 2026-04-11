"""Observation construction for the network microgrid environment."""

from __future__ import annotations

import numpy as np
import pandas as pd


OBSERVATION_SIZE = 19


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
    else:
        ts = pd.Timestamp(timestamp)
        hour = float(ts.hour) + float(ts.minute) / 60.0
    progress = float(step) / max(float(total_steps - 1), 1.0)
    battery_power_w = float(battery_info.get("actual_power", battery_info.get("effective_power", 0.0)))
    current_a = float(battery_info.get("current", 0.0))
    power_loss_w = float(battery_info.get("power_loss", 0.0))
    p_max_w = max(float(battery_info.get("p_max", config.battery_params.p_discharge_max)), 1e-9)
    r_int = float(battery_info.get("r_int", 0.0))
    line_limit_pct = max(float(config.network_line_loading_limit_pct), 1e-9)
    return np.array(
        [
            float(getattr(battery, "soc", 0.5)),
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
        ],
        dtype=np.float32,
    )
