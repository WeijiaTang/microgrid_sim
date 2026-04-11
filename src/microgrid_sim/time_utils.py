"""Shared time-grid helpers derived from ``dt_seconds``."""

from __future__ import annotations

import numpy as np
import pandas as pd

REFERENCE_START_TIMESTAMP = pd.Timestamp("2023-01-01 00:00:00")


def normalize_dt_seconds(dt_seconds: float) -> int:
    dt = float(dt_seconds)
    if dt <= 0.0:
        raise ValueError(f"dt_seconds must be positive, got {dt_seconds}")
    rounded = int(round(dt))
    if abs(dt - rounded) > 1e-9:
        raise ValueError(f"dt_seconds must be an integer number of seconds, got {dt_seconds}")
    if 86_400 % rounded != 0:
        raise ValueError(f"dt_seconds must divide 86400 exactly, got {dt_seconds}")
    return rounded


def steps_per_hour(dt_seconds: float) -> int:
    return 3_600 // normalize_dt_seconds(dt_seconds)


def steps_per_day(dt_seconds: float) -> int:
    return 86_400 // normalize_dt_seconds(dt_seconds)


def dt_hours(dt_seconds: float) -> float:
    return float(normalize_dt_seconds(dt_seconds)) / 3600.0


def simulation_steps(simulation_days: int | float, dt_seconds: float) -> int:
    steps = int(round(float(simulation_days) * steps_per_day(dt_seconds)))
    if steps <= 0:
        raise ValueError("simulation_days must produce at least one environment step")
    return steps


def hours_to_steps(hours: int | float, dt_seconds: float) -> int:
    return int(round(float(hours) * steps_per_hour(dt_seconds)))


def step_to_hour_of_day(step: int, dt_seconds: float) -> float:
    return (int(step) % steps_per_day(dt_seconds)) * dt_hours(dt_seconds)


def regular_time_index(periods: int, dt_seconds: float, start: pd.Timestamp | None = None) -> pd.DatetimeIndex:
    return pd.date_range(
        start=start or REFERENCE_START_TIMESTAMP,
        periods=max(int(periods), 0),
        freq=pd.to_timedelta(normalize_dt_seconds(dt_seconds), unit="s"),
    )


def month_index_from_timestamps(timestamps: pd.DatetimeIndex | pd.Series | np.ndarray) -> np.ndarray:
    index = pd.DatetimeIndex(pd.to_datetime(timestamps))
    return (index.year.to_numpy(dtype=int) * 100 + index.month.to_numpy(dtype=int)).astype(int)
