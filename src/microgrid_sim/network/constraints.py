"""Helpers for evaluating network constraint violations."""

from __future__ import annotations


def compute_voltage_violations(metrics: dict[str, float], v_min: float, v_max: float) -> tuple[float, float]:
    undervoltage = max(v_min - float(metrics.get("min_bus_voltage_pu", 1.0)), 0.0)
    overvoltage = max(float(metrics.get("max_bus_voltage_pu", 1.0)) - v_max, 0.0)
    return undervoltage, overvoltage


def compute_loading_violation(max_loading_pct: float, limit_pct: float) -> float:
    return max(float(max_loading_pct) - float(limit_pct), 0.0)
