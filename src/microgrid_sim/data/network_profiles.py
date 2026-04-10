"""Network-case time-series loading and scaling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..cases import NetworkCaseConfig
from .profiles import generate_load_power, generate_pv_power, generate_tou_price

VALID_NETWORK_REGIMES = ("base", "high_load", "high_pv", "network_stress", "tight_soc")
NETWORK_REGIME_DESCRIPTIONS = {
    "base": "Nominal demand, PV, and battery initialization settings.",
    "high_load": "Amplified feeder demand with stronger evening stress.",
    "high_pv": "Amplified PV injections to test midday reverse-flow sensitivity.",
    "network_stress": "Concurrent high demand and reduced PV support to intensify voltage and congestion risk.",
    "tight_soc": "Battery starts from a narrow low-SOC band to constrain dispatch flexibility.",
}


@dataclass(frozen=True)
class NetworkProfiles:
    load_w: np.ndarray
    pv_w: np.ndarray
    price: np.ndarray


def _trim_or_tile(values: np.ndarray, total_hours: int) -> np.ndarray:
    if len(values) == total_hours:
        return values.astype(float, copy=False)
    if len(values) > total_hours:
        return values[:total_hours].astype(float, copy=False)
    repeats = int(np.ceil(total_hours / max(len(values), 1)))
    return np.tile(values, repeats)[:total_hours].astype(float, copy=False)


def normalize_network_regime(regime: str) -> str:
    normalized = str(regime).strip().lower() or "base"
    if normalized not in VALID_NETWORK_REGIMES:
        raise ValueError(f"Unsupported network regime '{regime}'. Expected one of {VALID_NETWORK_REGIMES}.")
    return normalized


def describe_network_regime(regime: str) -> str:
    return NETWORK_REGIME_DESCRIPTIONS[normalize_network_regime(regime)]


def _hour_mask(hours: int, start_hour: int, end_hour: int) -> np.ndarray:
    hour_of_day = np.arange(hours, dtype=int) % 24
    return (hour_of_day >= int(start_hour)) & (hour_of_day < int(end_hour))


def _apply_regime_scaling(config: NetworkCaseConfig, load_w: np.ndarray, pv_w: np.ndarray, price: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    regime = normalize_network_regime(getattr(config, "regime", "base"))
    case_key = str(getattr(config, "case_key", "")).strip().lower()
    load_w = np.asarray(load_w, dtype=float).copy()
    pv_w = np.asarray(pv_w, dtype=float).copy()
    price = np.asarray(price, dtype=float).copy()
    hours = len(load_w)
    evening_mask = _hour_mask(hours, 17, 22)
    midday_mask = _hour_mask(hours, 10, 15)
    shoulder_mask = _hour_mask(hours, 14, 18)
    late_evening_mask = _hour_mask(hours, 19, 24)

    if regime == "high_load":
        load_w *= 1.18
        load_w[evening_mask] *= 1.10
    elif regime == "high_pv":
        pv_w *= 1.35
        pv_w[midday_mask] *= 1.12
    elif regime == "network_stress":
        load_w *= 1.22
        load_w[evening_mask] *= 1.15
        pv_w *= 0.90
        pv_w[midday_mask] *= 0.95
        if case_key == "ieee33_modified_network":
            # Make stressed distribution-feeder episodes more reproducible by
            # extending high-demand pressure into the afternoon/evening window
            # while reducing midday PV relief, but keep enough flexibility so
            # fidelity-driven policy differences can still emerge.
            load_w *= 1.04
            load_w[shoulder_mask] *= 1.05
            load_w[late_evening_mask] *= 1.07
            pv_w *= 0.95
            pv_w[midday_mask] *= 0.94
        center = 0.5 * (float(np.min(price)) + float(np.max(price)))
        price = np.clip(center + (price - center) * 1.20, 0.0, None)
        if case_key == "ieee33_modified_network":
            price[shoulder_mask] *= 1.05
            price[late_evening_mask] *= 1.10
    elif regime == "tight_soc":
        load_w *= 1.05
        load_w[evening_mask] *= 1.05
    return load_w, pv_w, price


def load_network_profiles(config: NetworkCaseConfig, total_hours: int | None = None) -> NetworkProfiles:
    hours = int(total_hours or (365 * 24))
    load_w = _trim_or_tile(generate_load_power(hours, peak_load=config.load_max_power, seed=config.seed + 1), hours)
    pv_w = _trim_or_tile(generate_pv_power(hours, pv_rated_power=config.pv_max_power, seed=config.seed), hours)
    price = _trim_or_tile(generate_tou_price(hours), hours)
    if abs(float(config.tou_price_spread_multiplier) - 1.0) > 1e-9:
        center = 0.5 * (float(np.min(price)) + float(np.max(price)))
        price = np.clip(center + (price - center) * float(config.tou_price_spread_multiplier), 0.0, None)
    load_w, pv_w, price = _apply_regime_scaling(config, load_w=load_w, pv_w=pv_w, price=price)
    return NetworkProfiles(load_w=load_w, pv_w=pv_w, price=price)
