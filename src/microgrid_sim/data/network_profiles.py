"""Network-case time-series loading and scaling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..cases import NetworkCaseConfig
from ..io.reader import read_dataset_bundle, resolve_bundle_files
from ..paths import NETWORK_DATA_ROOT
from ..time_utils import REFERENCE_START_TIMESTAMP, regular_time_index, steps_per_day
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
    timestamps: pd.DatetimeIndex


def _trim_or_tile(values: np.ndarray, total_steps: int) -> np.ndarray:
    if len(values) == total_steps:
        return values.astype(float, copy=False)
    if len(values) > total_steps:
        return values[:total_steps].astype(float, copy=False)
    repeats = int(np.ceil(total_steps / max(len(values), 1)))
    return np.tile(values, repeats)[:total_steps].astype(float, copy=False)


def _trim_or_build_timestamps(
    timestamps: pd.DatetimeIndex | None,
    total_steps: int,
    dt_seconds: float,
    start: pd.Timestamp | None = None,
) -> pd.DatetimeIndex:
    if timestamps is None or len(timestamps) <= 0:
        return regular_time_index(periods=total_steps, dt_seconds=dt_seconds, start=start or REFERENCE_START_TIMESTAMP)
    if len(timestamps) >= total_steps:
        return pd.DatetimeIndex(timestamps[:total_steps])
    return regular_time_index(periods=total_steps, dt_seconds=dt_seconds, start=pd.Timestamp(timestamps[0]))


def _network_case_dirname(case_key: str) -> str:
    normalized = str(case_key).strip().lower()
    mapping = {
        "cigre_eu_lv_network": "cigre_eu_lv",
        "ieee33_network": "ieee33",
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported network case_key '{case_key}' for network data resolution")
    return mapping[normalized]


def _resolve_network_case_data_dir(config: NetworkCaseConfig) -> Path | None:
    case_dirname = _network_case_dirname(getattr(config, "case_key", ""))
    candidates: list[Path] = []
    if getattr(config, "data_dir", None):
        root = Path(str(config.data_dir)).resolve()
        candidates.extend(
            [
                root,
                root / case_dirname,
                root / "processed" / "network_15min" / case_dirname,
                root / "network" / case_dirname,
            ]
        )
    candidates.append((NETWORK_DATA_ROOT / case_dirname).resolve())
    for candidate in candidates:
        try:
            bundle = read_dataset_bundle(candidate, total_hours=1, required_roles=("load", "pv", "price"))
        except (FileNotFoundError, NotADirectoryError, PermissionError, ValueError):
            continue
        if {"load", "pv", "price"}.issubset(bundle.keys()):
            return candidate
    return None


def normalize_network_regime(regime: str) -> str:
    normalized = str(regime).strip().lower() or "base"
    if normalized not in VALID_NETWORK_REGIMES:
        raise ValueError(f"Unsupported network regime '{regime}'. Expected one of {VALID_NETWORK_REGIMES}.")
    return normalized


def describe_network_regime(regime: str) -> str:
    return NETWORK_REGIME_DESCRIPTIONS[normalize_network_regime(regime)]


def _read_network_case_role(path: Path) -> tuple[pd.DatetimeIndex | None, np.ndarray]:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Network dataset file is empty: {path}")
    timestamps: pd.DatetimeIndex | None = None
    if "datetime" in frame.columns:
        timestamps = pd.DatetimeIndex(pd.to_datetime(frame["datetime"]))
    numeric_columns = [column for column in frame.columns if column != "datetime"] or list(frame.columns)
    for column in numeric_columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().any():
            return timestamps, values.to_numpy(dtype=float)
    raise ValueError(f"No numeric values found in network dataset file: {path}")


def _read_network_case_bundle(case_data_dir: Path, dt_seconds: float) -> NetworkProfiles:
    files = resolve_bundle_files(case_data_dir, required_roles=("load", "pv", "price"))
    if not {"load", "pv", "price"}.issubset(files.keys()):
        raise FileNotFoundError(f"Missing load/pv/price files in {case_data_dir}")
    load_ts, load_w = _read_network_case_role(files["load"])
    pv_ts, pv_w = _read_network_case_role(files["pv"])
    price_ts, price = _read_network_case_role(files["price"])
    timestamps: pd.DatetimeIndex | None = None
    for candidate in (load_ts, pv_ts, price_ts):
        if candidate is not None:
            timestamps = candidate
            break
    max_len = max(len(load_w), len(pv_w), len(price))
    return NetworkProfiles(
        load_w=_trim_or_tile(np.asarray(load_w, dtype=float), max_len),
        pv_w=_trim_or_tile(np.asarray(pv_w, dtype=float), max_len),
        price=_trim_or_tile(np.asarray(price, dtype=float), max_len),
        timestamps=_trim_or_build_timestamps(timestamps, max_len, dt_seconds=dt_seconds),
    )


def _is_canonical_network_bundle(case_data_dir: Path) -> bool:
    try:
        return case_data_dir.resolve().is_relative_to(NETWORK_DATA_ROOT.resolve())
    except AttributeError:
        resolved_case_dir = str(case_data_dir.resolve())
        resolved_root = str(NETWORK_DATA_ROOT.resolve())
        return resolved_case_dir.startswith(resolved_root)


def _rescale_loaded_network_profiles(
    config: NetworkCaseConfig,
    profiles: NetworkProfiles,
    *,
    case_data_dir: Path | None,
) -> NetworkProfiles:
    if case_data_dir is None or not _is_canonical_network_bundle(case_data_dir):
        return profiles

    load_w = np.asarray(profiles.load_w, dtype=float).copy()
    pv_w = np.asarray(profiles.pv_w, dtype=float).copy()
    load_peak = float(np.max(load_w)) if load_w.size else 0.0
    pv_peak = float(np.max(pv_w)) if pv_w.size else 0.0
    if load_peak > 0.0:
        load_w *= float(config.load_max_power) / load_peak
    if pv_peak > 0.0:
        pv_w *= float(config.pv_max_power) / pv_peak
    return NetworkProfiles(
        load_w=load_w,
        pv_w=pv_w,
        price=np.asarray(profiles.price, dtype=float).copy(),
        timestamps=pd.DatetimeIndex(profiles.timestamps),
    )


def _hour_mask(timestamps: pd.DatetimeIndex, start_hour: int, end_hour: int) -> np.ndarray:
    hour_of_day = timestamps.hour.to_numpy(dtype=float) + timestamps.minute.to_numpy(dtype=float) / 60.0
    return (hour_of_day >= float(start_hour)) & (hour_of_day < float(end_hour))


def _apply_regime_scaling(
    config: NetworkCaseConfig,
    load_w: np.ndarray,
    pv_w: np.ndarray,
    price: np.ndarray,
    timestamps: pd.DatetimeIndex,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    regime = normalize_network_regime(getattr(config, "regime", "base"))
    case_key = str(getattr(config, "case_key", "")).strip().lower()
    load_w = np.asarray(load_w, dtype=float).copy()
    pv_w = np.asarray(pv_w, dtype=float).copy()
    price = np.asarray(price, dtype=float).copy()
    evening_mask = _hour_mask(timestamps, 17, 22)
    midday_mask = _hour_mask(timestamps, 10, 15)
    shoulder_mask = _hour_mask(timestamps, 14, 18)
    late_evening_mask = _hour_mask(timestamps, 19, 24)

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
        if case_key == "ieee33_network":
            load_w *= 1.04
            load_w[shoulder_mask] *= 1.05
            load_w[late_evening_mask] *= 1.07
            pv_w *= 0.95
            pv_w[midday_mask] *= 0.94
        center = 0.5 * (float(np.min(price)) + float(np.max(price)))
        price = np.clip(center + (price - center) * 1.20, 0.0, None)
        if case_key == "ieee33_network":
            price[shoulder_mask] *= 1.05
            price[late_evening_mask] *= 1.10
    elif regime == "tight_soc":
        load_w *= 1.05
        load_w[evening_mask] *= 1.05
    return load_w, pv_w, price


def load_network_profiles(
    config: NetworkCaseConfig,
    total_steps: int | None = None,
    total_hours: int | None = None,
) -> NetworkProfiles:
    requested_steps = int(total_steps if total_steps is not None else total_hours if total_hours is not None else 0)
    case_data_dir = _resolve_network_case_data_dir(config)
    if case_data_dir is not None:
        full_profiles = _read_network_case_bundle(case_data_dir, dt_seconds=config.dt_seconds)
        full_profiles = _rescale_loaded_network_profiles(config, full_profiles, case_data_dir=case_data_dir)
    else:
        default_steps = requested_steps if requested_steps > 0 else 2 * 365 * steps_per_day(config.dt_seconds)
        full_profiles = NetworkProfiles(
            load_w=_trim_or_tile(
                generate_load_power(default_steps, peak_load=config.load_max_power, seed=config.seed + 1, dt_seconds=config.dt_seconds),
                default_steps,
            ),
            pv_w=_trim_or_tile(
                generate_pv_power(default_steps, pv_rated_power=config.pv_max_power, seed=config.seed, dt_seconds=config.dt_seconds),
                default_steps,
            ),
            price=_trim_or_tile(generate_tou_price(default_steps, dt_seconds=config.dt_seconds), default_steps),
            timestamps=regular_time_index(periods=default_steps, dt_seconds=config.dt_seconds, start=REFERENCE_START_TIMESTAMP),
        )

    if requested_steps <= 0:
        requested_steps = len(full_profiles.load_w)
    load_w = _trim_or_tile(np.asarray(full_profiles.load_w, dtype=float), requested_steps)
    pv_w = _trim_or_tile(np.asarray(full_profiles.pv_w, dtype=float), requested_steps)
    price = _trim_or_tile(np.asarray(full_profiles.price, dtype=float), requested_steps)
    timestamps = _trim_or_build_timestamps(full_profiles.timestamps, requested_steps, dt_seconds=config.dt_seconds)
    if abs(float(config.tou_price_spread_multiplier) - 1.0) > 1e-9:
        center = 0.5 * (float(np.min(price)) + float(np.max(price)))
        price = np.clip(center + (price - center) * float(config.tou_price_spread_multiplier), 0.0, None)
    load_w, pv_w, price = _apply_regime_scaling(config, load_w=load_w, pv_w=pv_w, price=price, timestamps=timestamps)
    return NetworkProfiles(load_w=load_w, pv_w=pv_w, price=price, timestamps=timestamps)
