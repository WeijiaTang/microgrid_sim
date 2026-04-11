"""Exogenous profile loading for MG-RES and MG-CIGRE."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..cases import MicrogridConfig
from ..io.reader import read_case_dataset, read_numeric_series
from ..paths import LEGACY_AGGREGATED_DATA_ROOT, LEGACY_YEARLY_DATA_ROOT, PROJECT_ROOT, resolve_data_dir
from ..time_utils import dt_hours, steps_per_day


def _trim_or_tile(values: np.ndarray, total_hours: Optional[int]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if total_hours is None:
        return values.astype(float, copy=False)
    if len(values) == total_hours:
        return values.astype(float, copy=False)
    if len(values) > total_hours:
        return values[:total_hours].astype(float, copy=False)
    repeats = int(np.ceil(total_hours / max(len(values), 1)))
    return np.tile(values, repeats)[:total_hours].astype(float, copy=False)


def validate_case_data(config: MicrogridConfig, data: dict) -> None:
    """Validate loaded data against basic units and case expectations."""

    pv = np.asarray(data["pv"], dtype=float)
    load = np.asarray(data["load"], dtype=float)
    price = np.asarray(data["price"], dtype=float)
    other = np.asarray(data.get("other", np.zeros_like(load)), dtype=float)
    wind = np.asarray(data.get("wind", np.zeros_like(load)), dtype=float)

    expected_len = int(data["hours"])
    for name, values in (("pv", pv), ("load", load), ("price", price), ("other", other), ("wind", wind)):
        if len(values) != expected_len:
            raise ValueError(f"{config.case_name}: '{name}' length {len(values)} != expected {expected_len}")

    if np.any(pv < -1e-9):
        raise ValueError(f"{config.case_name}: PV series must be non-negative in Watts")
    if np.any(load < -1e-9):
        raise ValueError(f"{config.case_name}: Load series must be non-negative in Watts")
    if np.any(price < -1e-9):
        raise ValueError(f"{config.case_name}: Price series must be non-negative in $/kWh")
    if np.any(wind < -1e-9):
        raise ValueError(f"{config.case_name}: Wind series must be non-negative in Watts")

    if config.strict_reproduction:
        peak_price = float(np.max(price))
        valley_price = float(np.min(price))
        if abs(peak_price - config.price_max) > 0.03:
            raise ValueError(f"{config.case_name}: peak price {peak_price:.5f} deviates from expected {config.price_max:.5f}")
        if valley_price > 0.45 or valley_price < 0.30:
            raise ValueError(f"{config.case_name}: valley price {valley_price:.5f} is inconsistent with TOU range")
        if config.case_key == "mg_cigre" and "other" not in data and "net" not in data:
            raise ValueError("MG-CIGRE strict reproduction requires 'other' or 'net' exogenous data")


def generate_pv_power(
    total_hours: int,
    pv_rated_power: float = 100_000.0,
    seed: Optional[int] = None,
    dt_seconds: float = 3600.0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    output = np.zeros(total_hours, dtype=float)
    delta_hours = dt_hours(dt_seconds)
    daily_steps = steps_per_day(dt_seconds)
    for idx in range(total_hours):
        hour = (idx % daily_steps) * delta_hours
        day = idx // daily_steps
        if 6 <= hour < 18:
            daylight = np.sin(np.pi * (hour - 6) / 12.0) ** 2
            seasonal = 0.8 + 0.2 * np.cos(2 * np.pi * ((day % 365) - 172) / 365.0)
            weather = np.clip(0.85 + 0.15 * rng.normal(), 0.35, 1.0)
            output[idx] = pv_rated_power * daylight * seasonal * weather
    return np.maximum(output, 0.0)


def generate_load_power(
    total_hours: int,
    peak_load: float = 67_900.0,
    seed: Optional[int] = None,
    dt_seconds: float = 3600.0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    output = np.zeros(total_hours, dtype=float)
    base = 0.33 * peak_load
    delta_hours = dt_hours(dt_seconds)
    daily_steps = steps_per_day(dt_seconds)
    for idx in range(total_hours):
        hour = (idx % daily_steps) * delta_hours
        day = idx // daily_steps
        morning = np.exp(-((hour - 8.5) ** 2) / (2 * 1.8**2))
        evening = np.exp(-((hour - 19.5) ** 2) / (2 * 2.4**2))
        seasonal = 1.0 + 0.12 * np.cos(2 * np.pi * ((day % 365) - 200) / 365.0)
        weekend = 0.92 if (day % 7) >= 5 else 1.0
        noise = np.clip(1.0 + 0.06 * rng.normal(), 0.85, 1.15)
        output[idx] = (base + 0.32 * peak_load * morning + 0.45 * peak_load * evening) * seasonal * weekend * noise
    return np.clip(output, 0.25 * peak_load, None)


def generate_tou_price(total_hours: int, dt_seconds: float = 3600.0) -> np.ndarray:
    daily = np.array([
        0.39073, 0.39073, 0.39073, 0.39073, 0.39073, 0.39073,
        0.39073, 0.39073, 0.39073, 0.39073, 0.39073, 0.39073,
        0.45100, 0.45100, 0.45100, 0.45100, 0.51373, 0.51373,
        0.51373, 0.51373, 0.45100, 0.45100, 0.39073, 0.39073,
    ])
    repeats_per_hour = max(int(round(3600.0 / float(dt_seconds))), 1)
    return _trim_or_tile(np.repeat(daily, repeats_per_hour), total_hours)


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _candidate_data_roots(data_dir: str | Path | None) -> list[Path]:
    roots: list[Path] = []
    if data_dir is not None:
        roots.append(Path(data_dir).resolve())
    roots.append((PROJECT_ROOT / "data").resolve())

    unique_roots: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key not in seen and root.exists():
            unique_roots.append(root)
            seen.add(key)
    return unique_roots


def _resolve_legacy_file(roots: list[Path], relative_path: str) -> Path | None:
    candidates: list[Path] = []
    for root in roots:
        candidates.append(root / relative_path)
        candidates.append(root / "legacy" / relative_path)
    return _first_existing(candidates)


def _build_wind_from_weather(weather_path: Path, total_hours: int) -> np.ndarray:
    weather = pd.read_csv(weather_path)
    if "wind_speed" not in weather.columns:
        return np.zeros(total_hours, dtype=float)
    wind_speed_ref = _trim_or_tile(weather["wind_speed"].to_numpy(dtype=float), total_hours)
    ref_height_m = 10.0
    hub_height_m = 80.0
    shear_exponent = 0.14
    site_exposure_multiplier = 2.0
    wind_speed_hub = wind_speed_ref * (hub_height_m / ref_height_m) ** shear_exponent * site_exposure_multiplier
    cut_in = 2.5
    rated = 11.0
    cut_out = 25.0
    power_pu = np.zeros_like(wind_speed_hub)
    ramp_mask = (wind_speed_hub >= cut_in) & (wind_speed_hub < rated)
    rated_mask = (wind_speed_hub >= rated) & (wind_speed_hub < cut_out)
    power_pu[ramp_mask] = (wind_speed_hub[ramp_mask] ** 3 - cut_in**3) / (rated**3 - cut_in**3)
    power_pu[rated_mask] = 1.0
    return 10_000.0 * np.clip(power_pu, 0.0, 1.0)


def _load_builtin_res_year_data(config: MicrogridConfig, year: int, total_hours: Optional[int], data_dir: str | Path | None) -> dict:
    roots = _candidate_data_roots(data_dir)
    load_path = _resolve_legacy_file(roots, f"yearly/load/load_power_{year}.csv")
    pv_path = _resolve_legacy_file(roots, f"yearly/pv/pv_power_yearly_los_angeles_{year}.csv")
    price_path = _resolve_legacy_file(roots, "yearly/tariff/price_profile.csv")

    if load_path is None or pv_path is None or price_path is None:
        raise FileNotFoundError(f"Missing built-in yearly residential files for {year}")

    load_full_raw = np.asarray(read_numeric_series(load_path), dtype=float)
    pv_full_raw = np.asarray(read_numeric_series(pv_path), dtype=float)
    load_scale = float(config.load_max_power / max(np.max(load_full_raw), 1e-9))
    pv_scale = float(config.pv_max_power / max(np.max(pv_full_raw), 1e-9))
    load_full = load_full_raw * load_scale
    pv_full = pv_full_raw * pv_scale
    hours_full = min(365 * 24, len(load_full), len(pv_full)) if total_hours is None else total_hours
    load = _trim_or_tile(load_full, hours_full)
    pv = _trim_or_tile(pv_full, hours_full)
    price = _trim_or_tile(read_numeric_series(price_path), hours_full)

    case_dir = (LEGACY_AGGREGATED_DATA_ROOT / "mg_res").resolve()
    payload = {
        "pv": pv,
        "load": load,
        "price": price,
        "other": np.zeros(hours_full, dtype=float),
        "wind": np.zeros(hours_full, dtype=float),
        "hours": int(hours_full),
        "source": f"builtin_res_year:{year}",
        "data_dir": str(data_dir) if data_dir is not None else str(PROJECT_ROOT / "data"),
        "case_dir": str(case_dir),
    }
    validate_case_data(config, payload)
    return payload


def _load_builtin_cigre_year_data(config: MicrogridConfig, year: int, total_hours: Optional[int], data_dir: str | Path | None) -> dict:
    roots = _candidate_data_roots(data_dir)
    load_path = _resolve_legacy_file(roots, f"yearly/load/load_power_{year}.csv")
    pv_path = _resolve_legacy_file(roots, f"yearly/pv/pv_power_yearly_los_angeles_{year}.csv")
    weather_path = _resolve_legacy_file(roots, f"yearly/weather/los_angeles_pv_data_{year}.csv")
    price_path = _resolve_legacy_file(roots, "yearly/tariff/price_profile.csv")

    if load_path is None or pv_path is None or price_path is None:
        raise FileNotFoundError(f"Missing built-in yearly CIGRE files for {year}")

    load_full_raw = np.asarray(read_numeric_series(load_path), dtype=float)
    pv_full_raw = np.asarray(read_numeric_series(pv_path), dtype=float)
    load_scale = float(config.load_max_power / max(np.max(load_full_raw), 1e-9))
    pv_scale = float(config.pv_max_power / max(np.max(pv_full_raw), 1e-9))
    load_full = load_full_raw * load_scale
    pv_full = pv_full_raw * pv_scale
    hours_full = min(365 * 24, len(load_full), len(pv_full)) if total_hours is None else total_hours
    load = _trim_or_tile(load_full, hours_full)
    pv = _trim_or_tile(pv_full, hours_full)
    price = _trim_or_tile(read_numeric_series(price_path), hours_full)
    wind = _build_wind_from_weather(weather_path, hours_full) if weather_path is not None else np.zeros(hours_full, dtype=float)

    case_dir = (LEGACY_AGGREGATED_DATA_ROOT / "mg_cigre").resolve()
    ref_load_path = case_dir / "load.csv"
    ref_other_path = case_dir / "other.csv"
    ref_net_path = case_dir / "net.csv"

    ref_load = _trim_or_tile(read_numeric_series(ref_load_path), hours_full) if ref_load_path.exists() else load
    year_other_path = case_dir / f"other_{year}.csv"
    year_net_path = case_dir / f"net_{year}.csv"
    if year_other_path.exists():
        other = _trim_or_tile(read_numeric_series(year_other_path), hours_full) * load_scale
    elif year_net_path.exists():
        net = _trim_or_tile(read_numeric_series(year_net_path), hours_full) * load_scale
        other = load - pv - net
    elif ref_other_path.exists():
        other = _trim_or_tile(read_numeric_series(ref_other_path), hours_full) * load_scale
    elif ref_net_path.exists():
        net = _trim_or_tile(read_numeric_series(ref_net_path), hours_full) * load_scale
        other = load - pv - net
    else:
        other = np.zeros(hours_full, dtype=float)

    payload = {
        "pv": pv,
        "load": load,
        "price": price,
        "other": other,
        "wind": wind,
        "hours": int(hours_full),
        "source": f"builtin_cigre_year:{year}",
        "data_dir": str(data_dir) if data_dir is not None else str(PROJECT_ROOT / 'data'),
        "case_dir": str(case_dir),
    }
    validate_case_data(config, payload)
    return payload


def load_case_data(config: MicrogridConfig, total_hours: Optional[int], data_dir: str | Path | None = None) -> dict:
    resolved_data_dir = Path(data_dir) if data_dir else resolve_data_dir()
    if config.case_key == "mg_cigre" and config.data_year is not None:
        return _load_builtin_cigre_year_data(config=config, year=int(config.data_year), total_hours=total_hours, data_dir=resolved_data_dir)
    if config.case_key == "mg_res" and config.data_year is not None:
        return _load_builtin_res_year_data(config=config, year=int(config.data_year), total_hours=total_hours, data_dir=resolved_data_dir)

    if resolved_data_dir is None and (config.use_real_data or config.strict_reproduction):
        raise FileNotFoundError(
            f"No external dataset directory found for {config.case_name}. Set MICROGRID_SIM_DATA_DIR or pass data_dir."
        )

    if resolved_data_dir is None:
        hours = 365 * 24 if total_hours is None else total_hours
        return {
            "pv": generate_pv_power(hours, config.pv_max_power, seed=config.seed, dt_seconds=config.dt_seconds),
            "load": generate_load_power(hours, config.load_max_power, seed=config.seed + 1, dt_seconds=config.dt_seconds),
            "price": generate_tou_price(hours, dt_seconds=config.dt_seconds),
            "other": np.zeros(hours, dtype=float),
            "wind": np.zeros(hours, dtype=float),
            "hours": hours,
            "source": "synthetic_fallback",
        }

    try:
        bundle = read_case_dataset(resolved_data_dir, case_key=config.case_key, total_hours=total_hours)
    except FileNotFoundError:
        if config.use_real_data or config.strict_reproduction:
            raise
        hours = 365 * 24 if total_hours is None else total_hours
        return {
            "pv": generate_pv_power(hours, config.pv_max_power, seed=config.seed, dt_seconds=config.dt_seconds),
            "load": generate_load_power(hours, config.load_max_power, seed=config.seed + 1, dt_seconds=config.dt_seconds),
            "price": generate_tou_price(hours, dt_seconds=config.dt_seconds),
            "other": np.zeros(hours, dtype=float),
            "wind": np.zeros(hours, dtype=float),
            "hours": hours,
            "source": "synthetic_fallback",
        }

    pv = np.asarray(bundle["pv"], dtype=float)
    load = np.asarray(bundle["load"], dtype=float)
    price = np.asarray(bundle["price"], dtype=float)
    hours = len(load)
    other = np.zeros(hours, dtype=float)
    if "other" in bundle:
        other = np.asarray(bundle["other"], dtype=float)
    elif "net" in bundle:
        other = load - pv - np.asarray(bundle["net"], dtype=float)
    wind = np.asarray(bundle.get("wind", np.zeros(hours, dtype=float)), dtype=float)

    payload = {
        "pv": pv,
        "load": load,
        "price": price,
        "other": other,
        "wind": wind,
        "hours": hours,
        "source": str(bundle.get("source", resolved_data_dir)),
        "data_dir": str(bundle.get("data_dir", resolved_data_dir)),
        "case_dir": str(bundle.get("case_dir", resolved_data_dir)),
    }
    validate_case_data(config, payload)
    return payload


def load_simulation_data(
    data_dir: str | Path | None = None,
    total_hours: int = 720,
    config: MicrogridConfig | None = None,
    use_yearly_pv: bool = True,
) -> dict:
    del use_yearly_pv
    return load_case_data(config or MicrogridConfig(), total_hours=total_hours, data_dir=data_dir)
