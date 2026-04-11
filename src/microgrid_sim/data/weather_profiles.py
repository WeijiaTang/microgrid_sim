"""Weather-backed reference profile preparation utilities."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..cases import CIGREEuropeanLVConfig, IEEE33Config
from ..paths import LEGACY_YEARLY_DATA_ROOT, NETWORK_DATA_ROOT, PROCESSED_DATA_ROOT, RAW_DATA_ROOT

REFERENCE_LOCATION = "Los Angeles, California, USA"
REFERENCE_SLUG = "los_angeles"
REFERENCE_START = pd.Timestamp("2023-01-01 00:00:00")
REFERENCE_END_HOURLY = pd.Timestamp("2024-12-31 23:00:00")
REFERENCE_END_15MIN = pd.Timestamp("2024-12-31 23:45:00")
REFERENCE_FREQ_15MIN = "15min"
REFERENCE_YEARS = (2023, 2024)
REFERENCE_LOAD_PEAK_W = 1_000_000.0

LINEAR_INTERPOLATION_COLUMNS = (
    "temperature",
    "dewpoint_temperature",
    "relative_humidity",
    "surface_pressure",
    "direct_normal_solar_radiation",
    "surface_diffuse_solar_radiation",
    "surface_solar_radiation",
    "surface_thermal_radiation",
    "wind_speed",
    "total_cloud_cover",
    "pv_power_watts",
)
STEP_DISTRIBUTION_COLUMNS = ("total_precipitation", "snow_depth")
STATIC_COLUMNS = ("coordinates", "model", "model elevation", "utc_offset")


@dataclass(frozen=True)
class PreparedReferenceOutputs:
    raw_dir: Path
    processed_dir: Path
    merged_hourly_weather_csv: Path
    weather_15min_csv: Path
    pv_15min_csv: Path
    load_reference_15min_csv: Path
    price_15min_csv: Path
    cigre_dir: Path
    ieee33_dir: Path
    canonical_cigre_dir: Path
    canonical_ieee33_dir: Path
    metadata_json: Path


def raw_weather_dir() -> Path:
    return RAW_DATA_ROOT / "weather" / "oikolab" / REFERENCE_SLUG


def processed_reference_dir() -> Path:
    return PROCESSED_DATA_ROOT / "reference_15min" / f"{REFERENCE_SLUG}_2023_2024"


def processed_network_15min_dir(case_dirname: str) -> Path:
    return PROCESSED_DATA_ROOT / "network_15min" / case_dirname


def canonical_network_dir(case_dirname: str) -> Path:
    return NETWORK_DATA_ROOT / case_dirname


def _legacy_tariff_hourly_template() -> np.ndarray:
    tariff_path = LEGACY_YEARLY_DATA_ROOT / "tariff" / "price_profile.csv"
    if not tariff_path.exists():
        raise FileNotFoundError(f"Missing legacy tariff source: {tariff_path}")
    values = pd.read_csv(tariff_path, header=None).iloc[:, 0].astype(float).to_numpy(dtype=float)
    if values.size < 24 or values.size % 24 != 0:
        raise ValueError(f"Legacy tariff file must contain whole-day hourly values: {tariff_path}")
    daily = values.reshape(-1, 24)
    template = daily[0].copy()
    if not np.allclose(daily, template[None, :], atol=1e-9, rtol=0.0):
        raise ValueError(f"Legacy tariff file is not day-stationary and cannot be collapsed safely: {tariff_path}")
    return template


def _legacy_weather_path(year: int) -> Path:
    return LEGACY_YEARLY_DATA_ROOT / "weather" / f"{REFERENCE_SLUG}_pv_data_{year}.csv"


def _raw_weather_filename(year: int) -> str:
    return f"{REFERENCE_SLUG}_hourly_{year}.csv"


def ensure_raw_weather_from_legacy() -> list[Path]:
    """Copy checked-in yearly hourly weather files into the canonical raw path."""

    destination_dir = raw_weather_dir()
    destination_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for year in REFERENCE_YEARS:
        source = _legacy_weather_path(year)
        if not source.exists():
            raise FileNotFoundError(f"Missing legacy weather source: {source}")
        target = destination_dir / _raw_weather_filename(year)
        shutil.copy2(source, target)
        copied.append(target)
    return copied


def load_raw_hourly_weather() -> pd.DataFrame:
    """Load the two-year hourly Los Angeles weather reference set."""

    frames: list[pd.DataFrame] = []
    for year in REFERENCE_YEARS:
        path = raw_weather_dir() / _raw_weather_filename(year)
        if not path.exists():
            raise FileNotFoundError(f"Missing raw weather file: {path}")
        frame = pd.read_csv(path)
        frame["datetime"] = pd.to_datetime(frame["datetime"])
        frames.append(frame)
    weather = pd.concat(frames, ignore_index=True)
    weather = weather.sort_values("datetime").drop_duplicates(subset=["datetime"]).reset_index(drop=True)
    if weather["datetime"].iloc[0] != REFERENCE_START:
        raise ValueError(f"Unexpected weather start timestamp: {weather['datetime'].iloc[0]}")
    if weather["datetime"].iloc[-1] != REFERENCE_END_HOURLY:
        raise ValueError(f"Unexpected weather end timestamp: {weather['datetime'].iloc[-1]}")
    return weather


def _interpolate_wind_direction(direction: pd.Series, target_index: pd.DatetimeIndex) -> pd.Series:
    radians = np.deg2rad(pd.to_numeric(direction, errors="coerce").ffill().bfill().to_numpy(dtype=float))
    unwrapped = np.unwrap(radians)
    base = pd.Series(unwrapped, index=direction.index, dtype=float)
    interpolated = base.reindex(target_index).interpolate(method="time", limit_direction="both")
    return pd.Series(np.mod(np.rad2deg(interpolated.to_numpy(dtype=float)), 360.0), index=target_index, dtype=float)


def resample_weather_to_15min(hourly_weather: pd.DataFrame) -> pd.DataFrame:
    """Upsample hourly weather to 15-minute resolution with variable-aware rules."""

    weather = hourly_weather.copy()
    weather["datetime"] = pd.to_datetime(weather["datetime"])
    weather = weather.set_index("datetime").sort_index()
    target_index = pd.date_range(start=REFERENCE_START, end=REFERENCE_END_15MIN, freq=REFERENCE_FREQ_15MIN)
    frame = pd.DataFrame(index=target_index)

    for column in STATIC_COLUMNS:
        if column in weather.columns:
            frame[column] = weather[column].reindex(target_index).ffill().bfill()

    for column in LINEAR_INTERPOLATION_COLUMNS:
        if column in weather.columns:
            numeric = pd.to_numeric(weather[column], errors="coerce")
            frame[column] = numeric.reindex(target_index).interpolate(method="time", limit_direction="both")

    if "wind_direction" in weather.columns:
        frame["wind_direction"] = _interpolate_wind_direction(weather["wind_direction"], target_index)

    for column in STEP_DISTRIBUTION_COLUMNS:
        if column in weather.columns:
            numeric = pd.to_numeric(weather[column], errors="coerce").fillna(0.0)
            frame[column] = numeric.reindex(target_index, method="ffill").fillna(0.0) / 4.0

    non_negative_columns = {
        "relative_humidity",
        "direct_normal_solar_radiation",
        "surface_diffuse_solar_radiation",
        "surface_solar_radiation",
        "surface_thermal_radiation",
        "wind_speed",
        "total_cloud_cover",
        "total_precipitation",
        "snow_depth",
        "pv_power_watts",
    }
    for column in non_negative_columns.intersection(frame.columns):
        frame[column] = np.clip(pd.to_numeric(frame[column], errors="coerce").fillna(0.0), 0.0, None)

    if "relative_humidity" in frame.columns:
        frame["relative_humidity"] = np.clip(frame["relative_humidity"], 0.0, 1.0)
    if "total_cloud_cover" in frame.columns:
        frame["total_cloud_cover"] = np.clip(frame["total_cloud_cover"], 0.0, 1.0)

    frame = frame.reset_index(names="datetime")
    return frame


def synthesize_reference_load_15min(weather_15min: pd.DataFrame, peak_load_w: float = REFERENCE_LOAD_PEAK_W) -> pd.Series:
    """Create a deterministic weather-aware feeder load reference profile."""

    timestamps = pd.to_datetime(weather_15min["datetime"])
    hour = timestamps.dt.hour.to_numpy(dtype=float) + timestamps.dt.minute.to_numpy(dtype=float) / 60.0
    weekday = timestamps.dt.dayofweek.to_numpy(dtype=int)
    dayofyear = timestamps.dt.dayofyear.to_numpy(dtype=float)
    month = timestamps.dt.month.to_numpy(dtype=int)
    temperature = pd.to_numeric(weather_15min["temperature"], errors="coerce").ffill().bfill().to_numpy(dtype=float)
    cloud_cover = pd.to_numeric(weather_15min.get("total_cloud_cover", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    pv_reference = pd.to_numeric(weather_15min.get("pv_power_watts", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    pv_relief = pv_reference / max(float(np.max(pv_reference)), 1e-9)

    morning_peak = np.exp(-0.5 * ((hour - 8.0) / 1.8) ** 2)
    midday_plateau = np.exp(-0.5 * ((hour - 13.5) / 3.2) ** 2)
    evening_peak = np.exp(-0.5 * ((hour - 19.5) / 2.6) ** 2)

    cooling = np.clip(temperature - 22.0, 0.0, None) / 12.0
    heating = np.clip(14.0 - temperature, 0.0, None) / 12.0
    weekend_factor = np.where(weekday >= 5, 0.94, 1.0)
    seasonal_factor = 1.0 + 0.06 * np.cos(2.0 * np.pi * (dayofyear - 220.0) / 366.0)
    monthly_factor = 1.0 + 0.02 * np.sin(2.0 * np.pi * (month - 1.0) / 12.0)
    deterministic_daily_factor = 1.0 + 0.03 * np.sin(2.0 * np.pi * dayofyear / 17.0) + 0.02 * np.cos(2.0 * np.pi * dayofyear / 29.0)
    cloud_factor = 1.0 + 0.05 * cloud_cover
    solar_relief = 1.0 - 0.05 * pv_relief

    load_shape = 0.42 + 0.19 * morning_peak + 0.08 * midday_plateau + 0.31 * evening_peak
    weather_factor = 1.0 + 0.24 * cooling + 0.10 * heating
    load = load_shape * weekend_factor * seasonal_factor * monthly_factor * deterministic_daily_factor * cloud_factor * solar_relief * weather_factor
    load = np.clip(load, 0.25, None)
    load = load / max(float(np.max(load)), 1e-9) * float(peak_load_w)
    return pd.Series(load, index=timestamps, dtype=float, name="value")


def build_price_reference_15min(index: pd.DatetimeIndex) -> pd.Series:
    """Expand the checked-in hourly TOU tariff to 15-minute resolution."""

    hourly_template = _legacy_tariff_hourly_template()
    steps_per_day = 24 * 4
    daily_15min = np.repeat(hourly_template, 4)
    prices = np.tile(daily_15min, int(np.ceil(len(index) / steps_per_day)))[: len(index)]
    return pd.Series(prices, index=index, dtype=float, name="value")


def _write_timeseries(path: Path, index: pd.DatetimeIndex, values: np.ndarray | pd.Series) -> None:
    frame = pd.DataFrame({"datetime": index, "value": np.asarray(values, dtype=float)})
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def prepare_reference_energy_profiles() -> PreparedReferenceOutputs:
    """Materialize raw hourly weather and processed 15-minute reference datasets."""

    ensure_raw_weather_from_legacy()
    raw_dir = raw_weather_dir()
    processed_dir = processed_reference_dir()
    processed_dir.mkdir(parents=True, exist_ok=True)

    hourly_weather = load_raw_hourly_weather()
    merged_hourly_weather_csv = raw_dir / f"{REFERENCE_SLUG}_hourly_2023_2024.csv"
    hourly_weather.to_csv(merged_hourly_weather_csv, index=False)

    weather_15min = resample_weather_to_15min(hourly_weather)
    weather_15min_csv = processed_dir / "weather_15min.csv"
    weather_15min.to_csv(weather_15min_csv, index=False)

    timestamps = pd.to_datetime(weather_15min["datetime"])
    pv_reference = pd.to_numeric(weather_15min["pv_power_watts"], errors="coerce").fillna(0.0).clip(lower=0.0)
    pv_15min_csv = processed_dir / "pv_reference_15min.csv"
    _write_timeseries(pv_15min_csv, timestamps, pv_reference)

    load_reference = synthesize_reference_load_15min(weather_15min)
    load_reference_15min_csv = processed_dir / "load_reference_15min.csv"
    _write_timeseries(load_reference_15min_csv, timestamps, load_reference.to_numpy(dtype=float))

    price_reference = build_price_reference_15min(pd.DatetimeIndex(timestamps))
    price_15min_csv = processed_dir / "price_reference_15min.csv"
    _write_timeseries(price_15min_csv, timestamps, price_reference.to_numpy(dtype=float))

    cigre_dir = processed_network_15min_dir("cigre_eu_lv")
    ieee33_dir = processed_network_15min_dir("ieee33")
    canonical_cigre_dir = canonical_network_dir("cigre_eu_lv")
    canonical_ieee33_dir = canonical_network_dir("ieee33")
    case_targets = {
        str(cigre_dir.resolve()): (cigre_dir, CIGREEuropeanLVConfig.load_max_power, CIGREEuropeanLVConfig.pv_max_power),
        str(ieee33_dir.resolve()): (ieee33_dir, IEEE33Config.load_max_power, IEEE33Config.pv_max_power),
        str(canonical_cigre_dir.resolve()): (
            canonical_cigre_dir,
            CIGREEuropeanLVConfig.load_max_power,
            CIGREEuropeanLVConfig.pv_max_power,
        ),
        str(canonical_ieee33_dir.resolve()): (
            canonical_ieee33_dir,
            IEEE33Config.load_max_power,
            IEEE33Config.pv_max_power,
        ),
    }
    for case_dir, load_peak, pv_peak in case_targets.values():
        load_scaled = load_reference.to_numpy(dtype=float) / max(float(load_reference.max()), 1e-9) * float(load_peak)
        pv_scaled = pv_reference.to_numpy(dtype=float) / max(float(np.max(pv_reference)), 1e-9) * float(pv_peak)
        _write_timeseries(case_dir / "load.csv", timestamps, load_scaled)
        _write_timeseries(case_dir / "pv.csv", timestamps, pv_scaled)
        _write_timeseries(case_dir / "price.csv", timestamps, price_reference.to_numpy(dtype=float))

    metadata = {
        "location": REFERENCE_LOCATION,
        "source": "Oikolab ERA5 hourly weather (checked-in yearly files copied to data/raw)",
        "raw_period_start": str(REFERENCE_START),
        "raw_period_end": str(REFERENCE_END_HOURLY),
        "processed_period_end": str(REFERENCE_END_15MIN),
        "processed_frequency": REFERENCE_FREQ_15MIN,
        "raw_hourly_rows": int(len(hourly_weather)),
        "processed_rows": int(len(weather_15min)),
        "processed_network_cases": ["cigre_eu_lv", "ieee33"],
        "price_profile": {
            "source": "data/legacy/yearly/tariff/price_profile.csv",
            "hourly_values": [float(value) for value in _legacy_tariff_hourly_template().tolist()],
            "unique_prices": sorted(float(value) for value in np.unique(_legacy_tariff_hourly_template())),
        },
        "reference_load_peak_w": float(REFERENCE_LOAD_PEAK_W),
    }
    metadata_json = processed_dir / "metadata.json"
    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return PreparedReferenceOutputs(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        merged_hourly_weather_csv=merged_hourly_weather_csv,
        weather_15min_csv=weather_15min_csv,
        pv_15min_csv=pv_15min_csv,
        load_reference_15min_csv=load_reference_15min_csv,
        price_15min_csv=price_15min_csv,
        cigre_dir=cigre_dir,
        ieee33_dir=ieee33_dir,
        canonical_cigre_dir=canonical_cigre_dir,
        canonical_ieee33_dir=canonical_ieee33_dir,
        metadata_json=metadata_json,
    )
