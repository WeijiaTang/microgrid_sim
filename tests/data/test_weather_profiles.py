from __future__ import annotations

import json

import pandas as pd

from microgrid_sim.paths import PROCESSED_DATA_ROOT, RAW_DATA_ROOT


def test_repo_bundles_raw_los_angeles_hourly_weather():
    raw_dir = RAW_DATA_ROOT / "weather" / "oikolab" / "los_angeles"
    year_2023 = raw_dir / "los_angeles_hourly_2023.csv"
    year_2024 = raw_dir / "los_angeles_hourly_2024.csv"
    merged = raw_dir / "los_angeles_hourly_2023_2024.csv"

    assert raw_dir.is_dir()
    assert year_2023.exists()
    assert year_2024.exists()
    assert merged.exists()

    weather_2023 = pd.read_csv(year_2023)
    weather_2024 = pd.read_csv(year_2024)
    weather_merged = pd.read_csv(merged)

    assert len(weather_2023) == 365 * 24
    assert len(weather_2024) == 366 * 24
    assert len(weather_merged) == len(weather_2023) + len(weather_2024)
    assert "pv_power_watts" in weather_merged.columns


def test_repo_bundles_processed_15min_reference_profiles():
    processed_dir = PROCESSED_DATA_ROOT / "reference_15min" / "los_angeles_2023_2024"
    weather = pd.read_csv(processed_dir / "weather_15min.csv")
    pv = pd.read_csv(processed_dir / "pv_reference_15min.csv")
    load = pd.read_csv(processed_dir / "load_reference_15min.csv")
    price = pd.read_csv(processed_dir / "price_reference_15min.csv")
    metadata = json.loads((processed_dir / "metadata.json").read_text(encoding="utf-8"))

    expected_rows = (365 + 366) * 24 * 4
    assert len(weather) == expected_rows
    assert len(pv) == expected_rows
    assert len(load) == expected_rows
    assert len(price) == expected_rows
    assert float(pv["value"].min()) >= 0.0
    assert float(load["value"].min()) > 0.0
    assert set(price["value"].round(5).unique()) == {0.39073, 0.45100, 0.51373}
    assert metadata["processed_rows"] == expected_rows


def test_repo_bundles_processed_network_15min_case_profiles():
    expected = {
        "cigre_eu_lv": {"load_peak": 120_000.0, "pv_peak": 18_000.0},
        "ieee33": {"load_peak": 4_000_000.0, "pv_peak": 450_000.0},
    }
    expected_rows = (365 + 366) * 24 * 4

    for case_dirname, peaks in expected.items():
        case_dir = PROCESSED_DATA_ROOT / "network_15min" / case_dirname
        load = pd.read_csv(case_dir / "load.csv")
        pv = pd.read_csv(case_dir / "pv.csv")
        price = pd.read_csv(case_dir / "price.csv")

        assert len(load) == expected_rows
        assert len(pv) == expected_rows
        assert len(price) == expected_rows
        assert float(load["value"].max()) == peaks["load_peak"]
        assert float(pv["value"].max()) == peaks["pv_peak"]
