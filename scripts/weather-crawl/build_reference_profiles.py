"""Build raw and processed Los Angeles reference datasets for the project."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from microgrid_sim.data.weather_profiles import prepare_reference_energy_profiles


def main() -> int:
    outputs = prepare_reference_energy_profiles()
    payload = {
        "raw_dir": str(outputs.raw_dir),
        "processed_dir": str(outputs.processed_dir),
        "merged_hourly_weather_csv": str(outputs.merged_hourly_weather_csv),
        "weather_15min_csv": str(outputs.weather_15min_csv),
        "pv_15min_csv": str(outputs.pv_15min_csv),
        "load_reference_15min_csv": str(outputs.load_reference_15min_csv),
        "price_15min_csv": str(outputs.price_15min_csv),
        "cigre_dir": str(outputs.cigre_dir),
        "ieee33_dir": str(outputs.ieee33_dir),
        "canonical_cigre_dir": str(outputs.canonical_cigre_dir),
        "canonical_ieee33_dir": str(outputs.canonical_ieee33_dir),
        "metadata_json": str(outputs.metadata_json),
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
