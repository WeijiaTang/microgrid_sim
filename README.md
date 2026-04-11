# Microgrid DRL Simulation

The datasets expected under `data/` are not distributed publicly in this repository. They can be provided by the authors upon reasonable request.

The repository now uses a network-first data layout:

- `data/raw/weather/oikolab/los_angeles/` for raw 2023-2024 hourly weather
- `data/processed/reference_15min/...` for two-year 15-minute reference datasets
- `data/processed/network_15min/<case>/load.csv,pv.csv,price.csv` for current `NetworkMicrogridEnv` cases
- `data/legacy/aggregated/...` and `data/legacy/yearly/...` for historical aggregated-case assets
- The repository now bundles canonical `cigre_eu_lv` and `ieee33` network profiles directly under `data/processed/network_15min/`

See `data/README.md` for the canonical directory structure.

The project is now network-first and should be run via package entrypoints instead of the legacy `experiments/` folder.

Quick smoke test:

```bash
uv run python -m microgrid_sim.cli smoke --case cigre --model thevenin --days 1 --steps 4
```

IEEE 33-bus smoke test:

```bash
uv run python -m microgrid_sim.cli smoke --case ieee33 --model simple --days 1 --steps 4
```

Paper figure regeneration:

```bash
uv run python scripts/plot/paper_case_study_figures.py --help
```

Prepare raw weather and 15-minute processed profiles:

```bash
uv run python scripts/weather-crawl/build_reference_profiles.py
```

Legacy `experiments/` workflows have been removed as part of the network-first migration.
