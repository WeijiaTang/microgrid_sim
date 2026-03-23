# Microgrid DRL Simulation

## Quick Start

```bash
# 1) Install dependencies (without PyTorch wheel)
uv sync

# 2) Install PyTorch based on your platform
# CUDA 12.1
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
# CUDA 11.8
uv pip install torch --index-url https://download.pytorch.org/whl/cu118
# CPU only
uv pip install torch --index-url https://download.pytorch.org/whl/cpu

# 3) Show unified CLI
uv run python experiments/run_experiments.py --help
```

## Unified CLI Commands

Main commands implemented in `experiments/run_experiments.py`:

```bash
# Train a single DRL model
uv run python experiments/run_experiments.py train --agent sac --model thevenin --steps 300000

# Evaluate a trained model
uv run python experiments/run_experiments.py eval --agent sac --model thevenin --load-model ./models/sac_thevenin.zip

# Evaluate on MG-CIGRE PBM environment
uv run python experiments/run_experiments.py cigre-eval --load-model ./models/sac_thevenin.zip --days 365

# Run full comparison
uv run python experiments/run_experiments.py compare --steps 300000 --days 365 --output ./results

# Run eta sensitivity experiment
uv run python experiments/run_experiments.py eta-compare --etas 0.9,0.95,1.0 --seeds 42,43,44

# Run MILP baseline for both MG-RES and MG-CIGRE
uv run python experiments/run_experiments.py milp-both --res-days 30 --cigre-days 365
```

Embedded experiment commands (invoked from `run_experiments.py`):

```bash
# Focused CIGRE PBM-vs-EBM runner
uv run python experiments/run_experiments.py cigre-gap-run --steps 880000 --output-dir ./results/cigre_d4_880k --models-dir ./models/cigre_d4_880k

# Multi-seed CIGRE run (writes per-seed subfolders under output-dir/models-dir)
uv run python experiments/run_experiments.py cigre-gap-run --steps 880000 --seeds 42,43,44,45,46,47 --output-dir ./results/cigre_d4_multiseed_880k --models-dir ./models/cigre_d4_multiseed_880k

# Focused residential PBM-vs-EBM runner
# Default residential benchmark is grid-connected, paper-scale, frame-stacked, and battery-first.
uv run python experiments/run_experiments.py residential-d4-run --steps 300000 --output-dir ./results/residential_lowload_300k --models-dir ./models/residential_lowload_300k

# Residential annual eval (365d) + peak-import penalty + 6 seeds
uv run python experiments/run_experiments.py residential-d4-run --steps 666000 --train-days 365 --eval-days 365 --seeds 42,43,44,45,46,47 --peak-import-penalty-per-kw 1.5 --peak-import-threshold-kw 3.0 --output-dir ./results/residential_lowload_multiseed_666k --models-dir ./models/residential_lowload_multiseed_666k

# Residential annual eval (365d) + optional backup generator
uv run python experiments/run_experiments.py residential-d4-run --steps 300000 --enable-generator --output-dir ./results/residential_backupgen_300k --models-dir ./models/residential_backupgen_300k

# Residential annual eval (365d) + islanded (NSE) stress test + fair training protocol + 6 seeds
uv run python experiments/run_experiments.py residential-d4-run --steps 910000 --train-days 365 --eval-days 365 --seeds 42,43,44,45,46,47 --fair-train --islanded --nse-penalty-per-kwh 100.0 --midday-pv-boost-multiplier 1.25 --evening-load-boost-multiplier 1.35 --battery-power-scale 2.0 --output-dir ./results/residential_islanded_fair_multiseed_910k --models-dir ./models/residential_islanded_fair_multiseed_910k
 
# Multi-window eval (robustness across start hours)
uv run python experiments/run_experiments.py residential-d4-run --steps 300000 --seed 44 --eval-start-hours 0,6,12,18 --output-dir ./results/residential_lowload_300k --models-dir ./models/residential_lowload_300k

# Multi-seed + multi-window eval (writes per-seed subfolders under output-dir/models-dir)
uv run python experiments/run_experiments.py residential-d4-run --steps 300000 --seeds 42,43,44,45,46,47 --eval-start-hours 0,6,12,18 --output-dir ./results/residential_lowload_multiseed_300k --models-dir ./models/residential_lowload_multiseed_300k

# Residential D9 grid search
uv run python experiments/run_experiments.py residential-d9-grid --steps 5000 --output-root ./results/residential_d9_grid --models-root ./models/residential_d9_grid
```

These commands run from embedded modules under `experiments/` and do not require a separate `scripts/` directory.

Common add-ons:

```bash
# Smoke run (quick self-check after CLI/env changes)
uv run python experiments/run_experiments.py compare --fast --output ./results/smoke
#
# Note: --fast is intentionally lightweight (short horizons + small training steps) and is meant for smoke testing only.

# Fix random seed (reproducible runs)
uv run python experiments/run_experiments.py train --agent sac --model thevenin --steps 50000 --seed 42

# Force CPU (no GPU or debugging GPU differences)
uv run python experiments/run_experiments.py train --agent sac --model thevenin --steps 50000 --cpu

# eval requires --load-model; --output exports timeseries CSV
uv run python experiments/run_experiments.py eval --agent sac --model thevenin --load-model ./models/sac_thevenin.zip --days 365 --output ./results/eval_thevenin

# MG-CIGRE eval can override data directory (default: auto-detect under project data/)
uv run python experiments/run_experiments.py cigre-eval --load-model ./models/sac_thevenin.zip --days 365 --data-dir ./data

# eta-compare can override model directory and output directory
uv run python experiments/run_experiments.py eta-compare --etas 0.9,0.95,1.0 --seeds 42,43,44 --models-dir ./models --output ./results/eta_compare
```

## Data Layout

- `data/mg_res/` requires `load.csv`, `pv.csv`, `price.csv`
- `data/mg_cigre/` requires `load.csv`, `pv.csv`, `price.csv` and either `other.csv` or `net.csv`
- `data_template/` keeps open-source templates only

Use `MICROGRID_SIM_DATA_DIR` to point to your private dataset root.

## Prompt For Other AI Assistants

Use this prompt when handing this repository to another coding assistant:

```text
You are modifying a microgrid DRL simulation/experiment repository with a unified CLI at experiments/run_experiments.py.

Goals and constraints:
1) Keep MG-RES and MG-CIGRE logic case-aware and consistent with paper-style assumptions.
2) Prefer minimal, surgical changes; do not redesign directory structure.
3) Validate every CLI or environment change with --help and at least one smoke run (compare --fast).
4) Keep output contracts stable: results/*/report.json and results/*/summary.csv (avoid renaming/removing existing fields).
5) If you add or rename commands, update both README.md (English) and README-zhcn.md (Chinese) in the same patch.

High-level structure:
- experiments/run_experiments.py: unified CLI (train/eval/compare/eta-compare/milp-both + embedded commands).
- experiments/_cmd_*.py: embedded experiment commands forwarded by the unified CLI.
- src/microgrid_sim/envs/microgrid.py: environment dynamics + cost/reward computation.
- src/microgrid_sim/cases.py: case configs (MG-RES / MG-CIGRE) and feature toggles.
- src/microgrid_sim/models/: battery models (thevenin/simple) and dispatchable units.

Primary entrypoint:
- uv run python experiments/run_experiments.py --help

Important:
- Keep command behavior deterministic and reproducible (prefer fixed --seed).
- Do not remove existing experiment result fields unless explicitly requested.
```
