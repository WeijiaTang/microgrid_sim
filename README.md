# microgrid-sim

A research platform for Deep Reinforcement Learning (DRL) applied to microgrid battery storage control. Built on pandapower (power flow) and stable-baselines3 (DRL agents).

## Quick Start

```bash
uv sync
uv run python -m microgrid_sim.cli smoke --case ieee33 --model simple --days 1 --steps 4
```

## Core Features

- **Multi-fidelity battery models**: Simple efficiency-based model (EBM) and Thevenin-based physical model (PBM) with configurable loss terms
- **Safety shield**: Inventory Teacher mechanism for constraint-aware action correction and behavior regulation
- **Cross-fidelity & cross-year evaluation**: Train on one battery model, evaluate on another; train on 2023 data, test on 2024
- **Network support**: IEEE 33-bus and CIGRE European LV distribution networks
- **Multiple DRL algorithms**: SAC, PPO, TD3, DDPG, D4PG, TQC, TRPO
- **Morphology-aware validation**: BalancedGate (P4) protocol for selecting checkpoints with healthy battery behavior

## Entry Points

| Purpose | Command |
|---|---|
| Smoke test | `uv run python -m microgrid_sim.cli smoke --case ieee33 --model simple --days 1 --steps 4` |
| DRL training/eval | `uv run python scripts/analysis/short_cross_fidelity_probe.py` |
| GA baseline | `uv run python scripts/analysis/genetic_dispatch_baseline.py` |
| MILP upper bound | `uv run python scripts/analysis/full_year_oracle_compare.py` |
| Results summary | `uv run python scripts/analysis/fidelity_summary_tables.py` |
| Tests | `uv run pytest tests/ -q` |

See `CLI.md` for the full command reference.

## Data

Datasets under `data/` are not distributed in this repository and must be requested from the authors.

## Project Structure

```
src/microgrid_sim/
  cli.py                   # CLI entrypoint
  cases.py                 # Case configuration (IEEE33, CIGRE LV)
  envs/                    # Gym environment, wrappers, observation/reward builders
  models/                  # Battery models (Simple, Thevenin)
  network/                 # Power network builders, adapters, constraint checks
  training/                # Shield, behavior cloning, offline dataset
scripts/analysis/          # DRL training pipeline, baselines, results
data/                      # Network profiles (load, PV, price)
```

## License

See `LICENSE`.