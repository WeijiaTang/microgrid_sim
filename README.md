# Microgrid DRL Simulation

The datasets expected under `data/` are not distributed publicly in this repository. They can be provided by the authors upon reasonable request.

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

Legacy `experiments/` workflows have been removed as part of the network-first migration.
