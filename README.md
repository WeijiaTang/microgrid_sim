# Microgrid DRL Simulation

The datasets expected under `data/` are not distributed publicly in this repository. They can be provided by the authors upon reasonable request.

Residential:

```bash
python experiments/run_experiments.py residential-d4-run --agent sac --steps 666000 --train-days 365 --eval-days 365 --seeds 44,45,46,47,48,49 --fair-train --enable-validation-selection --output-dir ./results/ --models-dir ./models/
```

CIGRE:

```bash
python experiments/run_experiments.py cigre-gap-run --agent sac --steps 910000 --seeds 42,43,44,45,46,47 --output-dir ./results/ --models-dir ./models/
```
