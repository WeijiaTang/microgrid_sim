# Microgrid DRL Simulation

The datasets expected under `data/` are not distributed publicly in this repository. They can be provided by the authors upon reasonable request.

Residential:

```bash
python experiments/run_experiments.py residential-d4-run --agent sac --steps 660000 --seeds 44,45,46,47,48,49 --fair-train --enable-validation-selection --validation-days 30 --validation-interval-steps 2000 --validation-start-hours 0,744,1416,2160,2880,3624,4344,5088,5832,6552,7296,8016 --validation-eval-battery-model thevenin --eval-start-hours 0,2184,4368,6552 --monthly-demand-charge-threshold-kw 2.0 --monthly-demand-charge-per-kw 16 --sac-ent-coef auto_0.01 --sac-target-entropy-scale 0.25 --action-smoothing-coef 0.5 --action-max-delta 0.35 --action-rate-penalty 0.02 --enable-symmetric-battery-action --output-dir ./results/ --models-dir ./models/
```

CIGRE:

```bash
python experiments/run_experiments.py cigre-gap-run --agent sac --steps 910000 --seeds 42,43,44,45,46,47 --output-dir ./results/ --models-dir ./models/
```
