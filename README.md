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

## Year-split generalization (2023 train / 2024 eval)

For a reproducible DRL year-split protocol with deterministic evidence artifacts (results + TensorBoard + logs) and a convergence acceptance checklist, see:

- `CLI.md`

## 365-day benchmark commands

判读提醒：
- `--days 365` 只定义环境长度。
- 如果仍然使用 `--eval-steps 96` 且没有 `--eval-full-horizon`，那依然只是 96 步、也就是 1 天评估，不是全年评估。

全年 Oracle-MILP 上界：

```powershell
uv run python scripts/analysis/full_year_oracle_compare.py `
  --cases ieee33,cigre --regimes network_stress `
  --reward-profile paper_balanced --battery-model simple `
  --days 365 --seed 42 --efficiency-model realistic `
  --output-dir results/full_year_oracle_compare_seed42
```

全年 rolling GA：

```powershell
uv run python scripts/analysis/genetic_dispatch_baseline.py `
  --cases ieee33,cigre --regimes network_stress `
  --battery-models none,simple,thevenin `
  --reward-profile paper_balanced `
  --days 365 --seed 42 `
  --rolling-window-days 7 --rolling-stride-days 1 `
  --population-size 8 --generations 4 --elite-count 2 --mutation-scale 0.10 `
  --output-dir results/ga_baseline_rolling_365d_seed42_p8g4_w7s1
```

全年 PPO held-out year split：

```powershell
$exp = "yearsplit_ieee33_ppo_300k_train2023_eval2024_365d"
uv run python scripts/analysis/short_cross_fidelity_probe.py `
  --cases ieee33 --regimes network_stress `
  --reward-profile paper_balanced --agent ppo `
  --train-models simple --test-models simple `
  --train-steps 300000 --eval-steps 0 --eval-full-horizon `
  --days 365 `
  --train-year 2023 --train-episode-days 30 --train-random-start-within-year `
  --eval-year 2024 --eval-days 365 `
  --seed 42 --device cpu `
  --action-smoothing-coef 0.5 --action-max-delta 0.1 --action-rate-penalty 0.05 `
  --battery-feasibility-aware --symmetric-battery-action `
  --tensorboard-log results/myruns/tensorboard/$exp `
  --tb-log-name $exp `
  --output-dir results/myruns/$exp
```

全年 SAC held-out year split：

```powershell
$exp = "yearsplit_ieee33_sac_300k_train2023_eval2024_365d"
uv run python scripts/analysis/short_cross_fidelity_probe.py `
  --cases ieee33 --regimes network_stress `
  --reward-profile paper_balanced --agent sac `
  --train-models simple --test-models simple `
  --train-steps 300000 --eval-steps 0 --eval-full-horizon `
  --days 365 `
  --train-year 2023 --train-episode-days 30 --train-random-start-within-year `
  --eval-year 2024 --eval-days 365 `
  --seed 42 --device cpu `
  --action-smoothing-coef 0.5 --action-max-delta 0.1 --action-rate-penalty 0.05 `
  --battery-feasibility-aware --symmetric-battery-action `
  --tensorboard-log results/myruns/tensorboard/$exp `
  --tb-log-name $exp `
  --output-dir results/myruns/$exp
```

把 `ieee33` 替换为 `cigre` 并同步修改 `$exp`/输出目录，即可运行 CIGRE 的全年 PPO/SAC 长跑。更完整的命令清单和协议说明见 `CLI.md`。
