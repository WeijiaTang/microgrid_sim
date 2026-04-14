# Kaggle Runbook

本文件用于在 Kaggle 上复现实验，并保证输出工件可直接用于论文结果整理。

当前推荐的 Kaggle 路径是：

- 代码根目录：`/kaggle/input/datasets/tttwwwjjj/microgrid-code`
- 数据 processed 目录：`/kaggle/input/datasets/tttwwwjjj/microgrid-data/data/processed`

注意：

- 不建议把代码路径直接写成 `/kaggle/input/datasets/tttwwwjjj/microgrid-code/src/microgrid_sim`。
- 因为 notebook 需要找到仓库根目录下的 `pyproject.toml`、`scripts/`、`data/`、`src/`。
- 现在的 [`kaggle.ipynb`](/D:/EnergyStorage/Plan/Simple_Microgrid/microgrid_sim/kaggle.ipynb) 已经支持你误填到子目录时自动向上寻找仓库根目录。
- 如果 Kaggle 挂载路径不是 `/kaggle/input/<slug>/...`，而是 `/kaggle/input/datasets/<owner>/<slug>/...`，notebook 现在也会递归识别。
- 如果代码数据集里没有 `pyproject.toml`，但有 `src/microgrid_sim`，notebook 也会退回到 `PYTHONPATH` 模式继续运行。

## 1. 目标

Kaggle 侧推荐使用以下主链路：

- 主实验：`scripts/analysis/short_cross_fidelity_probe.py`
- 理论上界：`scripts/analysis/full_year_oracle_compare.py`
- 快速全年对照：`scripts/analysis/full_year_heuristic_lite_compare.py`

输出目标统一为：

- `summary.csv`
- `summary.json`
- `trajectories/*.csv`
- 可选的 `protocol_summary.csv` / `gap_summary.csv`

这些文件已经与仓库内聚合脚本兼容，可继续喂给：

- `scripts/analysis/fidelity_summary_tables.py`
- `scripts/analysis/paper_appendix_analysis.py`

## 2. 当前代码审查结论

对照 [`docs/chat/idea.md`](/D:/EnergyStorage/Plan/Simple_Microgrid/microgrid_sim/docs/chat/idea.md)，当前代码已经满足其中大部分创新验证前提，但有一个重要边界需要如实说明。

已满足：

- `Cross-fidelity train-test mismatch`
  - `short_cross_fidelity_probe.py` 已支持 `--train-models` 与 `--test-models` 分离，也支持 `simple+thevenin` 这类 mixed-fidelity 训练串联。
- `Value-of-fidelity regime map`
  - `fidelity_regime_map.py` 已支持多 `regime` 扫描，可直接分析不同工况下保真度差距。
- `Mixed-fidelity or curriculum training`
  - `short_cross_fidelity_probe.py` 已支持 mixed-fidelity 阶段比例与阶段学习率。
- 关键闭环修复
  - 终端 SOC 惩罚不再被 step reward clipping 吞掉。
  - observation 已包含 `soc_distance_to_target`、`available_charge_room`、`available_discharge_room`。
  - battery feasibility aware 模式默认使用负的 `battery_infeasible_penalty`。

尚未完全满足：

- `Battery-fidelity ladder for RL-based EMS`
  - 当前运行时明确支持的储能模型是 `simple`、`thevenin_loss_only`、`thevenin`。
  - 这已经能覆盖 `EBM -> loss-aware -> full Thevenin` 的主线，但还没有把 `Rint-only` 与 `Rint + thermal/stress` 作为独立、标准化、可直接从 CLI 选择的完整四层 ladder 明确固化。

因此，论文表述建议是：

- 现在已经足以支撑“保真度失配、工况敏感性、混合保真训练”的创新验证。
- 如果要把论文写成严格的“四层 battery-fidelity ladder benchmark”，还需要再把中间两层做成正式的命名模型与统一实验入口。

## 3. 本地短测结论

本地 CPU 已完成以下 DRL 短测：

- [`results/review_runs/ieee33_ppo_5k_7deval_seed42/summary.csv`](/D:/EnergyStorage/Plan/Simple_Microgrid/microgrid_sim/results/review_runs/ieee33_ppo_5k_7deval_seed42/summary.csv)
- [`results/review_runs/ieee33_sac_5k_7deval_seed42/summary.csv`](/D:/EnergyStorage/Plan/Simple_Microgrid/microgrid_sim/results/review_runs/ieee33_sac_5k_7deval_seed42/summary.csv)
- [`results/review_runs/cigre_ppo_5k_7deval_seed42/summary.csv`](/D:/EnergyStorage/Plan/Simple_Microgrid/microgrid_sim/results/review_runs/cigre_ppo_5k_7deval_seed42/summary.csv)
- [`results/review_runs/cigre_sac_5k_7deval_seed42/summary.csv`](/D:/EnergyStorage/Plan/Simple_Microgrid/microgrid_sim/results/review_runs/cigre_sac_5k_7deval_seed42/summary.csv)

短测读取结论：

- `IEEE33 + SAC` 明显优于 `IEEE33 + PPO` 的 5k 预算短测。
- `CIGRE` 在 5k 预算下更容易掉进低 SOC 边界吸引子。
- `PPO 20k + 30d eval` 的 CPU 本地试跑超过 15 分钟，说明 Kaggle 上不适合盲目把 PPO 预算直接抬高。

代表性指标：

- `IEEE33 + PPO 5k`
  - `final_soc = 0.1`
  - `soc_lower_dwell_fraction = 0.9167`
- `IEEE33 + SAC 5k`
  - `final_soc = 0.3694`
  - `soc_lower_dwell_fraction = 0.2560`
- `CIGRE + PPO 5k`
  - `final_soc = 0.1`
  - `soc_lower_dwell_fraction = 0.9673`
- `CIGRE + SAC 5k`
  - `final_soc = 0.1`
  - `soc_lower_dwell_fraction = 0.6592`

## 4. Kaggle 参数建议

推荐把 Kaggle 实验分成三档：

### 4.1 Smoke

用途：

- 检查数据、安装、日志、输出目录、CSV 工件是否正常。

建议：

- `agent = sac`
- `train_steps = 5000`
- `eval_days = 7`
- `case = ieee33`

### 4.2 Scout

用途：

- 快速摸参数，比较 PPO/SAC、IEEE33/CIGRE 的策略形态。

建议：

- `agent = sac`
- `train_steps = 10000 ~ 20000`
- `eval_days = 7 ~ 30`
- `train_episode_days = 30`
- 保持 action regularization 全开

### 4.3 Serious

用途：

- 论文主结果或至少接近论文主结果的 held-out 年度实验。

建议：

- `train_year = 2023`
- `eval_year = 2024`
- `days = 365`
- `train_episode_days = 30`
- `eval_days = 365`
- `eval_full_horizon = true`
- `agent` 先从 `sac` 开始，再决定是否追加 `ppo`
- `train_steps` 从 `100000+` 起步，严肃比较用 `200000 ~ 300000`

## 5. 固定正则配置

Kaggle 与本地建议统一使用：

```text
--action-smoothing-coef 0.5
--action-max-delta 0.1
--action-rate-penalty 0.05
--battery-feasibility-aware
--symmetric-battery-action
```

原因：

- 这是当前仓库中最能抑制边界僵尸策略和不可行动作刷屏的稳定配置。
- 同时结果文件里会记录 `battery_action_infeasible_gap`、`battery_action_infeasible_penalty`、`soc_lower_dwell_fraction` 等指标，便于论文分析。

## 6. 年度协议提醒

不要把以下命令当成全年证据：

```text
--days 365 --eval-steps 96
```

这只表示“365 天环境定义 + 96 步评估”，不是全年评估。

要做全年结论，必须使用：

```text
--train-year 2023
--eval-year 2024
--eval-days 365
--eval-full-horizon
```

## 7. Notebook 说明

根目录的 [`kaggle.ipynb`](/D:/EnergyStorage/Plan/Simple_Microgrid/microgrid_sim/kaggle.ipynb) 已经按 Kaggle 工作流准备好：

- 自动定位 `/kaggle/input` 里的代码数据集
- 将代码复制到 `/kaggle/working`
- 用 `pip install -e .` 安装仓库
- 一格参数区控制 DRL / Oracle / GA-lite 运行
- 自动读取 `summary.csv`
- 自动打包输出目录

推荐首次运行顺序：

1. `IEEE33 + SAC + 5k + 7d eval`
2. `IEEE33 + SAC + 20k + 30d eval`
3. `CIGRE + SAC + 20k + 30d eval`
4. `Oracle + Heuristic-lite`
5. 预算充足时再补 `PPO`

Oracle 年度对照必须与 DRL 使用同一时间窗。
如果 DRL 用的是 `--eval-year 2024 --eval-days 365`，那么 Oracle 也必须传 `--year 2024 --days 365`，否则会默认从 2023 起点取窗，导致基线年份错位。
