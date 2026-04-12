# CLI

本文件收录本仓库可复现实验的常用命令（以仓库根目录 `microgrid_sim/` 为工作目录）。

说明：
- 本仓库的主线“跨保真度/跨年度”DRL 对比入口是 `scripts/analysis/short_cross_fidelity_probe.py`。
- 目前仓库内未发现独立的 sklearn/xgboost 等“监督学习”训练脚本；非 DRL 的对比主要是 **优化/启发式 baseline**（MILP、规则、遗传算法）与 **结果聚合/制表脚本**。
- Windows PowerShell 下如需保存终端输出，使用 `| Tee-Object -FilePath <path>`。

---

## 0. 环境自检

先恢复 `uv` 环境，再跑单测确保环境可用：

```powershell
uv sync
```

如果要启用 `D4PG/DI-engine`：

```powershell
uv sync --extra d4pg
```

建议再跑单测：

```powershell
uv run pytest tests/envs/test_network_microgrid_env.py tests/envs/test_wrappers.py tests/analysis/test_short_cross_fidelity_probe.py tests/analysis/test_genetic_dispatch_baseline.py -q
```

---

## 1) DRL：2023 训练 / 2024 评估（季度 held-out 窗口）

推荐主协议：
- 训练环境长度：`--days 365`
- 训练采样窗口：`--train-episode-days 30 --train-random-start-within-year`
- 训练年份：`--train-year 2023`
- 正式评估年份：`--eval-year 2024`
- 正式评估长度：`--eval-days 365 --eval-full-horizon`

调试/短测协议：
- 训练窗口仍使用 `30` 天
- 评估先缩成 `30` 天 held-out 窗口
- 开启 TensorBoard 记录

### 1.1 最小复现（Q1 窗口 + 两个 seed）

- case: `ieee33`
- regime: `network_stress`
- reward: `paper_balanced`
- train: `2023` 年内随机 `7` 天窗口
- eval: `2024-01-01` 起 `30` 天窗口（offset=0）

```powershell
# seed42
uv run python scripts/analysis/short_cross_fidelity_probe.py `
  --cases ieee33 --regimes network_stress `
  --reward-profile paper_balanced --agent ppo `
  --train-models none,simple,thevenin --test-models none,simple,thevenin `
  --train-steps 200000 --eval-steps 0 --eval-full-horizon `
  --days 365 --train-year 2023 --train-episode-days 30 --train-random-start-within-year `
  --eval-year 2024 --eval-days 30 --eval-offset-days-within-year 0 `
  --action-smoothing-coef 0.5 --action-max-delta 0.1 --action-rate-penalty 0.05 `
  --battery-feasibility-aware --symmetric-battery-action `
  --seed 42 `
  --output-dir results/myruns/yearsplit_ieee33_q1_seed42_2023train_2024eval |
  Tee-Object -FilePath ./yearsplit_ieee33_q1_seed42_2023train_2024eval.log

# seed62
uv run python scripts/analysis/short_cross_fidelity_probe.py `
  --cases ieee33 --regimes network_stress `
  --reward-profile paper_balanced --agent ppo `
  --train-models none,simple,thevenin --test-models none,simple,thevenin `
  --train-steps 200000 --eval-steps 0 --eval-full-horizon `
  --days 365 --train-year 2023 --train-episode-days 30 --train-random-start-within-year `
  --eval-year 2024 --eval-days 30 --eval-offset-days-within-year 0 `
  --action-smoothing-coef 0.5 --action-max-delta 0.1 --action-rate-penalty 0.05 `
  --battery-feasibility-aware --symmetric-battery-action `
  --seed 62 `
  --output-dir results/yearsplit_ieee33_q1_seed62_2023train_2024eval |
  Tee-Object -FilePath ./yearsplit_ieee33_q1_seed62_2023train_2024eval.log
```

输出：
- `results/<exp>/summary.csv`
- `results/<exp>/summary.json`
- `results/<exp>/trajectories/*.csv`

### 1.1b 5k 收敛诊断（TensorBoard）

适合做 smoke test，看日志链路与 early-training 曲线是否正常，不适合作为最终性能结论：

```powershell
uv run python scripts/analysis/short_cross_fidelity_probe.py `
  --cases ieee33 --regimes network_stress `
  --reward-profile paper_balanced --agent ppo `
  --train-models simple --test-models simple `
  --train-steps 5000 --eval-steps 0 `
  --days 365 --train-year 2023 --train-episode-days 30 --train-random-start-within-year `
  --eval-year 2024 --eval-days 30 `
  --action-smoothing-coef 0.5 --action-max-delta 0.1 --action-rate-penalty 0.05 `
  --battery-feasibility-aware --symmetric-battery-action `
  --seed 42 `
  --tensorboard-log results/tensorboard/ppo_5k_ieee33_simple `
  --tb-log-name ppo_5k_ieee33_simple `
  --output-dir results/drl_5k_ieee33_simple_seed42

tensorboard --logdir results/tensorboard
```

建议重点看：
- `rollout/ep_rew_mean`
- `train/value_loss`
- `train/policy_gradient_loss`
- `train/entropy_loss`
- `train/std`

说明：
- 当前仓库里 `PPO` 默认 `n_steps=2048`，所以 `5k` 训练只会产生很少的更新点；更适合做日志/收敛方向检查，不适合做最终算法比较。
- 训练结果的 TensorBoard 元数据也会写入 `summary.csv` 的 `tensorboard_log_dir` 与 `tensorboard_run_name` 字段。

### 1.2 四季度评估（同一训练设定，eval offset = 0/91/182/274）

脚本一次只支持一个 `--eval-offset-days-within-year`，因此四季度建议分 4 次跑（输出目录分开）。

```powershell
# seed42, Q1
uv run python scripts/analysis/short_cross_fidelity_probe.py `
  --cases ieee33 --regimes network_stress `
  --reward-profile paper_balanced --agent ppo `
  --train-models none,simple,thevenin --test-models none,simple,thevenin `
  --train-steps 20000 --eval-steps 0 --eval-full-horizon `
  --days 7 --train-year 2023 --train-episode-days 7 --train-random-start-within-year `
  --eval-year 2024 --eval-days 30 --eval-offset-days-within-year 0 `
  --action-smoothing-coef 0.5 --action-max-delta 0.1 --action-rate-penalty 0.05 `
  --battery-feasibility-aware --symmetric-battery-action `
  --seed 42 --output-dir results/yearsplit_ieee33_q1_seed42

# seed42, Q2
uv run python scripts/analysis/short_cross_fidelity_probe.py `
  --cases ieee33 --regimes network_stress `
  --reward-profile paper_balanced --agent ppo `
  --train-models none,simple,thevenin --test-models none,simple,thevenin `
  --train-steps 20000 --eval-steps 0 --eval-full-horizon `
  --days 7 --train-year 2023 --train-episode-days 7 --train-random-start-within-year `
  --eval-year 2024 --eval-days 30 --eval-offset-days-within-year 91 `
  --action-smoothing-coef 0.5 --action-max-delta 0.1 --action-rate-penalty 0.05 `
  --battery-feasibility-aware --symmetric-battery-action `
  --seed 42 --output-dir results/yearsplit_ieee33_q2_seed42

# seed42, Q3
uv run python scripts/analysis/short_cross_fidelity_probe.py `
  --cases ieee33 --regimes network_stress `
  --reward-profile paper_balanced --agent ppo `
  --train-models none,simple,thevenin --test-models none,simple,thevenin `
  --train-steps 20000 --eval-steps 0 --eval-full-horizon `
  --days 7 --train-year 2023 --train-episode-days 7 --train-random-start-within-year `
  --eval-year 2024 --eval-days 30 --eval-offset-days-within-year 182 `
  --action-smoothing-coef 0.5 --action-max-delta 0.1 --action-rate-penalty 0.05 `
  --battery-feasibility-aware --symmetric-battery-action `
  --seed 42 --output-dir results/yearsplit_ieee33_q3_seed42

# seed42, Q4
uv run python scripts/analysis/short_cross_fidelity_probe.py `
  --cases ieee33 --regimes network_stress `
  --reward-profile paper_balanced --agent ppo `
  --train-models none,simple,thevenin --test-models none,simple,thevenin `
  --train-steps 20000 --eval-steps 0 --eval-full-horizon `
  --days 7 --train-year 2023 --train-episode-days 7 --train-random-start-within-year `
  --eval-year 2024 --eval-days 30 --eval-offset-days-within-year 274 `
  --action-smoothing-coef 0.5 --action-max-delta 0.1 --action-rate-penalty 0.05 `
  --battery-feasibility-aware --symmetric-battery-action `
  --seed 42 --output-dir results/yearsplit_ieee33_q4_seed42
```

同理把 `--seed 42` 改成 `--seed 62` 即可得到 seed62 的四季度结果。

### 1.3 Mixed-fidelity（可选）：`simple -> thevenin` 两阶段训练

```powershell
uv run python scripts/analysis/short_cross_fidelity_probe.py `
  --cases ieee33 --regimes network_stress `
  --reward-profile paper_balanced --agent ppo `
  --train-models simple+thevenin --test-models thevenin `
  --train-steps 20000 --eval-steps 0 --eval-full-horizon `
  --days 7 --train-year 2023 --train-episode-days 7 --train-random-start-within-year `
  --eval-year 2024 --eval-days 30 --eval-offset-days-within-year 0 `
  --mixed-fidelity-pretrain-fraction 0.5 `
  --action-smoothing-coef 0.5 --action-max-delta 0.1 --action-rate-penalty 0.05 `
  --battery-feasibility-aware --symmetric-battery-action `
  --seed 42 --output-dir results/yearsplit_ieee33_mixed_simple_thevenin_seed42
```

---

## 2) DRL：多 regime 的 fidelity gap 扫描（不含年度切分）

用于快速看不同 `regime` 下 `thevenin - simple` 的统计差异（默认短 horizon）：

```powershell
uv run python scripts/analysis/fidelity_regime_map.py `
  --cases ieee33 --regimes base,high_load,high_pv,network_stress,tight_soc `
  --train-models simple,thevenin --test-models simple,thevenin `
  --reward-profile paper_balanced --agent sac `
  --train-steps 2000 --eval-steps 96 --days 2 `
  --seeds 42,62 `
  --output-dir results/fidelity_regime_map_ieee33
```

输出：
- `results/fidelity_regime_map_ieee33/detail.csv`
- `results/fidelity_regime_map_ieee33/summary.csv`

---

## 3) 非 DRL（“ML/传统方法”baseline）：遗传算法调度（GA）

该脚本直接在固定窗口上优化 action 序列，不训练策略网络。

建议把 `none,simple,thevenin_loss_only,thevenin` 四个模型一起跑，直接对比“无储能 vs 三种储能保真度”。

### 3.1 7 天窗口：IEEE33

```powershell
uv run python scripts/analysis/genetic_dispatch_baseline.py `
  --cases ieee33 --regimes network_stress `
  --battery-models none,simple,thevenin_loss_only,thevenin `
  --reward-profile paper_balanced `
  --days 7 --seed 42 `
  --population-size 4 --generations 2 --elite-count 1 --mutation-scale 0.08 `
  --output-dir results/ga_baseline_ieee33_7d_seed42_p4g2
```

### 3.2 7 天窗口：CIGRE

```powershell
uv run python scripts/analysis/genetic_dispatch_baseline.py `
  --cases cigre --regimes network_stress `
  --battery-models none,simple,thevenin_loss_only,thevenin `
  --reward-profile paper_balanced `
  --days 7 --seed 42 `
  --population-size 4 --generations 2 --elite-count 1 --mutation-scale 0.08 `
  --output-dir results/ga_baseline_cigre_7d_seed42_p4g2
```

### 3.3 30 天窗口：IEEE33

长窗口 GA 成本很高，建议先用“粗搜索”配置做趋势检查：

```powershell
uv run python scripts/analysis/genetic_dispatch_baseline.py `
  --cases ieee33 --regimes network_stress `
  --battery-models none,simple,thevenin_loss_only,thevenin `
  --reward-profile paper_balanced `
  --days 30 --seed 42 `
  --population-size 2 --generations 1 --elite-count 1 --mutation-scale 0.08 `
  --output-dir results/ga_baseline_ieee33_30d_seed42_p2g1
```

### 3.4 30 天窗口：CIGRE

```powershell
uv run python scripts/analysis/genetic_dispatch_baseline.py `
  --cases cigre --regimes network_stress `
  --battery-models none,simple,thevenin_loss_only,thevenin `
  --reward-profile paper_balanced `
  --days 30 --seed 42 `
  --population-size 2 --generations 1 --elite-count 1 --mutation-scale 0.08 `
  --output-dir results/ga_baseline_cigre_30d_seed42_p2g1
```

每个输出目录都会包含：
- `summary.csv`
- `summary.json`
- `trajectories/*_actions.csv`
- `trajectories/*_trajectory.csv`

---

## 4) 结果聚合（把多个 DRL 实验目录汇总成表格）

把多个 `short_cross_fidelity_probe.py` 的 `summary.csv` 合并，并生成 paper-ready 表：

```powershell
uv run python scripts/analysis/fidelity_summary_tables.py `
  --input-dirs results/yearsplit_ieee33_q1_seed42,results/yearsplit_ieee33_q1_seed62 `
  --output-dir results/fidelity_summary_tables_yearsplit_ieee33
```

输出：
- `results/fidelity_summary_tables_yearsplit_ieee33/combined_detail.csv`
- `results/fidelity_summary_tables_yearsplit_ieee33/aggregate_summary.csv`
- `results/fidelity_summary_tables_yearsplit_ieee33/test_model_gap_summary.csv`
- `results/fidelity_summary_tables_yearsplit_ieee33/paper_key_metrics.csv`
- `results/fidelity_summary_tables_yearsplit_ieee33/best_train_family_by_test_env.csv`

---

## 5) 论文附录统计复算（基于 results/paper 工件）

如果你的 `results/paper/` 已存在（release 工件），可用该脚本复算附录统计：

```powershell
uv run python scripts/analysis/paper_appendix_analysis.py |
  Tee-Object -FilePath ./paper_appendix_analysis.log
```

输出目录：
- `results/paper/analysis/`

---

## 6) 绘图（paper figures）

```powershell
uv run python scripts/plot/paper_case_study_figures.py
```
