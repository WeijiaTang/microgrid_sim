# Microgrid DRL 仿真项目

## 快速开始

```bash
# 1) 安装依赖（不含 PyTorch）
uv sync

# 2) 安装 PyTorch（按你的平台选择）
# CUDA 12.1
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
# CUDA 11.8
uv pip install torch --index-url https://download.pytorch.org/whl/cu118
# 仅 CPU
uv pip install torch --index-url https://download.pytorch.org/whl/cpu

# 3) 查看统一 CLI
uv run python experiments/run_experiments.py --help
```

## 统一 CLI 命令

`experiments/run_experiments.py` 内置主命令：

```bash
# 训练单个 DRL 模型
uv run python experiments/run_experiments.py train --agent sac --model thevenin --steps 300000

# 评估已训练模型
uv run python experiments/run_experiments.py eval --agent sac --model thevenin --load-model ./models/sac_thevenin.zip

# 在 MG-CIGRE PBM 环境中评估
uv run python experiments/run_experiments.py cigre-eval --load-model ./models/sac_thevenin.zip --days 365

# 全量对比实验
uv run python experiments/run_experiments.py compare --steps 300000 --days 365 --output ./results

# eta 敏感性实验
uv run python experiments/run_experiments.py eta-compare --etas 0.9,0.95,1.0 --seeds 42,43,44

# MG-RES + MG-CIGRE 的 MILP 基线
uv run python experiments/run_experiments.py milp-both --res-days 30 --cigre-days 365
```

内嵌实验命令（统一通过 `run_experiments.py` 调用）：

```bash
# CIGRE 专项 PBM-vs-EBM
uv run python experiments/run_experiments.py cigre-gap-run --steps 880000 --output-dir ./results/cigre_d4_880k --models-dir ./models/cigre_d4_880k

# CIGRE 多 seed 运行（使用 --seeds 时，会在 output-dir/models-dir 下按 seed 建子目录）
uv run python experiments/run_experiments.py cigre-gap-run --steps 880000 --seeds 42,43,44,45,46,47 --output-dir ./results/cigre_d4_multiseed_880k --models-dir ./models/cigre_d4_multiseed_880k

# 住宅专项 PBM-vs-EBM
# 当前默认住宅 benchmark 为并网、论文尺度、带观测堆叠、且以电池为主。
uv run python experiments/run_experiments.py residential-d4-run --steps 300000 --output-dir ./results/residential_lowload_300k --models-dir ./models/residential_lowload_300k

# 住宅全年评估（365天）+ 峰值罚金 + 6 个随机种子
uv run python experiments/run_experiments.py residential-d4-run --steps 666000 --train-days 365 --eval-days 365 --seeds 42,43,44,45,46,47 --peak-import-penalty-per-kw 1.5 --peak-import-threshold-kw 3.0 --output-dir ./results/residential_lowload_multiseed_666k --models-dir ./models/residential_lowload_multiseed_666k

# 住宅全年评估（365天）+ 可选备用发电机
uv run python experiments/run_experiments.py residential-d4-run --steps 300000 --enable-generator --output-dir ./results/residential_backupgen_300k --models-dir ./models/residential_backupgen_300k

# 住宅全年评估（365天）+ 孤网（NSE）压力测试 + 公平训练协议 + 6 个随机种子
uv run python experiments/run_experiments.py residential-d4-run --steps 910000 --train-days 365 --eval-days 365 --seeds 42,43,44,45,46,47 --fair-train --islanded --nse-penalty-per-kwh 100.0 --midday-pv-boost-multiplier 1.25 --evening-load-boost-multiplier 1.35 --battery-power-scale 2.0 --output-dir ./results/residential_islanded_fair_multiseed_910k --models-dir ./models/residential_islanded_fair_multiseed_910k
 
# 多窗口评估（不同起始小时的鲁棒性）
uv run python experiments/run_experiments.py residential-d4-run --steps 300000 --seed 44 --eval-start-hours 0,6,12,18 --output-dir ./results/residential_lowload_300k --models-dir ./models/residential_lowload_300k

# 多 seed + 多窗口评估（使用 --seeds 时，会在 output-dir/models-dir 下按 seed 建子目录）
uv run python experiments/run_experiments.py residential-d4-run --steps 300000 --seeds 42,43,44,45,46,47 --eval-start-hours 0,6,12,18 --output-dir ./results/residential_lowload_multiseed_300k --models-dir ./models/residential_lowload_multiseed_300k

# 住宅 D9 参数网格搜索
uv run python experiments/run_experiments.py residential-d9-grid --steps 5000 --output-root ./results/residential_d9_grid --models-root ./models/residential_d9_grid
```

说明：这些命令已内嵌到 `experiments/` 模块中，不再依赖单独的 `scripts/` 目录。

常用补充：

```bash
# Smoke run（快速自检，推荐每次改 CLI/环境后先跑）
uv run python experiments/run_experiments.py compare --fast --output ./results/smoke
#
# 注意：--fast 会刻意保持轻量（更短的 horizon + 更少训练步数），只用于 smoke 测试。

# 固定随机种子（复现实验）
uv run python experiments/run_experiments.py train --agent sac --model thevenin --steps 50000 --seed 42

# 强制 CPU（没有 GPU 或排查 GPU 差异时）
uv run python experiments/run_experiments.py train --agent sac --model thevenin --steps 50000 --cpu

# eval 必须提供 --load-model；--output 用于导出 timeseries CSV
uv run python experiments/run_experiments.py eval --agent sac --model thevenin --load-model ./models/sac_thevenin.zip --days 365 --output ./results/eval_thevenin

# MG-CIGRE 评估可指定数据目录（默认自动探测 data/ 下数据集）
uv run python experiments/run_experiments.py cigre-eval --load-model ./models/sac_thevenin.zip --days 365 --data-dir ./data

# eta-compare 可指定模型目录与阈值（默认阈值 success-threshold-pct=1.0）
uv run python experiments/run_experiments.py eta-compare --etas 0.9,0.95,1.0 --seeds 42,43,44 --models-dir ./models --output ./results/eta_compare
```

## 数据目录约定

- `data/mg_res/` 需要：`load.csv`、`pv.csv`、`price.csv`
- `data/mg_cigre/` 需要：`load.csv`、`pv.csv`、`price.csv`，以及 `other.csv` 或 `net.csv`
- `data_template/` 为开源模板目录

可通过 `MICROGRID_SIM_DATA_DIR` 指向你的私有数据根目录。

## 给其他 AI 助手的提示词

可直接复制以下提示词给其他 AI 助手：

```text
你在修改一个微电网 DRL 仿真/实验仓库，统一 CLI 入口是 experiments/run_experiments.py。

项目目标与约束：
1）保持 MG-RES 与 MG-CIGRE 的 case-aware 逻辑，与论文风格假设一致；
2）采用最小改动（surgical changes），不要重构目录结构；
3）每次改 CLI 或环境后至少执行 --help 和一个 smoke run（compare --fast）；
4）保持输出契约稳定：results/*/report.json 与 results/*/summary.csv（尤其是已有字段不要随意改名/删除）；
5）新增或变更命令时，同步更新 README.md（英文）与 README-zhcn.md（中文）。

代码结构速览（优先入口）：
- experiments/run_experiments.py：统一 CLI（train/eval/compare/eta-compare/milp-both + 若干嵌入式子命令）。
- experiments/_cmd_*.py：嵌入式实验命令（由 run_experiments.py 转发参数）。
- src/microgrid_sim/envs/microgrid.py：核心环境动力学与 cost/reward 计算（电网购电、峰值罚金、月需量费、电池损耗/应力等）。
- src/microgrid_sim/cases.py：MG-RES / MG-CIGRE 的规模参数与开关（是否启用发电机、是否计入 component cost 等）。
- src/microgrid_sim/models/：电池模型（thevenin/simple）及可调度机组（DispatchableUnit）。

数据与路径约定：
- data/mg_res/ 与 data/mg_cigre/：数据集目录结构见 README；可用环境变量 MICROGRID_SIM_DATA_DIR 指向私有数据根目录。
- 统一 CLI 会尝试自动探测数据目录；必要时通过子命令参数显式覆盖（例如 cigre-eval --data-dir）。

输出契约（回归时重点关注）：
- 对比/专项实验会输出 results/**/report.json 与 results/**/summary.csv；字段应保持向后兼容。
- eval/cigre-eval 会导出 timeseries CSV（用于后处理与画图）。

复现与回归建议：
- 优先固定 --seed；避免隐式随机性导致结论漂移。
- 修改 reward/cost/物理模型后，先用 compare --fast 做 sanity check，再做长训练。

主入口：
- uv run python experiments/run_experiments.py --help
```
