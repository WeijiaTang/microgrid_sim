# Gemini CLI Instructional Context - Microgrid DRL Simulation

## Project Overview
`microgrid-sim` is a research platform for Deep Reinforcement Learning (DRL) applied to microgrid energy management. It integrates power network constraints with varying battery model fidelities to study "cross-fidelity" and "cross-year" generalization in DRL agents.

### Core Technologies
- **Language:** Python 3.9+
- **RL Frameworks:** `gymnasium`, `stable-baselines3`, `sb3-contrib`, `DI-engine` (optional for D4PG).
- **Power System Simulation:** `pandapower`.
- **Data Handling:** `numpy`, `pandas`, `scipy`.
- **Visualization:** `matplotlib`, `tensorboard`.
- **Dependency Management:** `uv`.

### Key Architecture
- **Environments (`src/microgrid_sim/envs/`):** `NetworkMicrogridEnv` is the primary Gymnasium environment. It supports multiple microgrid topologies (CIGRE LV, IEEE 33-bus) and reward profiles.
- **Behavior Regulation (SAIL-SAC):** The platform uses an "Inventory Teacher" mechanism to regulate DRL learning. This is distinct from a passive safety shield; it proactively generates "healthy" battery behavior (price-aware charging, mid-band SOC maintenance, reserve management) that is distilled into the agent's policy via Safe BC.
- **Battery Models:** Supports multiple fidelity levels (`none`, `simple`, `thevenin`).
- **Behavior Regulation (SAIL-SAC):**
  - **Inventory Teacher:** An active heuristic teacher that regulates DRL learning based on SOC zones, terminal targets, and price regimes.
  - **Online Safe BC (Distillation):** Periodically distills teacher-corrected actions from the replay buffer into the SAC actor to force-regulate learning behavior.
  - **Gated Validation:** Uses `inventory_value_gate` to select checkpoints based on inventory health, price-aware value recovery, and peak-reserve headroom.
- **Reporting (ABC-Layer):**
  - **A-Layer (Value):** Objective cost, savings vs. baseline.
  - **B-Layer (Inventory):** SOC midband dwell, target tracking MAE, boundary parking, and price-aware action fractions.
  - **C-Layer (Diagnostics):** Teacher activation fractions and guidance gap metrics.
- **Baselines:** Oracle-MILP, Genetic Algorithm (GA), and Heuristic Rules.
- **Experiment Protocol:** Focuses on year-split generalization (2023 train / 2024 eval). Checkpoint selection is "Inventory-First," prioritizing agents that demonstrate healthy battery morphology (e.g., `inventory_value_gate`) over raw reward.

---

## Building and Running

### Environment Setup
The project uses `uv` for dependency management.
```powershell
# Install dependencies
uv sync

# Optional: Install D4PG/DI-engine support
uv sync --extra d4pg

# Optional: Install CUDA 12.4 support
uv sync --extra cuda124
```

### Running Commands
The primary entry point is the `microgrid_sim.cli` module.
```powershell
# Run a smoke test
uv run python -m microgrid_sim.cli smoke --case ieee33 --model simple --days 1 --steps 4

# Run a main DRL experiment (short cross-fidelity probe)
uv run python scripts/analysis/short_cross_fidelity_probe.py --cases ieee33 --agent ppo ...
```
*Refer to `CLI.md` for a comprehensive list of experiment commands and long-running benchmark protocols.*

### Testing
Testing is handled by `pytest`.
```powershell
# Run all tests
uv run pytest

# Run specific environment tests
uv run pytest tests/envs/test_network_microgrid_env.py tests/envs/test_wrappers.py
```

---

## Development Conventions

### Code Style
- **Formatting:** Adheres to `black` (line length 100).
- **Structure:** 
  - Core logic resides in `src/microgrid_sim/`.
  - Analysis and plotting scripts are in `scripts/`.
  - Data files are organized in a network-first layout under `data/`.

### Testing Practices
- **Mirroring:** Test files in `tests/` mirror the structure of `src/microgrid_sim/`.
- **Regression:** New features or fixes should be accompanied by corresponding tests in the relevant subdirectory.
- **Smoke Tests:** Use `microgrid_sim.cli smoke` for quick environment validation.

### Data Management
- Raw weather data and processed network/load/PV profiles are stored in `data/`.
- **Note:** Large datasets are not tracked in Git; they must be provided or generated according to `data/README.md`.

---

## Important Files
- `pyproject.toml`: Project metadata and dependencies.
- `README.md`: General project overview and quick start.
- `CLI.md`: Detailed command reference for reproducible experiments.
- `src/microgrid_sim/envs/network_microgrid.py`: Core environment logic.
- `scripts/analysis/short_cross_fidelity_probe.py`: Main DRL training/evaluation script.
