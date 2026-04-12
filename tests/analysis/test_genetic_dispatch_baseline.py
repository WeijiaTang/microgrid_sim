from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_genetic_dispatch_baseline_generates_summary_and_trajectories(tmp_path: Path):
    output_dir = tmp_path / "ga_outputs"
    root = Path(__file__).resolve().parents[2]
    command = [
        sys.executable,
        str(root / "scripts" / "analysis" / "genetic_dispatch_baseline.py"),
        "--cases",
        "ieee33",
        "--regimes",
        "network_stress",
        "--battery-models",
        "simple",
        "--days",
        "1",
        "--seed",
        "42",
        "--population-size",
        "4",
        "--generations",
        "1",
        "--elite-count",
        "1",
        "--mutation-scale",
        "0.05",
        "--output-dir",
        str(output_dir),
    ]
    completed = subprocess.run(command, cwd=root, capture_output=True, text=True, check=True)
    assert "ga_dispatch" in completed.stdout

    summary_csv = output_dir / "summary.csv"
    summary_json = output_dir / "summary.json"
    trajectories_dir = output_dir / "trajectories"

    assert summary_csv.exists()
    assert summary_json.exists()
    assert trajectories_dir.exists()

    summary_df = pd.read_csv(summary_csv)
    assert len(summary_df) == 1
    assert summary_df.loc[0, "baseline"] == "ga_dispatch"
    assert summary_df.loc[0, "battery_model"] == "simple"
    assert "final_cumulative_cost" in summary_df.columns
    assert "final_cumulative_objective_cost" in summary_df.columns
    assert "total_terminal_soc_penalty" in summary_df.columns
    assert "objective_value" in summary_df.columns
    assert float(summary_df.loc[0, "objective_value"]) >= float(summary_df.loc[0, "final_cumulative_objective_cost"])

    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    assert len(payload) == 1
    assert payload[0]["case"] == "ieee33"

    trajectories = list(trajectories_dir.glob("*.csv"))
    assert len(trajectories) >= 2


def test_genetic_dispatch_baseline_accepts_loss_only_and_paper_aligned(tmp_path: Path):
    output_dir = tmp_path / "ga_loss_only_outputs"
    root = Path(__file__).resolve().parents[2]
    command = [
        sys.executable,
        str(root / "scripts" / "analysis" / "genetic_dispatch_baseline.py"),
        "--cases",
        "ieee33",
        "--regimes",
        "network_stress",
        "--battery-models",
        "thevenin_loss_only",
        "--reward-profile",
        "paper_aligned",
        "--days",
        "1",
        "--seed",
        "42",
        "--population-size",
        "4",
        "--generations",
        "1",
        "--elite-count",
        "1",
        "--mutation-scale",
        "0.05",
        "--output-dir",
        str(output_dir),
    ]
    subprocess.run(command, cwd=root, capture_output=True, text=True, check=True)

    summary_df = pd.read_csv(output_dir / "summary.csv")
    assert len(summary_df) == 1
    assert summary_df.loc[0, "battery_model"] == "thevenin_loss_only"
    assert summary_df.loc[0, "reward_profile"] == "paper_aligned"


def test_genetic_dispatch_baseline_accepts_none_baseline(tmp_path: Path):
    output_dir = tmp_path / "ga_none_outputs"
    root = Path(__file__).resolve().parents[2]
    command = [
        sys.executable,
        str(root / "scripts" / "analysis" / "genetic_dispatch_baseline.py"),
        "--cases",
        "ieee33",
        "--regimes",
        "network_stress",
        "--battery-models",
        "none",
        "--reward-profile",
        "paper_balanced",
        "--days",
        "1",
        "--seed",
        "42",
        "--population-size",
        "4",
        "--generations",
        "1",
        "--elite-count",
        "1",
        "--mutation-scale",
        "0.05",
        "--output-dir",
        str(output_dir),
    ]
    subprocess.run(command, cwd=root, capture_output=True, text=True, check=True)

    summary_df = pd.read_csv(output_dir / "summary.csv")
    assert len(summary_df) == 1
    assert summary_df.loc[0, "battery_model"] == "none"
    assert float(summary_df.loc[0, "soc_violation_total"]) == 0.0
