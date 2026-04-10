from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_fidelity_regime_map_generates_summary(tmp_path: Path):
    output_dir = tmp_path / "regime_map_outputs"
    root = Path(__file__).resolve().parents[2]
    command = [
        sys.executable,
        str(root / "scripts" / "analysis" / "fidelity_regime_map.py"),
        "--cases",
        "ieee33",
        "--regimes",
        "base,high_load",
        "--train-models",
        "simple",
        "--test-models",
        "simple,thevenin",
        "--agent",
        "sac",
        "--train-steps",
        "1",
        "--eval-steps",
        "2",
        "--days",
        "1",
        "--seed",
        "42",
        "--output-dir",
        str(output_dir),
    ]
    completed = subprocess.run(command, cwd=root, capture_output=True, text=True, check=True)
    assert "Fidelity Regime Gap Summary" in completed.stdout

    detail_csv = output_dir / "detail.csv"
    summary_csv = output_dir / "summary.csv"
    summary_json = output_dir / "summary.json"

    assert detail_csv.exists()
    assert summary_csv.exists()
    assert summary_json.exists()

    summary_df = pd.read_csv(summary_csv)
    assert not summary_df.empty
    assert list(summary_df.columns) == [
        "case",
        "regime",
        "train_model",
        "final_cumulative_cost_gap",
        "min_voltage_worst_gap",
        "max_line_loading_peak_gap",
        "max_line_current_peak_ka_gap",
        "mean_grid_import_mw_gap",
        "total_battery_loss_kwh_gap",
        "total_battery_stress_kwh_gap",
        "total_battery_throughput_kwh_gap",
        "soc_upper_dwell_fraction_gap",
        "soc_lower_dwell_fraction_gap",
        "infeasible_action_dwell_fraction_gap",
        "power_flow_failure_steps_gap",
    ]

    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    assert payload["action_regularization"] == {
        "smoothing_coef": 0.0,
        "max_delta": 0.0,
        "rate_penalty": 0.0,
        "battery_feasibility_aware": False,
        "battery_infeasible_penalty": 0.0,
        "symmetric_battery_action": False,
    }
    assert len(payload["summary"]) == 2
    assert {item["regime"] for item in payload["summary"]} == {"base", "high_load"}


def test_fidelity_regime_map_accepts_loss_only_and_paper_aligned(tmp_path: Path):
    output_dir = tmp_path / "regime_map_loss_only_outputs"
    root = Path(__file__).resolve().parents[2]
    command = [
        sys.executable,
        str(root / "scripts" / "analysis" / "fidelity_regime_map.py"),
        "--cases",
        "ieee33",
        "--regimes",
        "network_stress",
        "--train-models",
        "thevenin_loss_only",
        "--test-models",
        "simple,thevenin_loss_only",
        "--reward-profile",
        "paper_aligned",
        "--agent",
        "sac",
        "--train-steps",
        "1",
        "--eval-steps",
        "2",
        "--days",
        "1",
        "--seed",
        "42",
        "--output-dir",
        str(output_dir),
    ]
    subprocess.run(command, cwd=root, capture_output=True, text=True, check=True)
    summary_df = pd.read_csv(output_dir / "summary.csv")
    assert not summary_df.empty
    assert summary_df.loc[0, "train_model"] == "thevenin_loss_only"
