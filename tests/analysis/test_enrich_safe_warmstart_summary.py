from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_enrich_safe_warmstart_summary_overwrites_existing_files(tmp_path: Path):
    result_dir = tmp_path / "safe_compare_outputs"
    result_dir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    oracle_reference_path = tmp_path / "oracle_protocol_summary.csv"

    pd.DataFrame(
        [
            {
                "case": "ieee33",
                "regime": "network_stress",
                "controller_variant": "plain_sac",
                "train_model": "simple",
                "test_model": "simple",
                "seed": 42,
                "eval_window_label": "30d_off0",
                "eval_window_family": "30d",
                "final_cumulative_cost": 910.0,
                "final_cumulative_objective_cost": 900.0,
                "final_soc": 0.55,
                "shield_enabled": 0,
            }
        ]
    ).to_csv(result_dir / "summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "case": "ieee33",
                "regime": "network_stress",
                "controller_variant": "plain_sac",
                "train_model": "simple",
                "test_model": "simple",
                "mean_final_cumulative_cost": 910.0,
                "row_count": 1,
            }
        ]
    ).to_csv(result_dir / "summary_grouped.csv", index=False)
    pd.DataFrame(
        [
            {
                "protocol": "full_year_365d_lp_oracle",
                "case": "IEEE 33-bus",
                "case_key": "ieee33_network",
                "regime": "network_stress",
                "battery_model": "simple",
                "none_objective": 1000.0,
                "oracle_objective": 800.0,
            }
        ]
    ).to_csv(oracle_reference_path, index=False)

    command = [
        sys.executable,
        str(root / "scripts" / "analysis" / "enrich_safe_warmstart_summary.py"),
        str(result_dir),
        "--oracle-reference-csv",
        str(oracle_reference_path),
        "--overwrite-existing",
    ]
    completed = subprocess.run(command, cwd=root, capture_output=True, text=True, check=True)
    assert "[enrich]" in completed.stdout

    summary_df = pd.read_csv(result_dir / "summary.csv")
    grouped_df = pd.read_csv(result_dir / "summary_grouped.csv")
    summary_payload = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    grouped_payload = json.loads((result_dir / "summary_grouped.json").read_text(encoding="utf-8"))
    reviewer_value_df = pd.read_csv(result_dir / "reviewer_value_recovery_grouped.csv")
    reviewer_behavior_df = pd.read_csv(result_dir / "reviewer_battery_behavior_grouped.csv")
    paper_value_df = pd.read_csv(result_dir / "paper_main_value_recovery.csv")
    paper_behavior_df = pd.read_csv(result_dir / "paper_main_battery_behavior.csv")
    storyline_audit_df = pd.read_csv(result_dir / "storyline_audit.csv")

    assert summary_df.columns[:8].tolist() == [
        "case",
        "regime",
        "controller_variant",
        "train_model",
        "test_model",
        "seed",
        "eval_window_label",
        "eval_window_family",
    ]
    row = summary_df.loc[0]
    assert float(row["objective_none_cost"]) == 1000.0
    assert float(row["objective_oracle_cost"]) == 800.0
    assert float(row["objective_recovery_fraction_vs_oracle"]) == 0.5
    assert abs(float(row["objective_gap_to_oracle"]) - 100.0) < 1e-9
    assert summary_payload[0]["oracle_reference_csv"] == str(oracle_reference_path)

    grouped_row = grouped_df.loc[0]
    assert float(grouped_row["mean_objective_recovery_fraction_vs_oracle"]) == 0.5
    assert float(grouped_row["mean_objective_gap_to_oracle"]) == 100.0
    assert grouped_payload[0]["row_count"] == 1
    assert "mean_objective_recovery_fraction_vs_oracle" in reviewer_value_df.columns
    assert "mean_final_soc" in reviewer_behavior_df.columns
    assert "eval_window_family" in paper_value_df.columns
    assert "eval_window_family" in paper_behavior_df.columns
    assert "main_story_verdict" in storyline_audit_df.columns


def test_enrich_safe_warmstart_summary_blocks_mismatched_reference_windows(tmp_path: Path):
    result_dir = tmp_path / "safe_compare_outputs"
    result_dir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    oracle_reference_path = tmp_path / "oracle_protocol_summary.csv"

    pd.DataFrame(
        [
            {
                "case": "ieee33",
                "regime": "network_stress",
                "controller_variant": "plain_sac",
                "train_model": "simple",
                "test_model": "simple",
                "seed": 42,
                "eval_window_label": "30d_off0",
                "eval_window_family": "30d",
                "eval_window_start": "2024-01-01 00:00:00",
                "eval_window_end": "2024-01-30 23:45:00",
                "final_cumulative_cost": 910.0,
                "final_cumulative_objective_cost": 900.0,
                "final_soc": 0.55,
            }
        ]
    ).to_csv(result_dir / "summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "protocol": "full_year_365d_lp_oracle",
                "case": "IEEE 33-bus",
                "case_key": "ieee33_network",
                "regime": "network_stress",
                "battery_model": "simple",
                "none_objective": 1000.0,
                "oracle_objective": 800.0,
                "profile_start_timestamp": "2023-01-01T00:00:00.000000000",
                "profile_end_timestamp": "2023-12-31T23:45:00.000000000",
            }
        ]
    ).to_csv(oracle_reference_path, index=False)

    subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "analysis" / "enrich_safe_warmstart_summary.py"),
            str(result_dir),
            "--oracle-reference-csv",
            str(oracle_reference_path),
            "--overwrite-existing",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )

    row = pd.read_csv(result_dir / "summary.csv").loc[0]
    assert int(row["oracle_reference_window_compatible"]) == 0
    assert pd.isna(row["objective_none_cost"])
    assert pd.isna(row["objective_recovery_fraction_vs_oracle"])
