from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_compare_safe_warmstart_sac_generates_variant_summary(tmp_path: Path):
    output_dir = tmp_path / "safe_compare_outputs"
    dataset_path = tmp_path / "offline_dataset.csv"
    oracle_reference_path = tmp_path / "oracle_protocol_summary.csv"
    root = Path(__file__).resolve().parents[2]

    obs0 = json.dumps([0.0] * 36)
    obs1 = json.dumps([0.05] * 36)
    obs2 = json.dumps([0.1] * 36)
    pd.DataFrame(
        [
            {
                "case": "ieee33",
                "regime": "network_stress",
                "battery_model": "simple",
                "controller_source": "oracle",
                "step": 0,
                "obs": obs0,
                "next_obs": obs1,
                "action": 0.1,
                "reward": 1.0,
                "done": 0,
            },
            {
                "case": "ieee33",
                "regime": "network_stress",
                "battery_model": "simple",
                "controller_source": "oracle",
                "step": 1,
                "obs": obs1,
                "next_obs": obs2,
                "action": 0.15,
                "reward": 0.5,
                "done": 1,
            },
        ]
    ).to_csv(dataset_path, index=False)
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
        str(root / "scripts" / "analysis" / "compare_safe_warmstart_sac.py"),
        "--cases",
        "ieee33",
        "--regimes",
        "network_stress",
        "--train-models",
        "simple",
        "--test-models",
        "simple",
        "--controller-variants",
        "plain_sac,replay_warmstart_sac,bc_warmstart_sac,shielded_sac",
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
        "--offline-dataset",
        str(dataset_path),
        "--offline-dataset-controller-sources",
        "oracle",
        "--offline-dataset-max-transitions",
        "1",
        "--bc-pretrain-gradient-steps",
        "1",
        "--bc-pretrain-batch-size",
        "1",
        "--shield-hard-pullback-action",
        "0.2",
        "--oracle-reference-csv",
        str(oracle_reference_path),
        "--output-dir",
        str(output_dir),
    ]
    completed = subprocess.run(command, cwd=root, capture_output=True, text=True, check=True)
    assert "Safe / Warmstart Comparison Summary" in completed.stdout

    summary_df = pd.read_csv(output_dir / "summary.csv")
    grouped_df = pd.read_csv(output_dir / "summary_grouped.csv")
    reviewer_value_df = pd.read_csv(output_dir / "reviewer_value_recovery_grouped.csv")
    reviewer_behavior_df = pd.read_csv(output_dir / "reviewer_battery_behavior_grouped.csv")
    paper_value_df = pd.read_csv(output_dir / "paper_main_value_recovery.csv")
    paper_behavior_df = pd.read_csv(output_dir / "paper_main_battery_behavior.csv")
    storyline_audit_df = pd.read_csv(output_dir / "storyline_audit.csv")
    assert set(summary_df["controller_variant"]) == {"plain_sac", "replay_warmstart_sac", "bc_warmstart_sac", "shielded_sac"}
    assert {
        "case",
        "controller_variant",
        "train_model",
        "test_model",
        "eval_window_label",
        "eval_window_family",
        "final_cumulative_cost",
        "objective_none_cost",
        "objective_oracle_cost",
        "objective_recovery_fraction_vs_oracle",
        "oracle_normalized_objective_value_recovery",
        "objective_gap_to_oracle",
        "soc_midband_dwell_fraction",
        "soc_target_tracking_mae",
        "peak_price_discharge_action_fraction",
        "valley_price_charge_action_fraction",
        "inventory_teacher_activation_fraction",
        "train_validation_metric",
    }.issubset(summary_df.columns)
    assert {
        "mean_final_cumulative_cost",
        "mean_objective_none_cost",
        "mean_objective_oracle_cost",
        "mean_objective_recovery_fraction_vs_oracle",
        "mean_oracle_normalized_objective_value_recovery",
        "mean_objective_gap_to_oracle",
        "mean_soc_midband_dwell_fraction",
        "mean_soc_target_tracking_mae",
        "mean_peak_price_discharge_action_fraction",
        "mean_valley_price_charge_action_fraction",
        "mean_inventory_teacher_activation_fraction",
        "row_count",
    }.issubset(grouped_df.columns)
    assert "mean_objective_recovery_fraction_vs_oracle" in reviewer_value_df.columns
    assert "mean_soc_midband_dwell_fraction" in reviewer_behavior_df.columns
    assert "eval_window_family" in paper_value_df.columns
    assert "eval_window_family" in paper_behavior_df.columns
    assert {
        "controller_variant",
        "eval_window_family",
        "inventory_first_checkpoint",
        "value_recovery_status",
        "inventory_health_status",
        "protocol_dependence_status",
        "main_story_verdict",
    }.issubset(storyline_audit_df.columns)

    bc_row = summary_df.loc[summary_df["controller_variant"] == "bc_warmstart_sac"].iloc[0]
    assert bc_row["offline_dataset"] == str(dataset_path)
    assert int(bc_row["offline_bc_replay_seeded_transitions"]) == 1

    shield_row = summary_df.loc[summary_df["controller_variant"] == "shielded_sac"].iloc[0]
    assert int(shield_row["shield_enabled"]) == 1

    plain_row = summary_df.loc[summary_df["controller_variant"] == "plain_sac"].iloc[0]
    assert int(plain_row["shield_enabled"]) == 0
    assert plain_row["oracle_reference_csv"] == str(oracle_reference_path)
    assert float(plain_row["objective_none_cost"]) == 1000.0
    assert float(plain_row["objective_oracle_cost"]) == 800.0
    assert abs(
        float(plain_row["objective_savings_vs_none"])
        - (1000.0 - float(plain_row["final_cumulative_objective_cost"]))
    ) < 1e-9
    assert abs(
        float(plain_row["objective_oracle_savings_vs_none"])
        - (1000.0 - 800.0)
    ) < 1e-9
    assert abs(
        float(plain_row["objective_recovery_fraction_vs_oracle"])
        - (
            float(plain_row["objective_savings_vs_none"])
            / float(plain_row["objective_oracle_savings_vs_none"])
        )
    ) < 1e-9
    assert abs(
        float(plain_row["oracle_normalized_objective_value_recovery"])
        - float(plain_row["objective_recovery_fraction_vs_oracle"])
    ) < 1e-9
    assert abs(
        float(plain_row["objective_gap_to_oracle"])
        - (float(plain_row["final_cumulative_objective_cost"]) - 800.0)
    ) < 1e-9

    replay_row = summary_df.loc[summary_df["controller_variant"] == "replay_warmstart_sac"].iloc[0]
    assert replay_row["offline_dataset"] == str(dataset_path)
    assert int(replay_row["offline_bc_actor_gradient_steps"]) == 0

    grouped_plain_row = grouped_df.loc[grouped_df["controller_variant"] == "plain_sac"].iloc[0]
    assert abs(
        float(grouped_plain_row["mean_objective_recovery_fraction_vs_oracle"])
        - float(plain_row["objective_recovery_fraction_vs_oracle"])
    ) < 1e-9
    assert abs(
        float(grouped_plain_row["mean_oracle_normalized_objective_value_recovery"])
        - float(plain_row["oracle_normalized_objective_value_recovery"])
    ) < 1e-9

    trajectory_files = list((output_dir / "trajectories").rglob("*.csv"))
    assert len(trajectory_files) == 4


def test_compare_safe_warmstart_sac_supports_multiwindow_eval(tmp_path: Path):
    output_dir = tmp_path / "safe_compare_outputs"
    root = Path(__file__).resolve().parents[2]
    oracle_reference_path = tmp_path / "oracle_reference_windows.csv"

    pd.DataFrame(
        [
            {
                "case": "IEEE 33-bus",
                "case_key": "ieee33_network",
                "regime": "network_stress",
                "battery_model": "simple",
                "reference_window_days": 1,
                "reference_offset_days_within_year": 0,
                "none_objective": 100.0,
                "oracle_objective": 90.0,
                "profile_start_timestamp": "2024-01-01T00:00:00.000000000",
                "profile_end_timestamp": "2024-01-01T23:45:00.000000000",
            },
            {
                "case": "IEEE 33-bus",
                "case_key": "ieee33_network",
                "regime": "network_stress",
                "battery_model": "simple",
                "reference_window_days": 1,
                "reference_offset_days_within_year": 1,
                "none_objective": 110.0,
                "oracle_objective": 95.0,
                "profile_start_timestamp": "2024-01-02T00:00:00.000000000",
                "profile_end_timestamp": "2024-01-02T23:45:00.000000000",
            },
        ]
    ).to_csv(oracle_reference_path, index=False)

    command = [
        sys.executable,
        str(root / "scripts" / "analysis" / "compare_safe_warmstart_sac.py"),
        "--cases",
        "ieee33",
        "--regimes",
        "network_stress",
        "--train-models",
        "simple",
        "--test-models",
        "simple",
        "--controller-variants",
        "plain_sac",
        "--agent",
        "sac",
        "--train-steps",
        "1",
        "--days",
        "1",
        "--seed",
        "42",
        "--eval-year",
        "2024",
        "--eval-window-days-list",
        "1",
        "--eval-offset-days-list",
        "0,1",
        "--oracle-reference-csv",
        str(oracle_reference_path),
        "--output-dir",
        str(output_dir),
    ]
    subprocess.run(command, cwd=root, capture_output=True, text=True, check=True)

    summary_df = pd.read_csv(output_dir / "summary.csv")
    paper_value_df = pd.read_csv(output_dir / "paper_main_value_recovery.csv")
    storyline_audit_df = pd.read_csv(output_dir / "storyline_audit.csv")
    assert len(summary_df) == 2
    assert set(summary_df["eval_window_label"]) == {"1d_off0", "1d_off1"}
    assert set(summary_df["oracle_reference_window_compatible"]) == {1}
    assert len(paper_value_df) == 1
    assert paper_value_df.loc[0, "eval_window_family"] == "1d"
    assert int(paper_value_df.loc[0, "window_count"]) == 2
    assert len(storyline_audit_df) == 1
    assert storyline_audit_df.loc[0, "window_coverage_status"] == "multiwindow_partial"
