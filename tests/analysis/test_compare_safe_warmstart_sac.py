from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.compare_safe_warmstart_sac import (
    _attach_oracle_value_recovery,
    _artifact_stem,
    _preferred_oracle_reference_csv,
    _variant_args,
    attach_reviewer_pass_fail_summary,
    build_paper_main_value_recovery,
    build_storyline_audit,
)


def test_artifact_stem_omits_variant_to_avoid_windows_path_growth():
    stem = _artifact_stem(
        case_key="ieee33",
        regime="network_stress",
        agent="sac",
        train_model="simple",
        test_model="simple",
        eval_window_label="30d_off0",
        seed=42,
    )

    assert "shielded_replay_warmstart_sac" not in stem
    assert stem == "ieee33_network_stress_sac_tr-simple_te-simple_30d_off0_s42"
    assert len(stem) < 80


def test_preferred_oracle_reference_uses_network_replayed_sibling(tmp_path: Path):
    lossless = tmp_path / "oracle_reference_windows.csv"
    network = tmp_path / "network_replayed_oracle_reference_windows.csv"
    lossless.write_text("metric,none_cost,oracle_cost\n", encoding="utf-8")
    network.write_text("metric,none_cost,oracle_cost\n", encoding="utf-8")

    assert _preferred_oracle_reference_csv(lossless) == str(network)


def test_preferred_oracle_reference_explicit_network_path(tmp_path: Path):
    lossless = tmp_path / "oracle_reference_windows.csv"
    explicit = tmp_path / "custom_network_reference.csv"
    lossless.write_text("metric,none_cost,oracle_cost\n", encoding="utf-8")
    explicit.write_text("metric,none_cost,oracle_cost\n", encoding="utf-8")

    assert _preferred_oracle_reference_csv(lossless, explicit) == str(explicit)


def test_variant_args_cleanly_disable_unowned_protocol_components(tmp_path: Path):
    base = argparse.Namespace(
        output_dir=str(tmp_path),
        tb_log_name="suite",
        offline_dataset="teacher.csv",
        offline_dataset_controller_sources="heuristic",
        offline_dataset_max_transitions=4000,
        bc_pretrain_gradient_steps=16,
        online_safe_bc_gradient_steps=32,
        online_safe_bc_max_samples=4000,
        shield_enabled=True,
        train_steps=100,
        shield_delta_penalty_coef=0.0,
        shield_delta_penalty_start=-1.0,
        shield_delta_penalty_end=-1.0,
        shield_delta_penalty_warmup_steps=0,
    )

    plain = _variant_args(base, controller_variant="plain_sac", variant_output_dir=tmp_path / "plain")
    assert plain.offline_dataset == ""
    assert plain.offline_dataset_controller_sources == ""
    assert plain.offline_dataset_max_transitions == 0
    assert plain.bc_pretrain_gradient_steps == 0
    assert plain.online_safe_bc_gradient_steps == 0
    assert plain.online_safe_bc_max_samples == 0
    assert plain.shield_enabled is False

    shielded = _variant_args(base, controller_variant="shielded_sac", variant_output_dir=tmp_path / "shielded")
    assert shielded.offline_dataset == ""
    assert shielded.bc_pretrain_gradient_steps == 0
    assert shielded.online_safe_bc_gradient_steps == 0
    assert shielded.shield_enabled is True

    replay = _variant_args(base, controller_variant="replay_warmstart_sac", variant_output_dir=tmp_path / "replay")
    assert replay.offline_dataset == "teacher.csv"
    assert replay.bc_pretrain_gradient_steps == 0
    assert replay.online_safe_bc_gradient_steps == 0
    assert replay.shield_enabled is False

    bc = _variant_args(base, controller_variant="bc_warmstart_sac", variant_output_dir=tmp_path / "bc")
    assert bc.offline_dataset == "teacher.csv"
    assert bc.bc_pretrain_gradient_steps == 16
    assert bc.online_safe_bc_gradient_steps == 0
    assert bc.shield_enabled is False

    sail = _variant_args(
        base,
        controller_variant="shielded_replay_warmstart_sac",
        variant_output_dir=tmp_path / "sail",
    )
    assert sail.offline_dataset == "teacher.csv"
    assert sail.bc_pretrain_gradient_steps == 0
    assert sail.online_safe_bc_gradient_steps == 32
    assert sail.online_safe_bc_max_samples == 4000
    assert sail.shield_enabled is True

    shielded_bc = _variant_args(
        base,
        controller_variant="shielded_bc_warmstart_sac",
        variant_output_dir=tmp_path / "shielded_bc",
    )
    assert shielded_bc.offline_dataset == "teacher.csv"
    assert shielded_bc.bc_pretrain_gradient_steps == 16
    assert shielded_bc.online_safe_bc_gradient_steps == 0
    assert shielded_bc.shield_enabled is True

    sda = _variant_args(
        base,
        controller_variant="shield_dependence_aware_sac",
        variant_output_dir=tmp_path / "sda",
    )
    assert sda.offline_dataset == "teacher.csv"
    assert sda.bc_pretrain_gradient_steps == 0
    assert sda.shield_enabled is True
    assert sda.shield_delta_penalty_start == 0.0
    assert sda.shield_delta_penalty_end == 1.0
    assert sda.shield_delta_penalty_warmup_steps == 50
    assert sda.online_safe_bc_gradient_steps == 32
    assert sda.online_safe_bc_max_samples == 4000


def test_attach_reviewer_pass_fail_summary_marks_value_and_morphology():
    summary_df = pd.DataFrame(
        [
            {
                "objective_recovery_fraction_vs_oracle": 0.1,
                "soc_midband_dwell_fraction": 0.08,
                "peak_price_discharge_action_fraction": 0.25,
                "valley_price_charge_action_fraction": 0.22,
                "mean_abs_shield_delta": 0.02,
                "shield_material_activation_fraction": 0.40,
                "shield_strong_activation_fraction": 0.10,
            },
            {
                "objective_recovery_fraction_vs_oracle": -0.1,
                "soc_midband_dwell_fraction": 0.01,
                "peak_price_discharge_action_fraction": 0.05,
                "valley_price_charge_action_fraction": 0.02,
                "mean_abs_shield_delta": 0.20,
                "shield_material_activation_fraction": 0.95,
                "shield_strong_activation_fraction": 0.90,
            },
        ]
    )

    enriched = attach_reviewer_pass_fail_summary(summary_df)

    assert int(enriched.loc[0, "value_recovery_pass"]) == 1
    assert int(enriched.loc[0, "morphology_behavior_pass"]) == 1
    assert int(enriched.loc[0, "shield_internalization_pass"]) == 1
    assert int(enriched.loc[0, "reviewer_ready_pass"]) == 1
    assert int(enriched.loc[1, "value_recovery_pass"]) == 0
    assert int(enriched.loc[1, "morphology_pass"]) == 0
    assert int(enriched.loc[1, "reviewer_ready_pass"]) == 0


def test_oracle_recovery_keeps_raw_index_and_clips_display_value(tmp_path: Path):
    reference_path = tmp_path / "oracle_reference.csv"
    pd.DataFrame(
        [
            {
                "case": "IEEE 33-bus",
                "case_key": "ieee33_network",
                "regime": "network_stress",
                "battery_model": "simple",
                "none_objective": 100.0,
                "oracle_objective": 80.0,
            }
        ]
    ).to_csv(reference_path, index=False)
    summary_df = pd.DataFrame(
        [
            {
                "case": "ieee33",
                "regime": "network_stress",
                "controller_variant": "shielded_replay_warmstart_sac",
                "train_model": "simple",
                "test_model": "simple",
                "eval_window_label": "30d_off0",
                "eval_window_family": "30d",
                "final_cumulative_objective_cost": 75.0,
            }
        ]
    )

    enriched = _attach_oracle_value_recovery(summary_df, reference_csv=str(reference_path))

    assert float(enriched.loc[0, "objective_recovery_fraction_vs_oracle_raw"]) == 1.25
    assert float(enriched.loc[0, "objective_recovery_fraction_vs_oracle"]) == 1.25
    assert float(enriched.loc[0, "objective_recovery_fraction_vs_oracle_display"]) == 1.0
    assert float(enriched.loc[0, "oracle_normalized_objective_value_recovery_raw"]) == 1.25
    assert float(enriched.loc[0, "oracle_normalized_objective_value_recovery_display"]) == 1.0

    paper_value_df = build_paper_main_value_recovery(enriched)
    assert float(paper_value_df.loc[0, "mean_objective_recovery_fraction_vs_oracle_display"]) == 1.0
    assert float(paper_value_df.loc[0, "mean_objective_recovery_fraction_vs_oracle_raw"]) == 1.25


def test_storyline_audit_marks_p4_like_morphology_as_improved_but_fragile():
    summary_df = pd.DataFrame(
        [
            {
                "case": "ieee33",
                "regime": "network_stress",
                "controller_variant": "shielded_replay_warmstart_sac",
                "train_model": "simple",
                "test_model": "simple",
                "eval_window_label": "30d_off0",
                "eval_window_family": "30d",
                "train_validation_metric": "inventory_value_gate_shield",
                "oracle_normalized_objective_value_recovery": 0.401,
                "objective_recovery_fraction_vs_oracle": 0.401,
                "final_cumulative_objective_cost": 90.0,
                "objective_none_cost": 100.0,
                "objective_oracle_cost": 80.0,
                "objective_gap_to_oracle": 10.0,
                "final_soc": 0.52,
                "total_terminal_soc_penalty": 0.02,
                "soc_midband_dwell_fraction": 0.301,
                "soc_target_tracking_mae": 0.12,
                "soc_upper_parking_fraction": 0.0,
                "soc_lower_parking_fraction": 0.0,
                "peak_price_mean_discharge_limit_ratio": 0.50,
                "peak_price_discharge_action_fraction": 0.768,
                "valley_price_charge_action_fraction": 0.498,
                "morphology_behavior_pass": 0.916,
                "infeasible_action_dwell_fraction": 0.0,
                "mean_abs_shield_delta": 0.048,
                "shield_material_activation_fraction": 0.429,
                "shield_strong_activation_fraction": 0.229,
            }
        ]
    )

    audit_df = build_storyline_audit(summary_df)

    assert audit_df.loc[0, "inventory_health_status"] == "improved_but_fragile"
    assert audit_df.loc[0, "main_story_verdict"] == "battery_story_supported"


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
