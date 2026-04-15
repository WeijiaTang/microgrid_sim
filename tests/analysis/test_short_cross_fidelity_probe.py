from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_short_cross_fidelity_probe_generates_summary_and_trajectory(tmp_path: Path):
    output_dir = tmp_path / "probe_outputs"
    root = Path(__file__).resolve().parents[2]
    command = [
        sys.executable,
        str(root / "scripts" / "analysis" / "short_cross_fidelity_probe.py"),
        "--cases",
        "ieee33",
        "--regimes",
        "high_load",
        "--train-models",
        "simple",
        "--test-models",
        "simple",
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
    assert "Short Cross-Fidelity Summary" in completed.stdout

    summary_csv = output_dir / "summary.csv"
    summary_json = output_dir / "summary.json"
    trajectories_dir = output_dir / "trajectories"

    assert summary_csv.exists()
    assert summary_json.exists()
    assert trajectories_dir.exists()

    summary_df = pd.read_csv(summary_csv)
    assert not summary_df.empty
    assert list(summary_df.columns) == [
        "case",
        "regime",
        "reward_profile",
        "agent",
        "seed",
        "train_model",
        "test_model",
        "train_steps",
        "eval_steps",
        "learning_rate",
        "tensorboard_log_dir",
        "tensorboard_run_name",
        "action_smoothing_coef",
        "action_max_delta",
        "action_rate_penalty",
        "battery_feasibility_aware",
        "battery_infeasible_penalty",
        "symmetric_battery_action",
        "rule_guidance_mix",
        "rule_guidance_decay_steps",
        "train_year",
        "eval_year",
        "train_episode_days",
        "eval_config_days",
        "train_window_start",
        "train_window_end",
        "eval_window_start",
        "eval_window_end",
        "train_random_start_within_year",
        "train_validation_days",
        "train_validation_offset_days_within_year",
        "train_validation_window_count",
        "train_validation_checkpoint_every",
        "train_validation_metric",
        "train_validation_terminal_penalty_weight",
        "train_validation_boundary_dwell_weight",
        "train_validation_infeasible_dwell_weight",
        "validation_best_metric_value",
        "validation_best_total_reward",
        "validation_best_objective_cost",
        "validation_best_checkpoint_step",
        "causal_heuristic_warmstart_steps",
        "causal_heuristic_warmstart_policy",
        "causal_heuristic_warmstart_steps_applied",
        "eval_full_horizon",
        "mixed_fidelity_stage_fractions",
        "mixed_fidelity_stage_learning_rates",
        "resolved_train_stages",
        "resolved_train_stage_count",
        "resolved_train_stage_fractions",
        "resolved_train_stage_steps",
        "resolved_train_stage_learning_rates",
        "steps",
        "total_reward",
        "final_soc",
        "final_temperature_c",
        "final_cumulative_cost",
        "final_cumulative_objective_cost",
        "final_terminal_soc_deviation",
        "total_terminal_soc_penalty",
        "min_voltage_worst",
        "max_line_loading_peak",
        "max_line_current_peak_ka",
        "mean_grid_import_mw",
        "total_battery_loss_kwh",
        "total_battery_stress_kwh",
        "total_battery_throughput_kwh",
        "mean_abs_battery_action_delta",
        "total_action_rate_penalty",
        "mean_battery_action_infeasible_gap",
        "total_battery_action_infeasible_penalty",
        "soc_upper_dwell_fraction",
        "soc_lower_dwell_fraction",
        "infeasible_action_dwell_fraction",
        "power_flow_failure_steps",
    ]

    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    assert len(payload) == 1
    assert payload[0]["case"] == "ieee33"
    assert payload[0]["regime"] == "high_load"
    assert payload[0]["train_model"] == "simple"
    assert payload[0]["test_model"] == "simple"
    assert float(payload[0]["learning_rate"]) == 3e-4

    trajectories = list(trajectories_dir.glob("*.csv"))
    assert len(trajectories) >= 1


def test_short_cross_fidelity_probe_supports_year_split_windows(tmp_path: Path):
    output_dir = tmp_path / "probe_year_split_outputs"
    root = Path(__file__).resolve().parents[2]
    command = [
        sys.executable,
        str(root / "scripts" / "analysis" / "short_cross_fidelity_probe.py"),
        "--cases",
        "ieee33",
        "--regimes",
        "network_stress",
        "--train-models",
        "simple",
        "--test-models",
        "simple",
        "--agent",
        "sac",
        "--train-steps",
        "1",
        "--eval-steps",
        "2",
        "--days",
        "3",
        "--train-year",
        "2023",
        "--eval-year",
        "2024",
        "--train-episode-days",
        "7",
        "--eval-days",
        "1",
        "--train-random-start-within-year",
        "--output-dir",
        str(output_dir),
    ]
    subprocess.run(command, cwd=root, capture_output=True, text=True, check=True)

    summary_df = pd.read_csv(output_dir / "summary.csv")
    assert int(summary_df.loc[0, "train_year"]) == 2023
    assert int(summary_df.loc[0, "eval_year"]) == 2024
    assert int(summary_df.loc[0, "train_episode_days"]) == 7
    assert int(summary_df.loc[0, "eval_config_days"]) == 1
    assert int(summary_df.loc[0, "train_random_start_within_year"]) == 1
    assert str(summary_df.loc[0, "train_window_start"]).startswith("2023-01-01")
    assert str(summary_df.loc[0, "eval_window_start"]).startswith("2024-01-01")

    trajectory = pd.read_csv(next((output_dir / "trajectories").glob("*.csv")))
    assert str(trajectory.loc[0, "timestamp"]).startswith("2024-01-01")


def test_short_cross_fidelity_probe_exports_action_regularization_fields(tmp_path: Path):
    output_dir = tmp_path / "probe_regularized_outputs"
    root = Path(__file__).resolve().parents[2]
    command = [
        sys.executable,
        str(root / "scripts" / "analysis" / "short_cross_fidelity_probe.py"),
        "--cases",
        "ieee33",
        "--regimes",
        "network_stress",
        "--train-models",
        "simple",
        "--test-models",
        "thevenin",
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
        "--action-smoothing-coef",
        "0.5",
        "--action-max-delta",
        "0.2",
        "--action-rate-penalty",
        "0.5",
        "--battery-feasibility-aware",
        "--battery-infeasible-penalty",
        "-1.0",
        "--symmetric-battery-action",
        "--rule-guidance-mix",
        "0.6",
        "--rule-guidance-decay-steps",
        "10",
        "--output-dir",
        str(output_dir),
    ]
    subprocess.run(command, cwd=root, capture_output=True, text=True, check=True)

    summary_df = pd.read_csv(output_dir / "summary.csv")
    assert float(summary_df.loc[0, "action_smoothing_coef"]) == 0.5
    assert float(summary_df.loc[0, "action_max_delta"]) == 0.2
    assert float(summary_df.loc[0, "action_rate_penalty"]) == 0.5
    assert int(summary_df.loc[0, "battery_feasibility_aware"]) == 1
    assert float(summary_df.loc[0, "battery_infeasible_penalty"]) == -1.0
    assert int(summary_df.loc[0, "symmetric_battery_action"]) == 1
    assert float(summary_df.loc[0, "rule_guidance_mix"]) == 0.6
    assert int(summary_df.loc[0, "rule_guidance_decay_steps"]) == 10

    trajectory = pd.read_csv(next((output_dir / "trajectories").glob("*.csv")))
    for column in [
        "battery_action_raw",
        "battery_action_applied",
        "battery_action_delta",
        "action_rate_penalty",
        "policy_action_pre_guidance",
        "rule_based_action_hint",
        "rule_guided_action",
        "rule_guidance_mix",
        "action_after_rule_guidance",
        "battery_action_feasible_low",
        "battery_action_feasible_high",
        "battery_charge_fraction_feasible",
        "battery_discharge_fraction_feasible",
        "battery_action_infeasible_gap",
        "battery_action_infeasible_penalty",
        "soc_upper_bound_hit",
        "soc_lower_bound_hit",
        "battery_action_infeasible_flag",
    ]:
        assert column in trajectory.columns


def test_short_cross_fidelity_probe_uses_negative_default_infeasible_penalty_when_feasibility_aware(tmp_path: Path):
    output_dir = tmp_path / "probe_feasibility_default_outputs"
    root = Path(__file__).resolve().parents[2]
    command = [
        sys.executable,
        str(root / "scripts" / "analysis" / "short_cross_fidelity_probe.py"),
        "--cases",
        "ieee33",
        "--regimes",
        "network_stress",
        "--train-models",
        "simple",
        "--test-models",
        "simple",
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
        "--battery-feasibility-aware",
        "--output-dir",
        str(output_dir),
    ]
    subprocess.run(command, cwd=root, capture_output=True, text=True, check=True)

    summary_df = pd.read_csv(output_dir / "summary.csv")
    assert int(summary_df.loc[0, "battery_feasibility_aware"]) == 1
    assert float(summary_df.loc[0, "battery_infeasible_penalty"]) == -1.0


def test_short_cross_fidelity_probe_exports_learning_rate_field(tmp_path: Path):
    output_dir = tmp_path / "probe_learning_rate_outputs"
    root = Path(__file__).resolve().parents[2]
    command = [
        sys.executable,
        str(root / "scripts" / "analysis" / "short_cross_fidelity_probe.py"),
        "--cases",
        "ieee33",
        "--regimes",
        "network_stress",
        "--train-models",
        "thevenin",
        "--test-models",
        "thevenin",
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
        "--learning-rate",
        "5e-5",
        "--output-dir",
        str(output_dir),
    ]
    subprocess.run(command, cwd=root, capture_output=True, text=True, check=True)

    summary_df = pd.read_csv(output_dir / "summary.csv")
    assert float(summary_df.loc[0, "learning_rate"]) == 5e-5


def test_short_cross_fidelity_probe_exports_validation_and_warmstart_fields(tmp_path: Path):
    output_dir = tmp_path / "probe_validation_warmstart_outputs"
    root = Path(__file__).resolve().parents[2]
    command = [
        sys.executable,
        str(root / "scripts" / "analysis" / "short_cross_fidelity_probe.py"),
        "--cases",
        "ieee33",
        "--regimes",
        "network_stress",
        "--train-models",
        "simple",
        "--test-models",
        "simple",
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
        "--train-year",
        "2023",
        "--train-validation-days",
        "1",
        "--train-validation-offset-days-within-year",
        "0",
        "--train-validation-checkpoint-every",
        "1",
        "--causal-heuristic-warmstart-steps",
        "1",
        "--causal-heuristic-warmstart-policy",
        "blended",
        "--output-dir",
        str(output_dir),
    ]
    subprocess.run(command, cwd=root, capture_output=True, text=True, check=True)

    summary_df = pd.read_csv(output_dir / "summary.csv")
    assert int(summary_df.loc[0, "train_validation_days"]) == 1
    assert str(summary_df.loc[0, "train_validation_offset_days_within_year"]) == "0"
    assert int(summary_df.loc[0, "train_validation_window_count"]) == 1
    assert int(summary_df.loc[0, "train_validation_checkpoint_every"]) == 1
    assert summary_df.loc[0, "train_validation_metric"] == "health_objective"
    assert float(summary_df.loc[0, "train_validation_terminal_penalty_weight"]) == 1.0
    assert float(summary_df.loc[0, "train_validation_boundary_dwell_weight"]) == 20000.0
    assert float(summary_df.loc[0, "train_validation_infeasible_dwell_weight"]) == 20000.0
    assert int(summary_df.loc[0, "validation_best_checkpoint_step"]) == 1
    assert int(summary_df.loc[0, "causal_heuristic_warmstart_steps"]) == 1
    assert summary_df.loc[0, "causal_heuristic_warmstart_policy"] == "blended"
    assert int(summary_df.loc[0, "causal_heuristic_warmstart_steps_applied"]) == 1


def test_short_cross_fidelity_probe_supports_mixed_fidelity_train_spec(tmp_path: Path):
    output_dir = tmp_path / "probe_mixed_fidelity_outputs"
    root = Path(__file__).resolve().parents[2]
    command = [
        sys.executable,
        str(root / "scripts" / "analysis" / "short_cross_fidelity_probe.py"),
        "--cases",
        "ieee33",
        "--regimes",
        "network_stress",
        "--train-models",
        "simple+thevenin",
        "--test-models",
        "thevenin",
        "--agent",
        "sac",
        "--train-steps",
        "2",
        "--eval-steps",
        "2",
        "--days",
        "1",
        "--seed",
        "42",
        "--mixed-fidelity-pretrain-fraction",
        "0.5",
        "--output-dir",
        str(output_dir),
    ]
    subprocess.run(command, cwd=root, capture_output=True, text=True, check=True)

    summary_df = pd.read_csv(output_dir / "summary.csv")
    assert len(summary_df) == 1
    assert summary_df.loc[0, "train_model"] == "simple+thevenin"
    assert summary_df.loc[0, "test_model"] == "thevenin"
    assert summary_df.loc[0, "resolved_train_stages"] == "simple,thevenin"
    assert int(summary_df.loc[0, "resolved_train_stage_count"]) == 2
    assert summary_df.loc[0, "resolved_train_stage_steps"] == "1,1"


def test_short_cross_fidelity_probe_supports_three_stage_mixed_fidelity_train_spec(tmp_path: Path):
    output_dir = tmp_path / "probe_three_stage_mixed_fidelity_outputs"
    root = Path(__file__).resolve().parents[2]
    command = [
        sys.executable,
        str(root / "scripts" / "analysis" / "short_cross_fidelity_probe.py"),
        "--cases",
        "ieee33",
        "--regimes",
        "network_stress",
        "--train-models",
        "simple+thevenin_loss_only+thevenin",
        "--test-models",
        "thevenin",
        "--agent",
        "sac",
        "--train-steps",
        "3",
        "--eval-steps",
        "2",
        "--days",
        "1",
        "--seed",
        "42",
        "--mixed-fidelity-stage-fractions",
        "0.34,0.33,0.33",
        "--output-dir",
        str(output_dir),
    ]
    subprocess.run(command, cwd=root, capture_output=True, text=True, check=True)

    summary_df = pd.read_csv(output_dir / "summary.csv")
    assert len(summary_df) == 1
    assert summary_df.loc[0, "train_model"] == "simple+thevenin_loss_only+thevenin"
    assert summary_df.loc[0, "test_model"] == "thevenin"
    assert summary_df.loc[0, "mixed_fidelity_stage_fractions"] == "0.34,0.33,0.33"
    assert summary_df.loc[0, "resolved_train_stage_fractions"] == "0.340000,0.330000,0.330000"
    assert summary_df.loc[0, "resolved_train_stage_steps"] == "1,1,1"


def test_short_cross_fidelity_probe_supports_stage_learning_rates(tmp_path: Path):
    output_dir = tmp_path / "probe_stage_learning_rate_outputs"
    root = Path(__file__).resolve().parents[2]
    command = [
        sys.executable,
        str(root / "scripts" / "analysis" / "short_cross_fidelity_probe.py"),
        "--cases",
        "ieee33",
        "--regimes",
        "network_stress",
        "--train-models",
        "thevenin_loss_only+thevenin",
        "--test-models",
        "thevenin",
        "--agent",
        "sac",
        "--train-steps",
        "3",
        "--eval-steps",
        "2",
        "--days",
        "1",
        "--seed",
        "42",
        "--mixed-fidelity-stage-fractions",
        "0.67,0.33",
        "--mixed-fidelity-stage-learning-rates",
        "3e-4,5e-5",
        "--output-dir",
        str(output_dir),
    ]
    subprocess.run(command, cwd=root, capture_output=True, text=True, check=True)

    summary_df = pd.read_csv(output_dir / "summary.csv")
    assert len(summary_df) == 1
    assert summary_df.loc[0, "mixed_fidelity_stage_fractions"] == "0.67,0.33"
    assert summary_df.loc[0, "mixed_fidelity_stage_learning_rates"] == "3e-4,5e-5"
    assert summary_df.loc[0, "resolved_train_stage_learning_rates"] == "0.0003,5e-05"
    assert summary_df.loc[0, "resolved_train_stage_steps"] == "2,1"


def test_short_cross_fidelity_probe_supports_loss_only_and_paper_aligned_reward(tmp_path: Path):
    output_dir = tmp_path / "probe_loss_only_outputs"
    root = Path(__file__).resolve().parents[2]
    command = [
        sys.executable,
        str(root / "scripts" / "analysis" / "short_cross_fidelity_probe.py"),
        "--cases",
        "ieee33",
        "--regimes",
        "network_stress",
        "--train-models",
        "thevenin_loss_only",
        "--test-models",
        "thevenin_loss_only",
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
    assert len(summary_df) == 1
    assert summary_df.loc[0, "train_model"] == "thevenin_loss_only"
    assert summary_df.loc[0, "test_model"] == "thevenin_loss_only"


def test_short_cross_fidelity_probe_supports_paper_balanced_reward(tmp_path: Path):
    output_dir = tmp_path / "probe_paper_balanced_outputs"
    root = Path(__file__).resolve().parents[2]
    command = [
        sys.executable,
        str(root / "scripts" / "analysis" / "short_cross_fidelity_probe.py"),
        "--cases",
        "ieee33",
        "--regimes",
        "network_stress",
        "--train-models",
        "thevenin",
        "--test-models",
        "thevenin",
        "--reward-profile",
        "paper_balanced",
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
    assert len(summary_df) == 1
    assert summary_df.loc[0, "train_model"] == "thevenin"
    assert summary_df.loc[0, "test_model"] == "thevenin"


def test_short_cross_fidelity_probe_accepts_none_baseline(tmp_path: Path):
    output_dir = tmp_path / "probe_none_outputs"
    root = Path(__file__).resolve().parents[2]
    command = [
        sys.executable,
        str(root / "scripts" / "analysis" / "short_cross_fidelity_probe.py"),
        "--cases",
        "ieee33",
        "--regimes",
        "network_stress",
        "--train-models",
        "none",
        "--test-models",
        "none",
        "--reward-profile",
        "paper_balanced",
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
    assert len(summary_df) == 1
    assert summary_df.loc[0, "train_model"] == "none"
    assert summary_df.loc[0, "test_model"] == "none"
    assert float(summary_df.loc[0, "total_battery_loss_kwh"]) == 0.0
    assert float(summary_df.loc[0, "total_battery_stress_kwh"]) == 0.0


def test_short_cross_fidelity_probe_exports_tensorboard_metadata_and_events(tmp_path: Path):
    output_dir = tmp_path / "probe_tb_outputs"
    tb_dir = tmp_path / "tensorboard"
    root = Path(__file__).resolve().parents[2]
    command = [
        sys.executable,
        str(root / "scripts" / "analysis" / "short_cross_fidelity_probe.py"),
        "--cases",
        "ieee33",
        "--regimes",
        "network_stress",
        "--train-models",
        "simple",
        "--test-models",
        "simple",
        "--agent",
        "ppo",
        "--train-steps",
        "8",
        "--eval-steps",
        "2",
        "--days",
        "1",
        "--seed",
        "42",
        "--tensorboard-log",
        str(tb_dir),
        "--tb-log-name",
        "unit_tb_probe",
        "--output-dir",
        str(output_dir),
    ]
    subprocess.run(command, cwd=root, capture_output=True, text=True, check=True)

    summary_df = pd.read_csv(output_dir / "summary.csv")
    assert len(summary_df) == 1
    assert summary_df.loc[0, "tensorboard_log_dir"] == str(tb_dir)
    assert summary_df.loc[0, "tensorboard_run_name"] == "unit_tb_probe"
    assert list(tb_dir.rglob("events.out.tfevents.*"))
