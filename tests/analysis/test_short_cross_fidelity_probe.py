from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.short_cross_fidelity_probe import (
    _causal_heuristic_action,
    _peak_price_reserve_metrics,
    _validation_metric_value,
    apply_ieee33_sac_default_protocol,
    build_parser,
    resolve_training_schedule,
)
from microgrid_sim.cases import IEEE33Config
from microgrid_sim.envs.network_microgrid import NetworkMicrogridEnv


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
        "train_validation_peak_reserve_weight",
        "train_validation_peak_discharge_limit_threshold",
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
        "peak_price_step_fraction",
        "peak_price_mean_soc",
        "peak_price_mean_discharge_limit_ratio",
        "peak_price_low_discharge_limit_dwell_fraction",
        "mean_battery_action_infeasible_gap",
        "mean_battery_internal_clip_gap_w",
        "total_battery_action_infeasible_penalty",
        "soc_upper_dwell_fraction",
        "soc_lower_dwell_fraction",
        "infeasible_action_dwell_fraction",
        "internal_clip_dwell_fraction",
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
        "battery_command_requested_w",
        "battery_command_applied_w",
        "battery_internal_clip_gap_w",
        "soc_upper_bound_hit",
        "soc_lower_bound_hit",
        "battery_action_infeasible_flag",
        "battery_internal_clip_flag",
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
    assert float(summary_df.loc[0, "train_validation_peak_reserve_weight"]) == 0.0
    assert float(summary_df.loc[0, "train_validation_peak_discharge_limit_threshold"]) == 0.25
    assert int(summary_df.loc[0, "validation_best_checkpoint_step"]) == 1
    assert int(summary_df.loc[0, "causal_heuristic_warmstart_steps"]) == 1
    assert summary_df.loc[0, "causal_heuristic_warmstart_policy"] == "blended"
    assert int(summary_df.loc[0, "causal_heuristic_warmstart_steps_applied"]) == 1


def test_causal_heuristic_action_respects_current_feasible_battery_bounds():
    env = NetworkMicrogridEnv(IEEE33Config(simulation_days=1, seed=42, battery_model="thevenin_full", regime="network_stress"))
    try:
        env.reset(seed=42)
        env.current_step = int(pd.Series(env._profiles.pv_w).astype(float).idxmax())
        env.battery.soc = float(env.config.battery_params.soc_max) - 1e-4
        min_command_w, _ = env.battery.power_command_bounds(dt=float(env.config.dt_seconds))
        charge_limit_w = max(float(-min_command_w), 0.0)
        assert charge_limit_w < float(env.config.battery_params.p_charge_max)

        charge_action = _causal_heuristic_action(env, "rule")
        assert float(charge_action.reshape(-1)[0]) == pytest.approx(-1.0, abs=1e-3)
    finally:
        env.close()


def test_short_cross_fidelity_probe_applies_ieee33_sac_default_validation_protocol_for_research_scale_runs():
    raw_argv = [
        "--cases",
        "ieee33",
        "--agent",
        "sac",
        "--train-steps",
        "5000",
    ]
    args = build_parser().parse_args(raw_argv)

    patched = apply_ieee33_sac_default_protocol(args, case_key="ieee33", raw_argv=raw_argv)

    assert bool(getattr(patched, "ieee33_sac_default_protocol_applied", False)) is True
    assert int(patched.days) == 30
    assert int(patched.train_year) == 2023
    assert int(patched.eval_year) == 2024
    assert int(patched.train_episode_days) == 30
    assert int(patched.eval_days) == 30
    assert bool(patched.train_random_start_within_year) is True
    assert bool(patched.eval_full_horizon) is True
    assert int(patched.train_validation_days) == 7
    assert str(patched.train_validation_offset_days_within_year) == "0,91,182,273"
    assert int(patched.train_validation_checkpoint_every) == 1000
    assert str(patched.train_validation_metric) == "health_objective"
    assert float(patched.train_validation_peak_reserve_weight) == 10000.0
    assert float(patched.train_validation_peak_discharge_limit_threshold) == 0.25


def test_short_cross_fidelity_probe_respects_explicit_ieee33_sac_protocol_overrides():
    raw_argv = [
        "--cases",
        "ieee33",
        "--agent",
        "sac",
        "--train-steps",
        "10000",
        "--days",
        "7",
        "--train-year",
        "2024",
        "--eval-year",
        "2024",
        "--train-episode-days",
        "14",
        "--eval-days",
        "14",
        "--train-validation-days",
        "14",
        "--train-validation-offset-days-within-year",
        "30,120",
        "--train-validation-checkpoint-every",
        "1000",
    ]
    args = build_parser().parse_args(raw_argv)

    patched = apply_ieee33_sac_default_protocol(args, case_key="ieee33", raw_argv=raw_argv)

    assert bool(getattr(patched, "ieee33_sac_default_protocol_applied", False)) is True
    assert int(patched.days) == 7
    assert int(patched.train_year) == 2024
    assert int(patched.eval_year) == 2024
    assert int(patched.train_episode_days) == 14
    assert int(patched.eval_days) == 14
    assert int(patched.train_validation_days) == 14
    assert str(patched.train_validation_offset_days_within_year) == "30,120"
    assert int(patched.train_validation_checkpoint_every) == 1000


def test_short_cross_fidelity_probe_keeps_coarser_default_validation_interval_for_longer_ieee33_sac_runs():
    raw_argv = [
        "--cases",
        "ieee33",
        "--agent",
        "sac",
        "--train-steps",
        "20000",
    ]
    args = build_parser().parse_args(raw_argv)

    patched = apply_ieee33_sac_default_protocol(args, case_key="ieee33", raw_argv=raw_argv)

    assert bool(getattr(patched, "ieee33_sac_default_protocol_applied", False)) is True
    assert int(patched.train_validation_checkpoint_every) == 5000


def test_peak_price_reserve_metrics_capture_evening_headroom_collapse():
    trajectory = pd.DataFrame(
        {
            "price": [0.30, 0.52, 0.60, 0.45],
            "soc": [0.80, 0.22, 0.11, 0.50],
            "battery_discharge_power_limit_w": [500_000.0, 80_000.0, 20_000.0, 400_000.0],
        }
    )
    metrics = _peak_price_reserve_metrics(
        trajectory,
        peak_price_threshold=0.51373,
        discharge_limit_scale_w=500_000.0,
        low_discharge_limit_threshold=0.25,
    )
    assert metrics["peak_price_step_fraction"] == pytest.approx(0.5)
    assert metrics["peak_price_mean_soc"] == pytest.approx(0.165)
    assert metrics["peak_price_mean_discharge_limit_ratio"] == pytest.approx(0.1)
    assert metrics["peak_price_low_discharge_limit_dwell_fraction"] == pytest.approx(1.0)


def test_health_objective_validation_can_penalize_peak_reserve_collapse():
    summary = {
        "final_cumulative_cost": 1000.0,
        "total_terminal_soc_penalty": 0.0,
        "soc_upper_dwell_fraction": 0.0,
        "soc_lower_dwell_fraction": 0.0,
        "infeasible_action_dwell_fraction": 0.0,
        "peak_price_low_discharge_limit_dwell_fraction": 0.75,
    }
    baseline = _validation_metric_value(summary, "health_objective", {"peak_reserve_weight": 0.0})
    penalized = _validation_metric_value(summary, "health_objective", {"peak_reserve_weight": 10000.0})
    assert baseline == pytest.approx(1000.0)
    assert penalized == pytest.approx(8500.0)


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


def test_short_cross_fidelity_probe_accepts_new_fidelity_ladder_stage_names():
    args = build_parser().parse_args(
        [
            "--train-models",
            "thevenin_rint_only+thevenin_rint_thermal_stress+thevenin_full",
            "--train-steps",
            "12",
        ]
    )

    schedule = resolve_training_schedule("thevenin_rint_only+thevenin_rint_thermal_stress+thevenin_full", args)

    assert schedule["stages"] == ["thevenin_rint_only", "thevenin_rint_thermal_stress", "thevenin_full"]
    assert schedule["stage_count"] == 3
    assert sum(schedule["stage_steps"]) == 12


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
