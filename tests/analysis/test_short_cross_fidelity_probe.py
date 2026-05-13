from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.analysis.short_cross_fidelity_probe as short_cross_fidelity_probe_module
from scripts.analysis.short_cross_fidelity_probe import (
    _causal_heuristic_action,
    _peak_price_reserve_metrics,
    _seed_replay_buffer_with_causal_heuristic,
    _validation_metric_value,
    adaptive_online_safe_bc_gradient_steps,
    apply_ieee33_full_fair_protocol,
    apply_ieee33_sac_default_protocol,
    build_parser,
    build_env,
    online_safe_bc_priority_config,
    resolve_train_window,
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
    required_columns = {
        "case",
        "regime",
        "agent",
        "train_model",
        "test_model",
        "train_validation_metric",
        "train_validation_midband_dwell_weight",
        "train_validation_soc_target_tracking_weight",
        "train_validation_peak_discharge_headroom_weight",
        "train_validation_valley_charge_weight",
        "train_validation_peak_discharge_weight",
        "final_cumulative_cost",
        "final_cumulative_objective_cost",
        "peak_price_mean_discharge_limit_ratio",
        "peak_price_discharge_action_fraction",
        "valley_price_charge_action_fraction",
        "soc_midband_dwell_fraction",
        "soc_target_tracking_mae",
        "soc_upper_parking_fraction",
        "soc_lower_parking_fraction",
        "inventory_teacher_activation_fraction",
        "mean_abs_inventory_teacher_gap",
    }
    assert required_columns.issubset(set(summary_df.columns))

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
    assert summary_df.loc[0, "rule_guidance_policy"] == "rule"

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
        "step_reward_before_clip",
        "step_reward_after_clip",
        "battery_shaping_penalty",
        "reward_after_battery_shaping",
        "reward_after_peak_reserve_penalty",
        "reward_after_terminal_penalty",
        "reward_wrapper_adjustment",
        "import_cost",
        "export_revenue",
        "net_energy_cost",
        "grid_limit_penalty_cost",
        "total_grid_cost",
        "soc_center_penalty",
        "soc_edge_penalty",
        "boundary_dwell_penalty",
        "boundary_dwell_proximity",
        "boundary_dwell_lower_proximity",
        "boundary_dwell_upper_proximity",
        "peak_reserve_shortfall",
        "peak_reserve_penalty",
        "discharge_limit_ratio",
        "undervoltage",
        "overvoltage",
        "line_overload_pct",
        "transformer_overload_pct",
        "power_flow_failure_penalty",
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


def test_short_cross_fidelity_probe_supports_shield_fields(tmp_path: Path):
    output_dir = tmp_path / "probe_shield_outputs"
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
        "--shield-enabled",
        "--shield-hard-pullback-action",
        "0.2",
        "--output-dir",
        str(output_dir),
    ]
    subprocess.run(command, cwd=root, capture_output=True, text=True, check=True)

    summary_df = pd.read_csv(output_dir / "summary.csv")
    assert int(summary_df.loc[0, "shield_enabled"]) == 1
    assert float(summary_df.loc[0, "shield_hard_pullback_action"]) == 0.2
    assert "shield_activation_fraction" in summary_df.columns

    trajectory = pd.read_csv(next((output_dir / "trajectories").glob("*.csv")))
    assert "shield_pre_action" in trajectory.columns
    assert "shield_post_action" in trajectory.columns
    assert "shield_applied" in trajectory.columns


def test_short_cross_fidelity_probe_supports_offline_bc_warmstart_fields(tmp_path: Path):
    output_dir = tmp_path / "probe_offline_bc_outputs"
    dataset_path = tmp_path / "offline_dataset.csv"
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
        "--output-dir",
        str(output_dir),
    ]
    subprocess.run(command, cwd=root, capture_output=True, text=True, check=True)

    summary_df = pd.read_csv(output_dir / "summary.csv")
    assert summary_df.loc[0, "offline_dataset"] == str(dataset_path)
    assert summary_df.loc[0, "offline_dataset_controller_sources"] == "oracle"
    assert int(summary_df.loc[0, "offline_dataset_max_transitions"]) == 1
    assert int(summary_df.loc[0, "bc_pretrain_gradient_steps"]) == 1
    assert int(summary_df.loc[0, "bc_pretrain_batch_size"]) == 1
    assert int(summary_df.loc[0, "offline_bc_dataset_rows"]) == 2
    assert int(summary_df.loc[0, "offline_bc_replay_seeded_transitions"]) == 1
    assert int(summary_df.loc[0, "offline_bc_actor_gradient_steps"]) == 1


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
        charge_value = float(charge_action.reshape(-1)[0])
        normalized_feasible_limit = charge_limit_w / float(env.config.battery_params.p_charge_max)
        assert charge_value < 0.0
        assert abs(charge_value) <= normalized_feasible_limit + 1e-3
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
    assert float(patched.action_smoothing_coef) == 0.5
    assert float(patched.action_max_delta) == 0.1
    assert float(patched.action_rate_penalty) == 0.05
    assert bool(patched.battery_feasibility_aware) is True
    assert bool(patched.symmetric_battery_action) is True


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
    assert str(getattr(patched, "reward_profile", "")) == "paper_balanced"


def test_short_cross_fidelity_probe_keeps_default_validation_windows_when_only_checkpoint_interval_is_overridden():
    raw_argv = [
        "--cases",
        "ieee33",
        "--agent",
        "sac",
        "--train-steps",
        "30000",
        "--train-validation-checkpoint-every",
        "5000",
    ]
    args = build_parser().parse_args(raw_argv)

    patched = apply_ieee33_sac_default_protocol(args, case_key="ieee33", raw_argv=raw_argv)

    assert bool(getattr(patched, "ieee33_sac_default_protocol_applied", False)) is True
    assert int(patched.train_validation_days) == 7
    assert str(patched.train_validation_offset_days_within_year) == "0,91,182,273"
    assert int(patched.train_validation_checkpoint_every) == 5000
    assert str(patched.train_validation_metric) == "health_objective"
    assert str(getattr(patched, "reward_profile", "")) == "paper_balanced"


def test_short_cross_fidelity_probe_applies_ieee33_full_fair_protocol_to_full_fidelity_runs():
    raw_argv = [
        "--protocol-profile",
        "ieee33_full_fair",
        "--cases",
        "ieee33",
        "--agent",
        "sac",
        "--train-models",
        "thevenin_full",
    ]
    args = build_parser().parse_args(raw_argv)

    patched = apply_ieee33_sac_default_protocol(args, case_key="ieee33", raw_argv=raw_argv)
    patched = apply_ieee33_full_fair_protocol(
        patched,
        case_key="ieee33",
        train_model="thevenin_full",
        raw_argv=raw_argv,
    )

    assert bool(getattr(patched, "ieee33_sac_default_protocol_applied", False)) is True
    assert bool(getattr(patched, "ieee33_full_fair_protocol_applied", False)) is True
    assert int(patched.train_steps) == 50000
    assert float(patched.learning_rate) == pytest.approx(1e-4)
    assert bool(getattr(patched, "train_disable_explicit_battery_degradation_penalties", False)) is True
    assert float(patched.rule_guidance_mix) == pytest.approx(0.2)
    assert int(patched.rule_guidance_decay_steps) == 10000
    assert str(getattr(patched, "rule_guidance_policy", "rule")) == "rule"
    assert int(patched.train_validation_checkpoint_every) == 2500
    assert str(getattr(patched, "reward_profile", "")) == "paper_balanced"


def test_short_cross_fidelity_probe_full_fair_protocol_does_not_override_explicit_full_fidelity_settings():
    raw_argv = [
        "--protocol-profile",
        "ieee33_full_fair",
        "--cases",
        "ieee33",
        "--agent",
        "sac",
        "--train-models",
        "thevenin_full",
        "--train-steps",
        "60000",
        "--learning-rate",
        "5e-5",
        "--rule-guidance-mix",
        "0.35",
        "--rule-guidance-decay-steps",
        "7000",
        "--train-validation-checkpoint-every",
        "4000",
    ]
    args = build_parser().parse_args(raw_argv)

    patched = apply_ieee33_sac_default_protocol(args, case_key="ieee33", raw_argv=raw_argv)
    patched = apply_ieee33_full_fair_protocol(
        patched,
        case_key="ieee33",
        train_model="thevenin_full",
        raw_argv=raw_argv,
    )

    assert bool(getattr(patched, "ieee33_full_fair_protocol_applied", False)) is True
    assert int(patched.train_steps) == 60000
    assert float(patched.learning_rate) == pytest.approx(5e-5)
    assert bool(getattr(patched, "train_disable_explicit_battery_degradation_penalties", False)) is True
    assert float(patched.rule_guidance_mix) == pytest.approx(0.35)
    assert int(patched.rule_guidance_decay_steps) == 7000
    assert str(getattr(patched, "rule_guidance_policy", "rule")) == "rule"
    assert int(patched.train_validation_checkpoint_every) == 4000


def test_short_cross_fidelity_probe_keeps_explicit_reward_profile_for_ieee33_protocols():
    raw_argv = [
        "--protocol-profile",
        "ieee33_full_fair",
        "--cases",
        "ieee33",
        "--agent",
        "sac",
        "--train-models",
        "thevenin_full",
        "--reward-profile",
        "network",
    ]
    args = build_parser().parse_args(raw_argv)

    patched = apply_ieee33_sac_default_protocol(args, case_key="ieee33", raw_argv=raw_argv)
    patched = apply_ieee33_full_fair_protocol(
        patched,
        case_key="ieee33",
        train_model="thevenin_full",
        raw_argv=raw_argv,
    )

    assert str(getattr(patched, "reward_profile", "")) == "network"


def test_short_cross_fidelity_probe_full_fair_protocol_skips_non_full_fidelity_runs():
    raw_argv = [
        "--protocol-profile",
        "ieee33_full_fair",
        "--cases",
        "ieee33",
        "--agent",
        "sac",
        "--train-models",
        "simple",
    ]
    args = build_parser().parse_args(raw_argv)

    patched = apply_ieee33_sac_default_protocol(args, case_key="ieee33", raw_argv=raw_argv)
    patched = apply_ieee33_full_fair_protocol(
        patched,
        case_key="ieee33",
        train_model="simple",
        raw_argv=raw_argv,
    )

    assert bool(getattr(patched, "ieee33_sac_default_protocol_applied", False)) is True
    assert bool(getattr(patched, "ieee33_full_fair_protocol_applied", False)) is False
    assert int(patched.train_steps) == 5000
    assert float(patched.learning_rate) == pytest.approx(3e-4)
    assert float(patched.rule_guidance_mix) == pytest.approx(0.0)
    assert int(patched.rule_guidance_decay_steps) == 0


def test_short_cross_fidelity_probe_applies_ieee33_full_fair_closure_protocol_defaults():
    raw_argv = [
        "--protocol-profile",
        "ieee33_full_fair_closure",
        "--cases",
        "ieee33",
        "--agent",
        "sac",
        "--train-models",
        "thevenin_full",
    ]
    args = build_parser().parse_args(raw_argv)

    patched = apply_ieee33_sac_default_protocol(args, case_key="ieee33", raw_argv=raw_argv)
    patched = apply_ieee33_full_fair_protocol(
        patched,
        case_key="ieee33",
        train_model="thevenin_full",
        raw_argv=raw_argv,
    )

    assert bool(getattr(patched, "ieee33_sac_default_protocol_applied", False)) is True
    assert bool(getattr(patched, "ieee33_full_fair_protocol_applied", False)) is True
    assert bool(getattr(patched, "ieee33_full_fair_closure_protocol_applied", False)) is True
    assert bool(getattr(patched, "train_disable_explicit_battery_degradation_penalties", False)) is True
    assert int(patched.train_steps) == 50000
    assert float(patched.learning_rate) == pytest.approx(1e-4)
    assert float(patched.rule_guidance_mix) == pytest.approx(0.2)
    assert int(patched.rule_guidance_decay_steps) == 20000
    assert str(getattr(patched, "rule_guidance_policy", "")) == "terminal_balanced"
    assert int(getattr(patched, "causal_heuristic_warmstart_steps", 0)) == 5000
    assert str(getattr(patched, "causal_heuristic_warmstart_policy", "")) == "terminal_balanced"


def test_short_cross_fidelity_probe_full_fair_closure_does_not_force_large_warmstart_on_tiny_debug_runs():
    raw_argv = [
        "--protocol-profile",
        "ieee33_full_fair_closure",
        "--cases",
        "ieee33",
        "--agent",
        "sac",
        "--train-models",
        "thevenin_full",
        "--train-steps",
        "1",
    ]
    args = build_parser().parse_args(raw_argv)

    patched = apply_ieee33_sac_default_protocol(args, case_key="ieee33", raw_argv=raw_argv)
    patched = apply_ieee33_full_fair_protocol(
        patched,
        case_key="ieee33",
        train_model="thevenin_full",
        raw_argv=raw_argv,
    )

    assert int(getattr(patched, "causal_heuristic_warmstart_steps", 0)) == 0


def test_short_cross_fidelity_probe_applies_ieee33_full_fair_closure_gate_protocol_defaults():
    raw_argv = [
        "--protocol-profile",
        "ieee33_full_fair_closure_gate",
        "--cases",
        "ieee33",
        "--agent",
        "sac",
        "--train-models",
        "thevenin_full",
    ]
    args = build_parser().parse_args(raw_argv)

    patched = apply_ieee33_sac_default_protocol(args, case_key="ieee33", raw_argv=raw_argv)
    patched = apply_ieee33_full_fair_protocol(
        patched,
        case_key="ieee33",
        train_model="thevenin_full",
        raw_argv=raw_argv,
    )

    assert bool(getattr(patched, "ieee33_sac_default_protocol_applied", False)) is True
    assert bool(getattr(patched, "ieee33_full_fair_protocol_applied", False)) is True
    assert bool(getattr(patched, "ieee33_full_fair_closure_protocol_applied", False)) is True
    assert bool(getattr(patched, "train_disable_explicit_battery_degradation_penalties", False)) is True
    assert int(patched.train_steps) == 50000
    assert float(patched.learning_rate) == pytest.approx(1e-4)
    assert float(patched.rule_guidance_mix) == pytest.approx(0.1)
    assert int(patched.rule_guidance_decay_steps) == 10000
    assert str(getattr(patched, "rule_guidance_policy", "")) == "terminal_balanced"
    assert int(getattr(patched, "causal_heuristic_warmstart_steps", 0)) == 2500
    assert str(getattr(patched, "causal_heuristic_warmstart_policy", "")) == "blended"
    assert str(getattr(patched, "train_validation_metric", "")) == "inventory_value_gate"


def test_short_cross_fidelity_probe_applies_ieee33_full_fair_staged_gate_protocol_defaults():
    raw_argv = [
        "--protocol-profile",
        "ieee33_full_fair_staged_gate",
        "--cases",
        "ieee33",
        "--agent",
        "sac",
        "--train-models",
        "thevenin_full",
    ]
    args = build_parser().parse_args(raw_argv)

    patched = apply_ieee33_sac_default_protocol(args, case_key="ieee33", raw_argv=raw_argv)
    patched = apply_ieee33_full_fair_protocol(
        patched,
        case_key="ieee33",
        train_model="thevenin_full",
        raw_argv=raw_argv,
    )

    assert bool(getattr(patched, "ieee33_sac_default_protocol_applied", False)) is True
    assert bool(getattr(patched, "ieee33_full_fair_protocol_applied", False)) is True
    assert bool(getattr(patched, "ieee33_full_fair_closure_protocol_applied", False)) is True
    assert bool(getattr(patched, "train_disable_explicit_battery_degradation_penalties", False)) is True
    assert int(patched.train_steps) == 50000
    assert float(patched.learning_rate) == pytest.approx(1e-4)
    assert float(patched.rule_guidance_mix) == pytest.approx(0.0)
    assert int(patched.rule_guidance_decay_steps) == 0
    assert str(getattr(patched, "rule_guidance_policy", "")) == "blended"
    assert int(getattr(patched, "causal_heuristic_warmstart_steps", 0)) == 5000
    assert str(getattr(patched, "causal_heuristic_warmstart_policy", "")) == "blended"
    assert str(getattr(patched, "train_validation_metric", "")) == "inventory_value_gate"


def test_short_cross_fidelity_probe_applies_ieee33_full_fair_staged_gate_reserve_protocol_defaults():
    raw_argv = [
        "--protocol-profile",
        "ieee33_full_fair_staged_gate_reserve",
        "--cases",
        "ieee33",
        "--agent",
        "sac",
        "--train-models",
        "thevenin_full",
    ]
    args = build_parser().parse_args(raw_argv)

    patched = apply_ieee33_sac_default_protocol(args, case_key="ieee33", raw_argv=raw_argv)
    patched = apply_ieee33_full_fair_protocol(
        patched,
        case_key="ieee33",
        train_model="thevenin_full",
        raw_argv=raw_argv,
    )

    assert bool(getattr(patched, "ieee33_full_fair_protocol_applied", False)) is True
    assert bool(getattr(patched, "train_disable_explicit_battery_degradation_penalties", False)) is True
    assert float(getattr(patched, "rule_guidance_mix", 1.0)) == pytest.approx(0.0)
    assert int(getattr(patched, "rule_guidance_decay_steps", 1)) == 0
    assert str(getattr(patched, "causal_heuristic_warmstart_policy", "")) == "blended"
    assert str(getattr(patched, "train_validation_metric", "")) == "inventory_value_gate"
    assert float(getattr(patched, "train_validation_peak_reserve_weight", 0.0)) == pytest.approx(20000.0)
    assert float(getattr(patched, "train_validation_gate_peak_reserve_dwell_threshold", -1.0)) == pytest.approx(0.5)
    assert float(getattr(patched, "train_peak_reserve_weight_scale", 0.0)) == pytest.approx(3.0)
    assert float(getattr(patched, "train_peak_reserve_power_floor", 0.0)) == pytest.approx(0.35)


def test_short_cross_fidelity_probe_full_fair_protocol_allows_explicit_penalties_to_be_kept():
    raw_argv = [
        "--protocol-profile",
        "ieee33_full_fair_staged_gate_reserve",
        "--cases",
        "ieee33",
        "--agent",
        "sac",
        "--train-models",
        "thevenin_full",
        "--train-keep-explicit-battery-degradation-penalties",
    ]
    args = build_parser().parse_args(raw_argv)

    patched = apply_ieee33_sac_default_protocol(args, case_key="ieee33", raw_argv=raw_argv)
    patched = apply_ieee33_full_fair_protocol(
        patched,
        case_key="ieee33",
        train_model="thevenin_full",
        raw_argv=raw_argv,
    )

    assert bool(getattr(patched, "ieee33_full_fair_protocol_applied", False)) is True
    assert bool(getattr(patched, "train_disable_explicit_battery_degradation_penalties", True)) is False


def test_short_cross_fidelity_probe_excludes_explicit_validation_windows_from_random_training_starts():
    raw_argv = [
        "--cases",
        "ieee33",
        "--agent",
        "sac",
        "--train-steps",
        "25000",
    ]
    args = build_parser().parse_args(raw_argv)
    patched = apply_ieee33_sac_default_protocol(args, case_key="ieee33", raw_argv=raw_argv)

    train_window = resolve_train_window(case_key="ieee33", regime="network_stress", args=patched)

    assert train_window is not None
    assert bool(train_window["random_episode_start"]) is True
    explicit_hours = tuple(int(value) for value in train_window["full_year_random_start_hours"])
    explicit_days = {hour // 24 for hour in explicit_hours}
    validation_offsets = {0, 91, 182, 273}
    validation_days = 7
    for offset in validation_offsets:
        blocked_days = set(range(offset - int(patched.train_episode_days) + 1, offset + validation_days))
        blocked_days = {day for day in blocked_days if day >= 0}
        assert explicit_days.isdisjoint(blocked_days)


def test_short_cross_fidelity_probe_raises_when_validation_exclusions_leave_no_random_training_start():
    args = build_parser().parse_args(
        [
            "--days",
            "30",
            "--seed",
            "42",
            "--train-year",
            "2023",
            "--train-episode-days",
            "364",
            "--train-random-start-within-year",
            "--train-validation-days",
            "7",
            "--train-validation-offset-days-within-year",
            "0,1",
        ]
    )

    with pytest.raises(ValueError, match="No admissible random training starts remain"):
        resolve_train_window(case_key="ieee33", regime="network_stress", args=args)


def test_training_can_disable_explicit_battery_degradation_penalties_without_affecting_eval():
    args = build_parser().parse_args(
        [
            "--reward-profile",
            "paper_balanced",
            "--train-disable-explicit-battery-degradation-penalties",
        ]
    )

    train_env = build_env(
        case_key="ieee33",
        battery_model="thevenin_full",
        days=1,
        seed=42,
        regime="network_stress",
        args=args,
        training=True,
    )
    eval_env = build_env(
        case_key="ieee33",
        battery_model="thevenin_full",
        days=1,
        seed=42,
        regime="network_stress",
        args=args,
        training=False,
    )
    try:
        train_cfg = train_env.unwrapped.config
        eval_cfg = eval_env.unwrapped.config
        assert float(train_cfg.battery_throughput_penalty_per_kwh) == 0.0
        assert float(train_cfg.battery_loss_penalty_per_kwh) == 0.0
        assert float(train_cfg.battery_stress_penalty_per_kwh) == 0.0
        assert float(eval_cfg.battery_throughput_penalty_per_kwh) > 0.0
        assert float(eval_cfg.battery_loss_penalty_per_kwh) > 0.0
        assert float(eval_cfg.battery_stress_penalty_per_kwh) > 0.0
    finally:
        train_env.close()
        eval_env.close()


def test_training_can_strengthen_peak_reserve_reward_without_affecting_eval():
    args = build_parser().parse_args(
        [
            "--reward-profile",
            "paper_balanced",
            "--train-peak-reserve-weight-scale",
            "3.0",
            "--train-peak-reserve-power-floor",
            "0.35",
        ]
    )

    train_env = build_env(
        case_key="ieee33",
        battery_model="thevenin_full",
        days=1,
        seed=42,
        regime="network_stress",
        args=args,
        training=True,
    )
    eval_env = build_env(
        case_key="ieee33",
        battery_model="thevenin_full",
        days=1,
        seed=42,
        regime="network_stress",
        args=args,
        training=False,
    )
    try:
        train_cfg = train_env.unwrapped.config
        eval_cfg = eval_env.unwrapped.config
        assert float(train_cfg.reward.w_peak_reserve) == pytest.approx(float(eval_cfg.reward.w_peak_reserve) * 3.0)
        assert float(train_cfg.reward.peak_reserve_power_floor) == pytest.approx(0.35)
        assert float(eval_cfg.reward.peak_reserve_power_floor) == pytest.approx(0.25)
    finally:
        train_env.close()
        eval_env.close()


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
        "final_cumulative_objective_cost": 1000.0,
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


def test_health_objective_validation_includes_explicit_battery_penalties_once():
    summary = {
        "final_cumulative_cost": 1000.0,
        "final_cumulative_objective_cost": 1125.0,
        "total_terminal_soc_penalty": 25.0,
        "soc_upper_dwell_fraction": 0.10,
        "soc_lower_dwell_fraction": 0.05,
        "infeasible_action_dwell_fraction": 0.20,
        "peak_price_low_discharge_limit_dwell_fraction": 0.25,
    }
    metric = _validation_metric_value(
        summary,
        "health_objective",
        {
            "terminal_penalty_weight": 3.0,
            "boundary_dwell_weight": 10.0,
            "infeasible_dwell_weight": 20.0,
            "peak_reserve_weight": 40.0,
        },
    )
    expected = 1125.0 + (3.0 - 1.0) * 25.0 + 10.0 * 0.15 + 20.0 * 0.20 + 40.0 * 0.25
    assert metric == pytest.approx(expected)


def test_health_objective_gate_validation_applies_large_penalty_once_gate_is_breached():
    summary = {
        "final_cumulative_cost": 1000.0,
        "final_cumulative_objective_cost": 1000.0,
        "total_terminal_soc_penalty": 0.0,
        "soc_upper_dwell_fraction": 0.049,
        "soc_lower_dwell_fraction": 0.05,
        "infeasible_action_dwell_fraction": 0.20,
        "peak_price_low_discharge_limit_dwell_fraction": 0.0,
    }
    metric = _validation_metric_value(
        summary,
        "health_objective_gate",
        {
            "boundary_dwell_weight": 0.0,
            "infeasible_dwell_weight": 0.0,
            "peak_reserve_weight": 0.0,
            "gate_dwell_threshold": 0.05,
            "gate_violation_weight": 1000.0,
        },
    )
    expected_gate_penalty = 1000.0 * (2.0 + ((0.20 - 0.05) / 0.05))
    assert metric == pytest.approx(1000.0 + expected_gate_penalty)


def test_health_objective_gate_validation_can_gate_peak_reserve_collapse():
    summary = {
        "final_cumulative_cost": 1000.0,
        "final_cumulative_objective_cost": 1000.0,
        "total_terminal_soc_penalty": 0.0,
        "soc_upper_dwell_fraction": 0.0,
        "soc_lower_dwell_fraction": 0.0,
        "infeasible_action_dwell_fraction": 0.0,
        "peak_price_low_discharge_limit_dwell_fraction": 0.75,
    }
    metric = _validation_metric_value(
        summary,
        "health_objective_gate",
        {
            "boundary_dwell_weight": 0.0,
            "infeasible_dwell_weight": 0.0,
            "peak_reserve_weight": 0.0,
            "gate_dwell_threshold": 0.05,
            "gate_peak_reserve_dwell_threshold": 0.5,
            "gate_violation_weight": 1000.0,
        },
    )
    expected_gate_penalty = 1000.0 * (1.0 + ((0.75 - 0.5) / 0.5))
    assert metric == pytest.approx(1000.0 + expected_gate_penalty)


def test_health_objective_gate_shield_validation_penalizes_shield_dependence():
    summary = {
        "final_cumulative_cost": 1000.0,
        "final_cumulative_objective_cost": 1000.0,
        "total_terminal_soc_penalty": 0.0,
        "soc_upper_dwell_fraction": 0.0,
        "soc_lower_dwell_fraction": 0.0,
        "infeasible_action_dwell_fraction": 0.0,
        "peak_price_low_discharge_limit_dwell_fraction": 0.0,
        "mean_abs_shield_delta": 0.20,
        "shield_material_activation_fraction": 0.40,
        "shield_strong_activation_fraction": 0.10,
        "final_terminal_soc_deviation": 0.05,
    }
    metric = _validation_metric_value(
        summary,
        "health_objective_gate_shield",
        {
            "boundary_dwell_weight": 0.0,
            "infeasible_dwell_weight": 0.0,
            "peak_reserve_weight": 0.0,
            "terminal_penalty_weight": 1.0,
            "shield_mean_delta_weight": 1000.0,
            "shield_material_dwell_weight": 2000.0,
            "shield_strong_dwell_weight": 4000.0,
            "final_soc_deviation_weight": 5000.0,
            "gate_dwell_threshold": 0.05,
            "gate_violation_weight": 1000.0,
            "gate_peak_reserve_dwell_threshold": -1.0,
        },
    )
    expected_base = 1000.0 + 1000.0 * 0.20 + 2000.0 * 0.40 + 4000.0 * 0.10 + 5000.0 * 0.05
    expected_gate_penalty = 1000.0 * (2.0 + ((0.40 - 0.05) / 0.05) + ((0.10 - 0.05) / 0.05))
    assert metric == pytest.approx(expected_base + expected_gate_penalty)


def test_health_objective_gate_shield_validation_supports_explicit_shield_gate_thresholds():
    summary = {
        "final_cumulative_cost": 1000.0,
        "final_cumulative_objective_cost": 1000.0,
        "total_terminal_soc_penalty": 0.0,
        "soc_upper_dwell_fraction": 0.0,
        "soc_lower_dwell_fraction": 0.0,
        "infeasible_action_dwell_fraction": 0.0,
        "peak_price_low_discharge_limit_dwell_fraction": 0.0,
        "mean_abs_shield_delta": 0.020,
        "shield_material_activation_fraction": 0.25,
        "shield_strong_activation_fraction": 0.03,
        "final_terminal_soc_deviation": 0.0,
    }
    metric = _validation_metric_value(
        summary,
        "health_objective_gate_shield",
        {
            "boundary_dwell_weight": 0.0,
            "infeasible_dwell_weight": 0.0,
            "peak_reserve_weight": 0.0,
            "terminal_penalty_weight": 1.0,
            "shield_mean_delta_weight": 0.0,
            "shield_material_dwell_weight": 0.0,
            "shield_strong_dwell_weight": 0.0,
            "final_soc_deviation_weight": 0.0,
            "gate_dwell_threshold": 0.05,
            "gate_violation_weight": 1000.0,
            "gate_peak_reserve_dwell_threshold": -1.0,
            "shield_mean_delta_threshold": 0.01,
            "shield_material_dwell_threshold": 0.20,
            "shield_strong_dwell_threshold": 0.02,
        },
    )
    expected_gate_penalty = 1000.0 * (
        3.0
        + ((0.020 - 0.010) / 0.010)
        + ((0.25 - 0.20) / 0.20)
        + ((0.03 - 0.02) / 0.02)
    )
    assert metric == pytest.approx(1000.0 + expected_gate_penalty)


def test_inventory_value_validation_penalizes_poor_inventory_morphology_and_value_capture():
    summary = {
        "final_cumulative_cost": 1000.0,
        "final_cumulative_objective_cost": 1000.0,
        "total_terminal_soc_penalty": 0.0,
        "soc_upper_dwell_fraction": 0.0,
        "soc_lower_dwell_fraction": 0.0,
        "infeasible_action_dwell_fraction": 0.0,
        "peak_price_low_discharge_limit_dwell_fraction": 0.0,
        "final_terminal_soc_deviation": 0.0,
        "soc_midband_dwell_fraction": 0.25,
        "soc_target_tracking_mae": 0.10,
        "peak_price_mean_discharge_limit_ratio": 0.20,
        "peak_price_discharge_action_fraction": 0.05,
        "peak_price_step_fraction": 1.0,
        "valley_price_charge_action_fraction": 0.10,
        "valley_price_mean_charge_limit_ratio": 0.60,
        "valley_price_step_fraction": 1.0,
    }
    metric = _validation_metric_value(
        summary,
        "inventory_value",
        {
            "boundary_dwell_weight": 0.0,
            "infeasible_dwell_weight": 0.0,
            "peak_reserve_weight": 0.0,
            "final_soc_deviation_weight": 0.0,
            "midband_dwell_weight": 100.0,
            "soc_target_tracking_weight": 200.0,
            "peak_discharge_headroom_weight": 300.0,
            "valley_charge_weight": 400.0,
            "peak_discharge_weight": 500.0,
            "peak_discharge_limit_threshold": 0.25,
        },
    )
    expected = (
        1000.0
        + 100.0 * (1.0 - 0.25)
        + 200.0 * 0.10
        + 300.0 * (0.25 - 0.20)
        + 400.0 * (0.60 - 0.10)
        + 500.0 * (0.20 - 0.05)
    )
    assert metric == pytest.approx(expected)


def test_inventory_value_gate_validation_reuses_gate_logic():
    summary = {
        "final_cumulative_cost": 1000.0,
        "final_cumulative_objective_cost": 1000.0,
        "total_terminal_soc_penalty": 0.0,
        "soc_upper_dwell_fraction": 0.051,
        "soc_lower_dwell_fraction": 0.0,
        "infeasible_action_dwell_fraction": 0.0,
        "peak_price_low_discharge_limit_dwell_fraction": 0.0,
        "soc_midband_dwell_fraction": 1.0,
        "soc_target_tracking_mae": 0.0,
        "peak_price_mean_discharge_limit_ratio": 0.5,
        "peak_price_discharge_action_fraction": 0.5,
        "peak_price_step_fraction": 1.0,
        "valley_price_charge_action_fraction": 0.5,
        "valley_price_mean_charge_limit_ratio": 0.5,
        "valley_price_step_fraction": 1.0,
    }
    metric = _validation_metric_value(
        summary,
        "inventory_value_gate",
        {
            "boundary_dwell_weight": 0.0,
            "infeasible_dwell_weight": 0.0,
            "peak_reserve_weight": 0.0,
            "midband_dwell_weight": 0.0,
            "soc_target_tracking_weight": 0.0,
            "peak_discharge_headroom_weight": 0.0,
            "valley_charge_weight": 0.0,
            "peak_discharge_weight": 0.0,
            "gate_dwell_threshold": 0.05,
            "gate_violation_weight": 1000.0,
        },
    )
    expected_gate_penalty = 1000.0 * (1.0 + ((0.051 - 0.05) / 0.05))
    assert metric == pytest.approx(1000.0 + expected_gate_penalty)




def test_adaptive_online_safe_bc_gradient_steps_can_trigger_on_inventory_morphology():
    args = build_parser().parse_args(
        [
            "--online-safe-bc-gradient-steps",
            "8",
            "--online-safe-bc-adaptive-scale-factor",
            "2",
            "--online-safe-bc-adaptive-max-gradient-steps",
            "64",
            "--online-safe-bc-adaptive-midband-dwell-threshold",
            "0.75",
            "--online-safe-bc-adaptive-soc-target-mae-threshold",
            "0.08",
            "--online-safe-bc-adaptive-peak-discharge-action-threshold",
            "0.20",
            "--online-safe-bc-adaptive-valley-charge-action-threshold",
            "0.20",
        ]
    )
    validation_state = {
        "last_validation_mean_shield_material_activation_fraction": 0.0,
        "last_validation_mean_abs_shield_delta": 0.0,
        "last_validation_mean_soc_midband_dwell_fraction": 0.40,
        "last_validation_mean_soc_target_tracking_mae": 0.12,
        "last_validation_mean_peak_price_discharge_action_fraction": 0.05,
        "last_validation_mean_valley_price_charge_action_fraction": 0.05,
        "stale_validation_rounds": 0,
    }

    assert adaptive_online_safe_bc_gradient_steps(args, validation_state) == 16


def test_adaptive_online_safe_bc_gradient_steps_backs_off_when_protocol_dependence_is_high():
    args = build_parser().parse_args(
        [
            "--online-safe-bc-gradient-steps",
            "8",
            "--online-safe-bc-adaptive-scale-factor",
            "2",
            "--online-safe-bc-adaptive-max-gradient-steps",
            "64",
            "--online-safe-bc-adaptive-shield-material-threshold",
            "0.40",
            "--online-safe-bc-adaptive-shield-delta-threshold",
            "0.015",
        ]
    )
    validation_state = {
        "last_validation_mean_shield_material_activation_fraction": 0.75,
        "last_validation_mean_abs_shield_delta": 0.02,
        "last_validation_mean_soc_midband_dwell_fraction": 0.95,
        "last_validation_mean_soc_target_tracking_mae": 0.02,
        "last_validation_mean_peak_price_discharge_action_fraction": 0.50,
        "last_validation_mean_valley_price_charge_action_fraction": 0.50,
        "stale_validation_rounds": 3,
    }

    assert adaptive_online_safe_bc_gradient_steps(args, validation_state) == 4


def test_adaptive_online_safe_bc_gradient_steps_does_not_upscale_when_recent_online_replay_is_small():
    args = build_parser().parse_args(
        [
            "--online-safe-bc-gradient-steps",
            "8",
            "--online-safe-bc-batch-size",
            "256",
            "--online-safe-bc-adaptive-scale-factor",
            "2",
            "--online-safe-bc-adaptive-max-gradient-steps",
            "64",
            "--online-safe-bc-adaptive-midband-dwell-threshold",
            "0.75",
            "--online-safe-bc-adaptive-soc-target-mae-threshold",
            "0.08",
            "--online-safe-bc-adaptive-peak-discharge-action-threshold",
            "0.20",
            "--online-safe-bc-adaptive-valley-charge-action-threshold",
            "0.20",
        ]
    )
    validation_state = {
        "online_safe_bc_replay_rows": 250,
        "last_validation_mean_shield_material_activation_fraction": 0.0,
        "last_validation_mean_abs_shield_delta": 0.0,
        "last_validation_mean_soc_midband_dwell_fraction": 0.40,
        "last_validation_mean_soc_target_tracking_mae": 0.12,
        "last_validation_mean_peak_price_discharge_action_fraction": 0.05,
        "last_validation_mean_valley_price_charge_action_fraction": 0.05,
        "stale_validation_rounds": 0,
    }

    assert adaptive_online_safe_bc_gradient_steps(args, validation_state) == 8


def test_online_safe_bc_priority_config_exposes_separate_value_row_weights():
    args = build_parser().parse_args([])
    config = online_safe_bc_priority_config(args)

    assert config["teacher_priority_coef"] == pytest.approx(2.0)
    assert config["peak_value_priority_coef"] == pytest.approx(0.75)
    assert config["valley_value_priority_coef"] == pytest.approx(0.5)


def test_online_safe_bc_priority_config_downscales_only_correction_priorities_for_small_recent_replay():
    args = build_parser().parse_args(
        [
            "--online-safe-bc-batch-size",
            "256",
            "--online-safe-bc-small-replay-priority-scale",
            "0.5",
            "--online-safe-bc-small-replay-min-rows-multiplier",
            "4",
        ]
    )
    config = online_safe_bc_priority_config(args, available_replay_rows=500)

    assert config["intervention_priority_coef"] == pytest.approx(2.0)
    assert config["boundary_priority_coef"] == pytest.approx(1.0)
    assert config["delta_priority_coef"] == pytest.approx(1.0)
    assert config["teacher_priority_coef"] == pytest.approx(2.0)
    assert config["peak_value_priority_coef"] == pytest.approx(0.75)
    assert config["valley_value_priority_coef"] == pytest.approx(0.5)
    assert config["terminal_priority_coef"] == pytest.approx(2.0)
    assert config["reserve_priority_coef"] == pytest.approx(1.0)
    assert config["terminal_deviation_priority_coef"] == pytest.approx(1.0)


def test_terminal_balanced_causal_heuristic_pushes_soc_toward_terminal_target():
    params = type(
        "Params",
        (),
        {
            "soc_init": 0.5,
            "soc_min": 0.1,
            "soc_max": 0.9,
            "p_charge_max": 100.0,
            "p_discharge_max": 100.0,
            "nominal_energy_wh": 1000.0,
        },
    )()
    config = type(
        "Config",
        (),
        {
            "battery_params": params,
            "dt_seconds": 3600.0,
            "grid_import_max": float("inf"),
            "grid_export_max": float("inf"),
            "terminal_soc_target": 0.5,
            "terminal_soc_tolerance": 0.05,
            "reward": type("Reward", (), {"valley_price": 0.1, "peak_price": 0.9})(),
        },
    )()
    battery = type(
        "Battery",
        (),
        {
            "soc": 0.2,
            "power_command_bounds": lambda self, dt: (-100.0, 100.0),
        },
    )()
    env = type(
        "Env",
        (),
        {
            "action_space": type("Space", (), {"shape": (1,)})(),
            "_profiles": type("Profiles", (), {"load_w": [0.0] * 10, "pv_w": [0.0] * 10, "price": [0.5] * 10})(),
            "config": config,
            "battery": battery,
            "current_step": 8,
            "total_steps": 10,
        },
    )()

    action = _causal_heuristic_action(env, "terminal_balanced")

    assert action.shape == (1,)
    assert float(action[0]) < 0.0


def test_terminal_balanced_causal_heuristic_pushes_away_from_upper_boundary_before_terminal_step():
    params = type(
        "Params",
        (),
        {
            "soc_init": 0.5,
            "soc_min": 0.1,
            "soc_max": 0.9,
            "p_charge_max": 100.0,
            "p_discharge_max": 100.0,
            "nominal_energy_wh": 1000.0,
        },
    )()
    config = type(
        "Config",
        (),
        {
            "battery_params": params,
            "dt_seconds": 3600.0,
            "grid_import_max": float("inf"),
            "grid_export_max": float("inf"),
            "terminal_soc_target": 0.5,
            "terminal_soc_tolerance": 0.05,
            "reward": type("Reward", (), {"valley_price": 0.9, "peak_price": 1.1})(),
        },
    )()
    battery = type(
        "Battery",
        (),
        {
            "soc": 0.86,
            "power_command_bounds": lambda self, dt: (-100.0, 100.0),
        },
    )()
    env = type(
        "Env",
        (),
        {
            "action_space": type("Space", (), {"shape": (1,)})(),
            "_profiles": type("Profiles", (), {"load_w": [0.0] * 10, "pv_w": [0.0] * 10, "price": [0.1] * 10})(),
            "config": config,
            "battery": battery,
            "current_step": 2,
            "total_steps": 10,
        },
    )()

    action = _causal_heuristic_action(env, "terminal_balanced")

    assert action.shape == (1,)
    assert float(action[0]) > 0.0


def test_causal_heuristic_warmstart_preserves_policy_action_in_replay_buffer(monkeypatch: pytest.MonkeyPatch):
    class DummyReplayBuffer:
        def __init__(self):
            self.calls = []

        def add(self, obs, next_obs, action, rewards, dones, infos):
            self.calls.append(
                {
                    "obs": np.array(obs, copy=True),
                    "next_obs": np.array(next_obs, copy=True),
                    "action": np.array(action, copy=True),
                    "rewards": np.array(rewards, copy=True),
                    "dones": np.array(dones, copy=True),
                    "infos": infos,
                }
            )

    class DummyVecEnv:
        def __init__(self):
            self.envs = [type("EnvHolder", (), {"unwrapped": type("Unwrapped", (), {"action_space": type("Space", (), {"shape": (1,)})()})()})()]

        def reset(self):
            return np.array([[0.0]], dtype=np.float32)

        def step(self, action):
            info = {
                "battery_action_applied": 0.25,
                "action_after_rule_guidance": 0.75,
            }
            return (
                np.array([[1.0]], dtype=np.float32),
                np.array([0.5], dtype=np.float32),
                np.array([False]),
                [info],
            )

    class DummyAgent:
        def __init__(self):
            self.replay_buffer = DummyReplayBuffer()
            self._vec_env = DummyVecEnv()

        def get_env(self):
            return self._vec_env

    args = type(
        "Args",
        (),
        {
            "causal_heuristic_warmstart_steps": 1,
            "causal_heuristic_warmstart_policy": "rule",
        },
    )()
    agent = DummyAgent()

    expected_action = np.array([[0.33]], dtype=np.float32)
    monkeypatch.setattr(
        short_cross_fidelity_probe_module,
        "_causal_heuristic_action",
        lambda _env, _policy: expected_action.reshape(-1),
    )
    collected = _seed_replay_buffer_with_causal_heuristic(agent, args)

    assert collected == 1
    assert len(agent.replay_buffer.calls) == 1
    stored = agent.replay_buffer.calls[0]["action"]
    assert stored.shape == (1, 1)
    assert float(stored[0, 0]) == pytest.approx(float(expected_action[0, 0]))


def test_causal_heuristic_warmstart_resets_rule_guidance_progress():
    class DummyReplayBuffer:
        def add(self, obs, next_obs, action, rewards, dones, infos):
            del obs, next_obs, action, rewards, dones, infos

    class DummyGuidanceWrapper:
        def __init__(self):
            self.reset_calls = 0
            self.env = None
            self.unwrapped = type("Unwrapped", (), {"action_space": type("Space", (), {"shape": (1,)})()})()

        def reset_guidance_progress(self):
            self.reset_calls += 1

    class DummyVecEnv:
        def __init__(self, wrapped_env):
            self.envs = [wrapped_env]

        def reset(self):
            return np.array([[0.0]], dtype=np.float32)

        def step(self, action):
            del action
            return (
                np.array([[1.0]], dtype=np.float32),
                np.array([0.5], dtype=np.float32),
                np.array([False]),
                [{}],
            )

    class DummyAgent:
        def __init__(self, wrapped_env):
            self.replay_buffer = DummyReplayBuffer()
            self._vec_env = DummyVecEnv(wrapped_env)

        def get_env(self):
            return self._vec_env

    wrapped_env = DummyGuidanceWrapper()
    args = type(
        "Args",
        (),
        {
            "causal_heuristic_warmstart_steps": 2,
            "causal_heuristic_warmstart_policy": "rule",
        },
    )()
    agent = DummyAgent(wrapped_env)

    collected = _seed_replay_buffer_with_causal_heuristic(agent, args)

    assert collected == 2
    assert wrapped_env.reset_calls == 1


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


def test_short_cross_fidelity_probe_exports_full_fair_protocol_fields(tmp_path: Path):
    output_dir = tmp_path / "probe_full_fair_outputs"
    root = Path(__file__).resolve().parents[2]
    command = [
        sys.executable,
        str(root / "scripts" / "analysis" / "short_cross_fidelity_probe.py"),
        "--protocol-profile",
        "ieee33_full_fair",
        "--cases",
        "ieee33",
        "--regimes",
        "network_stress",
        "--train-models",
        "thevenin_full",
        "--test-models",
        "thevenin_full",
        "--agent",
        "sac",
        "--train-steps",
        "1",
        "--eval-steps",
        "2",
        "--days",
        "1",
        "--train-year",
        "2023",
        "--eval-year",
        "2024",
        "--train-episode-days",
        "1",
        "--eval-days",
        "1",
        "--train-validation-days",
        "0",
        "--seed",
        "42",
        "--output-dir",
        str(output_dir),
    ]
    subprocess.run(command, cwd=root, capture_output=True, text=True, check=True)

    summary_df = pd.read_csv(output_dir / "summary.csv")
    assert len(summary_df) == 1
    assert summary_df.loc[0, "protocol_profile"] == "ieee33_full_fair"
    assert int(summary_df.loc[0, "ieee33_sac_default_protocol_applied"]) == 1
    assert int(summary_df.loc[0, "ieee33_full_fair_protocol_applied"]) == 1
    assert int(summary_df.loc[0, "ieee33_full_fair_closure_protocol_applied"]) == 0
    assert int(summary_df.loc[0, "train_disable_explicit_battery_degradation_penalties"]) == 1
    assert float(summary_df.loc[0, "learning_rate"]) == pytest.approx(1e-4)
    assert float(summary_df.loc[0, "rule_guidance_mix"]) == pytest.approx(0.2)
    assert summary_df.loc[0, "rule_guidance_policy"] == "rule"


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


def test_short_cross_fidelity_probe_objective_cost_includes_explicit_battery_penalties(tmp_path: Path):
    output_dir = tmp_path / "probe_objective_accounting_outputs"
    root = Path(__file__).resolve().parents[2]
    command = [
        sys.executable,
        str(root / "scripts" / "analysis" / "short_cross_fidelity_probe.py"),
        "--cases",
        "ieee33",
        "--regimes",
        "network_stress",
        "--train-models",
        "thevenin_full",
        "--test-models",
        "thevenin_full",
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
    row = summary_df.loc[0]
    cfg = IEEE33Config(simulation_days=1, battery_model="thevenin_full", reward_profile="paper_balanced")
    expected = (
        float(row["final_cumulative_cost"])
        + float(row["total_terminal_soc_penalty"])
        + float(cfg.battery_throughput_penalty_per_kwh) * float(row["total_battery_throughput_kwh"])
        + float(cfg.battery_loss_penalty_per_kwh) * float(row["total_battery_loss_kwh"])
        + float(cfg.battery_stress_penalty_per_kwh) * float(row["total_battery_stress_kwh"])
    )
    assert float(row["final_cumulative_objective_cost"]) == pytest.approx(expected)


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


def test_ieee33_reward_wrapper_diagnostics_exports_aggregates(tmp_path: Path):
    probe_output_dir = tmp_path / "probe_diag_outputs"
    diagnostics_root = tmp_path / "diag_outputs"
    root = Path(__file__).resolve().parents[2]

    probe_command = [
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
        "--battery-feasibility-aware",
        "--output-dir",
        str(probe_output_dir),
    ]
    subprocess.run(probe_command, cwd=root, capture_output=True, text=True, check=True)

    diagnostics_command = [
        sys.executable,
        str(root / "scripts" / "analysis" / "ieee33_reward_wrapper_diagnostics.py"),
        "--summary-csv",
        str(probe_output_dir / "summary.csv"),
        "--output-csv",
        str(diagnostics_root / "reward_diagnostics.csv"),
        "--output-json",
        str(diagnostics_root / "reward_diagnostics.json"),
    ]
    subprocess.run(diagnostics_command, cwd=root, capture_output=True, text=True, check=True)

    diagnostics_df = pd.read_csv(diagnostics_root / "reward_diagnostics.csv")
    assert len(diagnostics_df) == 1
    for column in [
        "weighted_cost_term",
        "battery_shaping_term",
        "terminal_soc_term",
        "wrapper_penalty_term",
        "dominant_reward_block",
        "clipped_step_fraction",
    ]:
        assert column in diagnostics_df.columns


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
