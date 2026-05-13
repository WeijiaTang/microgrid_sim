from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def _make_minimal_trajectory(path: Path) -> None:
    frame = pd.DataFrame(
        [
            {
                "step": 0,
                "reward": -1.0,
                "soc": 0.50,
                "temperature_c": 25.0,
                "load_w": 1000.0,
                "pv_w": 100.0,
                "price": 0.25,
                "grid_import_mw": 0.001,
                "grid_export_mw": 0.0,
                "min_bus_voltage_pu": 0.98,
                "max_bus_voltage_pu": 1.01,
                "max_line_loading_pct": 25.0,
                "mean_line_loading_pct": 12.5,
                "battery_power_mw": 0.02,
                "battery_action_applied": 0.10,
                "battery_action_feasible_low": -1.0,
                "battery_action_feasible_high": 1.0,
                "battery_charge_fraction_feasible": 1.0,
                "battery_discharge_fraction_feasible": 1.0,
                "battery_action_infeasible_gap": 0.0,
                "battery_internal_clip_gap_w": 0.0,
                "discharge_limit_ratio": 1.0,
                "rule_based_action_hint": 0.0,
                "rule_guidance_mix": 0.0,
                "terminal_soc_target": 0.5,
                "terminal_soc_tolerance": 0.05,
                "terminal_soc_deviation": 0.0,
                "peak_reserve_shortfall": 0.0,
                "p_max_w": 500000.0,
                "p_max_trend_w": 0.0,
                "cumulative_cost": 10.0,
                "terminal_soc_penalty": 0.0,
            },
            {
                "step": 1,
                "reward": -2.0,
                "soc": 0.48,
                "temperature_c": 25.1,
                "load_w": 1050.0,
                "pv_w": 120.0,
                "price": 0.26,
                "grid_import_mw": 0.0012,
                "grid_export_mw": 0.0,
                "min_bus_voltage_pu": 0.975,
                "max_bus_voltage_pu": 1.015,
                "max_line_loading_pct": 27.0,
                "mean_line_loading_pct": 13.0,
                "battery_power_mw": 0.01,
                "battery_action_applied": 0.05,
                "battery_action_feasible_low": -1.0,
                "battery_action_feasible_high": 1.0,
                "battery_charge_fraction_feasible": 1.0,
                "battery_discharge_fraction_feasible": 1.0,
                "battery_action_infeasible_gap": 0.01,
                "battery_internal_clip_gap_w": 0.0,
                "discharge_limit_ratio": 0.8,
                "rule_based_action_hint": 0.0,
                "rule_guidance_mix": 0.0,
                "terminal_soc_target": 0.5,
                "terminal_soc_tolerance": 0.05,
                "terminal_soc_deviation": -0.02,
                "peak_reserve_shortfall": 0.02,
                "p_max_w": 480000.0,
                "p_max_trend_w": -20000.0,
                "cumulative_cost": 21.0,
                "terminal_soc_penalty": 0.0,
            },
            {
                "step": 2,
                "reward": -3.0,
                "soc": 0.47,
                "temperature_c": 25.2,
                "load_w": 1100.0,
                "pv_w": 80.0,
                "price": 0.28,
                "grid_import_mw": 0.0014,
                "grid_export_mw": 0.0,
                "min_bus_voltage_pu": 0.97,
                "max_bus_voltage_pu": 1.02,
                "max_line_loading_pct": 28.0,
                "mean_line_loading_pct": 14.0,
                "battery_power_mw": 0.00,
                "battery_action_applied": 0.00,
                "battery_action_feasible_low": -1.0,
                "battery_action_feasible_high": 1.0,
                "battery_charge_fraction_feasible": 1.0,
                "battery_discharge_fraction_feasible": 1.0,
                "battery_action_infeasible_gap": 0.0,
                "battery_internal_clip_gap_w": 0.0,
                "discharge_limit_ratio": 0.7,
                "rule_based_action_hint": 0.0,
                "rule_guidance_mix": 0.0,
                "terminal_soc_target": 0.5,
                "terminal_soc_tolerance": 0.05,
                "terminal_soc_deviation": -0.03,
                "peak_reserve_shortfall": 0.05,
                "p_max_w": 470000.0,
                "p_max_trend_w": -10000.0,
                "cumulative_cost": 33.0,
                "terminal_soc_penalty": 5.0,
            },
        ]
    )
    frame.to_csv(path, index=False)


def test_export_offline_dataset_from_trajectory_glob(tmp_path: Path) -> None:
    input_dir = tmp_path / "results" / "ieee33_plain_sac" / "trajectories"
    input_dir.mkdir(parents=True)
    trajectory_csv = input_dir / "ieee33_network_stress_sac_train-simple_test-thevenin_seed42.csv"
    _make_minimal_trajectory(trajectory_csv)

    output_dir = tmp_path / "offline_dataset_outputs"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "analysis" / "export_offline_dataset.py"),
        "--trajectory-glob",
        str(trajectory_csv.relative_to(tmp_path)).replace("\\", "/"),
        "--controller-source",
        "sac_plain",
        "--output-dir",
        str(output_dir.relative_to(tmp_path)).replace("\\", "/"),
    ]
    completed = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=True)
    assert "Matched trajectories: 1" in completed.stdout

    combined_csv = output_dir / "combined_offline_dataset.csv"
    combined_meta = output_dir / "combined_offline_dataset.meta.json"
    single_csv = output_dir / f"{trajectory_csv.stem}_offline.csv"
    single_meta = output_dir / f"{trajectory_csv.stem}_offline.meta.json"

    assert combined_csv.exists()
    assert combined_meta.exists()
    assert single_csv.exists()
    assert single_meta.exists()

    frame = pd.read_csv(combined_csv)
    assert len(frame) == 2
    assert list(frame.columns) == [
        "case",
        "regime",
        "battery_model",
        "controller_source",
        "step",
        "obs",
        "next_obs",
        "action",
        "reward",
        "done",
        "soc",
        "grid_import_mw",
        "min_bus_voltage_pu",
        "max_line_loading_pct",
        "cumulative_cost",
        "terminal_penalty",
        "infeasible_gap",
        "source_row_count",
    ]
    assert frame.loc[0, "case"] == "ieee33"
    assert frame.loc[0, "regime"] == "network_stress"
    assert frame.loc[0, "battery_model"] == "thevenin"
    assert frame.loc[0, "controller_source"] == "sac_plain"
    assert isinstance(json.loads(frame.loc[0, "obs"]), list)
    assert isinstance(json.loads(frame.loc[0, "next_obs"]), list)
    assert frame.loc[1, "done"] == 1

    single_meta_obj = json.loads(single_meta.read_text(encoding="utf-8"))
    assert single_meta_obj["controller_source"] == "sac_plain"
    assert single_meta_obj["transition_count"] == 2
