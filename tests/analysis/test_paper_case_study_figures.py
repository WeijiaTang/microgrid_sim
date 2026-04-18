from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "plot" / "paper_case_study_figures.py"
SPEC = importlib.util.spec_from_file_location("paper_case_study_figures", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

prepare_cigre_protocol_frame = MODULE.prepare_cigre_protocol_frame
prepare_ieee33_stress_frames = MODULE.prepare_ieee33_stress_frames
prepare_storage_sanity_frame = MODULE.prepare_storage_sanity_frame


def test_prepare_storage_sanity_frame_normalizes_by_case_baseline() -> None:
    df = pd.DataFrame(
        [
            {"case": "cigre", "battery_model": "none", "final_cumulative_cost": 100.0, "undervoltage_total": 0.0, "line_overload_total": 0.0},
            {"case": "cigre", "battery_model": "simple", "final_cumulative_cost": 80.0, "undervoltage_total": 0.0, "line_overload_total": 0.0},
            {"case": "ieee33", "battery_model": "none", "final_cumulative_cost": 200.0, "undervoltage_total": 1.0, "line_overload_total": 2.0},
            {"case": "ieee33", "battery_model": "simple", "final_cumulative_cost": 150.0, "undervoltage_total": 0.5, "line_overload_total": 1.0},
        ]
    )

    prepared = prepare_storage_sanity_frame(df)
    simple_cigre = prepared[(prepared["case"] == "cigre") & (prepared["battery_model"] == "simple")].iloc[0]
    simple_ieee = prepared[(prepared["case"] == "ieee33") & (prepared["battery_model"] == "simple")].iloc[0]

    assert simple_cigre["normalized_cost"] == 0.8
    assert simple_ieee["normalized_cost"] == 0.75


def test_prepare_cigre_protocol_frame_orders_current_paper_protocols() -> None:
    df = pd.DataFrame(
        [
            {"protocol": "full_to_full", "gate_passes": 1, "seeds": 3, "mean_savings_vs_none": 1.0, "min_savings_vs_none": -1.0, "max_savings_vs_none": 2.0, "mean_final_soc": 0.2, "mean_throughput_kwh": 1000.0},
            {"protocol": "simple_to_full", "gate_passes": 3, "seeds": 3, "mean_savings_vs_none": 5.0, "min_savings_vs_none": 4.0, "max_savings_vs_none": 6.0, "mean_final_soc": 0.4, "mean_throughput_kwh": 2000.0},
            {"protocol": "simple_to_simple", "gate_passes": 3, "seeds": 3, "mean_savings_vs_none": 7.0, "min_savings_vs_none": 6.0, "max_savings_vs_none": 8.0, "mean_final_soc": 0.45, "mean_throughput_kwh": 2100.0},
            {"protocol": "mixed_simple+thevenin_to_full", "gate_passes": 3, "seeds": 3, "mean_savings_vs_none": 5.2, "min_savings_vs_none": 4.5, "max_savings_vs_none": 6.1, "mean_final_soc": 0.42, "mean_throughput_kwh": 2050.0},
        ]
    )

    prepared = prepare_cigre_protocol_frame(df)
    assert prepared["protocol"].tolist() == [
        "simple_to_simple",
        "simple_to_full",
        "mixed_simple+thevenin_to_full",
        "full_to_full",
    ]
    assert prepared["gate_label"].tolist() == ["3/3", "3/3", "3/3", "1/3"]


def test_prepare_ieee33_stress_frames_maps_gate_colors() -> None:
    matched = pd.DataFrame(
        [
            {"train_model": "simple", "policy_regime": "upper_attractor", "objective_savings_vs_none": -1.0, "final_soc": 0.9, "total_battery_throughput_kwh": 100.0},
            {"train_model": "thevenin_rint_only", "policy_regime": "usable_but_reserve_thin", "objective_savings_vs_none": 1.0, "final_soc": 0.2, "total_battery_throughput_kwh": 500.0},
            {"train_model": "thevenin_rint_thermal_stress", "policy_regime": "upper_attractor", "objective_savings_vs_none": -1.1, "final_soc": 0.9, "total_battery_throughput_kwh": 110.0},
            {"train_model": "thevenin_full", "policy_regime": "healthy_mid", "objective_savings_vs_none": 5.0, "final_soc": 0.3, "total_battery_throughput_kwh": 900.0},
        ]
    )
    gate = pd.DataFrame(
        [
            {"seed": 42, "train_model": "thevenin_rint_only", "reasonable_dispatch_gate": "pass", "objective_savings_vs_none": 1.0, "final_soc": 0.2},
            {"seed": 52, "train_model": "thevenin_full", "reasonable_dispatch_gate": "fail", "objective_savings_vs_none": -2.0, "final_soc": 0.9},
        ]
    )

    ladder, gate_frame = prepare_ieee33_stress_frames(matched, gate)
    assert ladder["model_label"].tolist() == ["Simple", "Rint-only", "Rint + thermal", "Full Thevenin"]
    assert gate_frame["pass_color"].tolist()[0] != gate_frame["pass_color"].tolist()[1]
