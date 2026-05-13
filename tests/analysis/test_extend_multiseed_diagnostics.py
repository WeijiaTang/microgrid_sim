from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "analysis" / "extend_multiseed_diagnostics.py"
SPEC = importlib.util.spec_from_file_location("extend_multiseed_diagnostics", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

gate_mask_from_columns = MODULE._gate_mask_from_columns


def test_gate_mask_uses_documented_strict_less_than_point_zero_five_threshold() -> None:
    frame = pd.DataFrame(
        [
            {"savings": 1.0, "upper": 0.049, "lower": 0.0, "infeasible": 0.0},
            {"savings": 1.0, "upper": 0.05, "lower": 0.0, "infeasible": 0.0},
            {"savings": 1.0, "upper": 0.0, "lower": 0.049, "infeasible": 0.0},
            {"savings": 1.0, "upper": 0.0, "lower": 0.05, "infeasible": 0.0},
            {"savings": 1.0, "upper": 0.0, "lower": 0.0, "infeasible": 0.049},
            {"savings": 1.0, "upper": 0.0, "lower": 0.0, "infeasible": 0.05},
            {"savings": -1.0, "upper": 0.0, "lower": 0.0, "infeasible": 0.0},
        ]
    )

    mask = gate_mask_from_columns(
        savings_vs_none=frame["savings"],
        upper_dwell=frame["upper"],
        lower_dwell=frame["lower"],
        infeasible_dwell=frame["infeasible"],
    )

    assert mask.tolist() == [True, False, True, False, True, False, False]
