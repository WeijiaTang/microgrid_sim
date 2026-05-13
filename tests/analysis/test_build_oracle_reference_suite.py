from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.analysis.build_oracle_reference_suite import build_window_jobs, combine_oracle_summaries


def test_build_window_jobs_supports_quarterly_and_full_year_windows():
    jobs = build_window_jobs([30, 365], [0, 91, 182, 273], max_window_days=365)
    assert jobs == [
        {"window_days": 30, "offset_days_within_year": 0},
        {"window_days": 30, "offset_days_within_year": 91},
        {"window_days": 30, "offset_days_within_year": 182},
        {"window_days": 30, "offset_days_within_year": 273},
        {"window_days": 365, "offset_days_within_year": 0},
    ]


def test_combine_oracle_summaries_adds_window_metadata(tmp_path: Path):
    run_a = tmp_path / "30d_off0"
    run_b = tmp_path / "365d_off0"
    run_a.mkdir(parents=True, exist_ok=True)
    run_b.mkdir(parents=True, exist_ok=True)
    path_a = run_a / "protocol_summary.csv"
    path_b = run_b / "protocol_summary.csv"

    pd.DataFrame(
        [
            {
                "case": "IEEE 33-bus",
                "case_key": "ieee33_network",
                "regime": "network_stress",
                "battery_model": "simple",
                "none_objective": 1000.0,
                "oracle_objective": 900.0,
            }
        ]
    ).to_csv(path_a, index=False)
    pd.DataFrame(
        [
            {
                "case": "IEEE 33-bus",
                "case_key": "ieee33_network",
                "regime": "network_stress",
                "battery_model": "simple",
                "none_objective": 2000.0,
                "oracle_objective": 1500.0,
            }
        ]
    ).to_csv(path_b, index=False)

    combined = combine_oracle_summaries([path_a, path_b])
    assert len(combined) == 2
    assert set(combined["reference_window_days"]) == {30, 365}
    assert set(combined["reference_offset_days_within_year"]) == {0}
