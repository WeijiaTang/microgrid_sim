from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_fidelity_summary_tables_aggregates_probe_outputs(tmp_path: Path):
    root = Path(__file__).resolve().parents[2]
    probe_dir_a = tmp_path / "probe_a"
    probe_dir_b = tmp_path / "probe_b"
    aggregate_dir = tmp_path / "aggregate"

    common_probe_args = [
        "--cases",
        "ieee33",
        "--regimes",
        "base",
        "--train-models",
        "simple",
        "--test-models",
        "simple,thevenin",
        "--agent",
        "sac",
        "--train-steps",
        "1",
        "--eval-steps",
        "2",
        "--days",
        "1",
    ]

    for seed, out_dir in [(42, probe_dir_a), (43, probe_dir_b)]:
        command = [
            sys.executable,
            str(root / "scripts" / "analysis" / "short_cross_fidelity_probe.py"),
            *common_probe_args,
            "--seed",
            str(seed),
            "--output-dir",
            str(out_dir),
        ]
        subprocess.run(command, cwd=root, capture_output=True, text=True, check=True)

    aggregate_cmd = [
        sys.executable,
        str(root / "scripts" / "analysis" / "fidelity_summary_tables.py"),
        "--input-dirs",
        f"{probe_dir_a},{probe_dir_b}",
        "--output-dir",
        str(aggregate_dir),
    ]
    completed = subprocess.run(aggregate_cmd, cwd=root, capture_output=True, text=True, check=True)
    assert "Fidelity Summary Tables" in completed.stdout

    combined_detail = aggregate_dir / "combined_detail.csv"
    aggregate_csv = aggregate_dir / "aggregate_summary.csv"
    gap_csv = aggregate_dir / "test_model_gap_summary.csv"
    paper_csv = aggregate_dir / "paper_key_metrics.csv"
    best_csv = aggregate_dir / "best_train_family_by_test_env.csv"
    summary_json = aggregate_dir / "summary.json"

    assert combined_detail.exists()
    assert aggregate_csv.exists()
    assert gap_csv.exists()
    assert paper_csv.exists()
    assert best_csv.exists()
    assert summary_json.exists()

    detail_df = pd.read_csv(combined_detail)
    aggregate_df = pd.read_csv(aggregate_csv)
    gap_df = pd.read_csv(gap_csv)
    paper_df = pd.read_csv(paper_csv)
    best_df = pd.read_csv(best_csv)
    payload = json.loads(summary_json.read_text(encoding="utf-8"))

    assert len(detail_df) == 4
    assert not aggregate_df.empty
    assert not gap_df.empty
    assert not paper_df.empty
    assert not best_df.empty
    assert "reward_profile" in detail_df.columns
    assert "regularized_training" in detail_df.columns
    assert "cost_rank_within_test_env" in paper_df.columns
    assert payload["combined_rows"] == 4
    assert payload["paper_rows"] == len(paper_df)
    assert payload["best_rows"] == len(best_df)
