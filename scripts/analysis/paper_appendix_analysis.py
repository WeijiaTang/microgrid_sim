#!/usr/bin/env python3
"""Rebuild appendix-ready statistical summaries from released paper artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = REPO_ROOT / "results" / "paper"
OUTPUT_DIR = PAPER_DIR / "analysis"
BOOTSTRAP_SAMPLES = 200_000
BOOTSTRAP_SEED = 20260323


@dataclass(frozen=True)
class CaseInference:
    case: str
    seed_count: int
    mean_gap_pct: float
    gap_ci95_low_pct: float
    gap_ci95_high_pct: float
    mean_delta_cost: float
    delta_cost_ci95_low: float
    delta_cost_ci95_high: float
    win_count: int
    sign_test_p_two_sided: float
    sign_test_p_one_sided: float


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Expected a non-empty 1-D array for bootstrap.")
    idx = rng.integers(0, values.size, size=(BOOTSTRAP_SAMPLES, values.size))
    boot_means = values[idx].mean(axis=1)
    low, high = np.quantile(boot_means, [0.025, 0.975])
    return float(low), float(high)


def compute_case_inference(case_name: str, rng: np.random.Generator) -> CaseInference:
    summary_path = PAPER_DIR / case_name / "summary_seeds.csv"
    df = pd.read_csv(summary_path)
    diffs = df["ebm_cost"].to_numpy(dtype=float) - df["pbm_cost"].to_numpy(dtype=float)
    gaps = df["gap_pct"].to_numpy(dtype=float)
    wins = int(np.sum(diffs > 0.0))
    n = int(len(df))
    gap_low, gap_high = _bootstrap_mean_ci(gaps, rng)
    diff_low, diff_high = _bootstrap_mean_ci(diffs, rng)
    return CaseInference(
        case=case_name,
        seed_count=n,
        mean_gap_pct=float(np.mean(gaps)),
        gap_ci95_low_pct=gap_low,
        gap_ci95_high_pct=gap_high,
        mean_delta_cost=float(np.mean(diffs)),
        delta_cost_ci95_low=diff_low,
        delta_cost_ci95_high=diff_high,
        win_count=wins,
        sign_test_p_two_sided=float(binomtest(wins, n=n, p=0.5, alternative="two-sided").pvalue),
        sign_test_p_one_sided=float(binomtest(wins, n=n, p=0.5, alternative="greater").pvalue),
    )


def compute_res_fixed_window_summary() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for path in sorted((PAPER_DIR / "res").glob("seed*/summary_windows.csv")):
        seed = int(path.parent.name.replace("seed", ""))
        frame = pd.read_csv(path)
        pbm = frame.loc[frame["method"] == "pbm", ["start_hour", "total_cost"]].rename(columns={"total_cost": "pbm_cost"})
        ebm = frame.loc[frame["method"] == "ebm", ["start_hour", "total_cost"]].rename(columns={"total_cost": "ebm_cost"})
        merged = pbm.merge(ebm, on="start_hour", how="inner")
        merged["seed"] = seed
        merged["start_day"] = (merged["start_hour"].astype(int) // 24).astype(int)
        merged["gap_pct"] = 100.0 * (merged["ebm_cost"] - merged["pbm_cost"]) / merged["pbm_cost"]
        merged["pbm_win"] = merged["ebm_cost"] > merged["pbm_cost"]
        rows.append(merged)
    merged = pd.concat(rows, ignore_index=True)
    summary = (
        merged.groupby(["start_hour", "start_day"], as_index=False)
        .agg(
            mean_pbm_cost=("pbm_cost", "mean"),
            mean_ebm_cost=("ebm_cost", "mean"),
            mean_gap_pct=("gap_pct", "mean"),
            std_gap_pct=("gap_pct", "std"),
            pbm_win_rate=("pbm_win", "mean"),
            pbm_win_count=("pbm_win", "sum"),
            seed_count=("seed", "count"),
        )
        .sort_values("start_hour")
    )
    summary["pbm_win_rate_pct"] = 100.0 * summary["pbm_win_rate"]
    return summary


def compute_agent_appendix_summary() -> pd.DataFrame:
    rows: list[dict] = []

    same_protocol_reports = {
        "SAC": REPO_ROOT / "results" / "paper" / "res" / "seed45" / "report.json",
        "PPO": REPO_ROOT / "results" / "_tmp_res45_paper_ppo20k" / "report.json",
        "DQN": REPO_ROOT / "results" / "_tmp_res45_paper_dqn20k" / "report.json",
    }
    for agent, path in same_protocol_reports.items():
        report = _load_json(path)
        rows.append(
            {
                "evidence_scope": "matched_res_seed45",
                "agent": agent,
                "seed_count": 1,
                "seed_label": "45",
                "pbm_cost": float(report["pbm_cost"]),
                "ebm_cost": float(report["ebm_cost"]),
                "gap_pct": float(report["gap_pct"]),
                "window_win_rate_pct": 100.0 * float(report["window_win_rate"]),
                "note": "Residential 20k fair-budget protocol with validation selection and action regularization.",
                "source": str(path.relative_to(REPO_ROOT)),
            }
        )

    td3_paths = sorted((REPO_ROOT / "results" / "_tmp_multiseed_td3_20k_monthval_fix").glob("seed*/report.json"))
    td3_reports = [_load_json(path) for path in td3_paths]
    rows.append(
        {
            "evidence_scope": "archived_res_multiseed",
            "agent": "TD3",
            "seed_count": len(td3_reports),
            "seed_label": ",".join(str(int(report["seed"])) for report in td3_reports),
            "pbm_cost": float(np.mean([float(report["pbm_cost"]) for report in td3_reports])),
            "ebm_cost": float(np.mean([float(report["ebm_cost"]) for report in td3_reports])),
            "gap_pct": float(np.mean([float(report["gap_pct"]) for report in td3_reports])),
            "window_win_rate_pct": float(100.0 * np.mean([float(report["window_win_rate"]) for report in td3_reports])),
            "note": "Archived 20k residential TD3 pilot package with matched PBM evaluation and multi-seed summary.",
            "source": str((REPO_ROOT / "results" / "_tmp_multiseed_td3_20k_monthval_fix").relative_to(REPO_ROOT)),
        }
    )

    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    inference_rows = [
        asdict(compute_case_inference("res", rng)),
        asdict(compute_case_inference("cigre", rng)),
    ]
    inference_df = pd.DataFrame(inference_rows)
    inference_df.to_csv(OUTPUT_DIR / "case_statistical_inference.csv", index=False)
    (OUTPUT_DIR / "case_statistical_inference.json").write_text(
        json.dumps(inference_rows, indent=2),
        encoding="utf-8",
    )

    res_window_df = compute_res_fixed_window_summary()
    res_window_df.to_csv(OUTPUT_DIR / "res_fixed_window_summary.csv", index=False)

    agent_df = compute_agent_appendix_summary()
    agent_df.to_csv(OUTPUT_DIR / "agent_appendix_summary.csv", index=False)

    print("Saved appendix analysis files:")
    print(f"  {OUTPUT_DIR / 'case_statistical_inference.csv'}")
    print(f"  {OUTPUT_DIR / 'case_statistical_inference.json'}")
    print(f"  {OUTPUT_DIR / 'res_fixed_window_summary.csv'}")
    print(f"  {OUTPUT_DIR / 'agent_appendix_summary.csv'}")


if __name__ == "__main__":
    main()
