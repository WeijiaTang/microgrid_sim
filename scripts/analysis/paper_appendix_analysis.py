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
    paired_cohens_dz: float


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_monitor_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, comment="#")


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
        paired_cohens_dz=float(np.mean(diffs) / np.std(diffs, ddof=1)) if n > 1 and np.std(diffs, ddof=1) > 0 else float("nan"),
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


def compute_res_validation_convergence() -> tuple[pd.DataFrame, pd.DataFrame]:
    canonical_seed_roots = [
        REPO_ROOT / "models" / "_tmp_multiseed_sac20k_a_fix" / "seed44",
        REPO_ROOT / "models" / "_tmp_multiseed_sac20k_a_fix" / "seed45",
        REPO_ROOT / "models" / "_tmp_multiseed_sac20k_a_fix" / "seed46",
        REPO_ROOT / "models" / "_tmp_multiseed_sac20k_b_fix" / "seed47",
        REPO_ROOT / "models" / "_tmp_multiseed_sac20k_b_fix" / "seed48",
        REPO_ROOT / "models" / "_tmp_res_multiseed_sac20k_actreg_a" / "seed49",
    ]
    detail_rows: list[dict] = []
    for seed_root in canonical_seed_roots:
        for path in sorted(seed_root.glob("*_validation.csv")):
            seed = int(seed_root.name.replace("seed", ""))
            method = "pbm" if path.name.startswith("pbm_") else "ebm"
            df = pd.read_csv(path)
            if df.empty:
                continue
            initial_row = df.iloc[0]
            final_row = df.iloc[-1]
            best_idx = df["validation_mean_cost"].astype(float).idxmin()
            best_row = df.loc[best_idx]
            tail = df["validation_mean_cost"].astype(float).tail(min(3, len(df)))
            initial_cost = float(initial_row["validation_mean_cost"])
            best_cost = float(best_row["validation_mean_cost"])
            final_cost = float(final_row["validation_mean_cost"])
            detail_rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "checkpoint_count": int(len(df)),
                    "initial_step": int(initial_row["total_steps_trained"]),
                    "best_step": int(best_row["total_steps_trained"]),
                    "final_step": int(final_row["total_steps_trained"]),
                    "initial_mean_cost": initial_cost,
                    "best_mean_cost": best_cost,
                    "final_mean_cost": final_cost,
                    "initial_to_best_improvement_pct": 100.0 * (initial_cost - best_cost) / initial_cost,
                    "initial_to_final_improvement_pct": 100.0 * (initial_cost - final_cost) / initial_cost,
                    "final_above_best_pct": 100.0 * (final_cost - best_cost) / best_cost if best_cost > 0.0 else 0.0,
                    "tail_mean_cost": float(tail.mean()),
                    "tail_std_cost": float(tail.std(ddof=0)),
                    "best_is_final": bool(int(best_idx) == int(len(df) - 1)),
                    "source": str(path.relative_to(REPO_ROOT)),
                }
            )

    detail_df = pd.DataFrame(detail_rows).sort_values(["method", "seed"]).reset_index(drop=True)
    summary_df = (
        detail_df.groupby("method", as_index=False)
        .agg(
            seed_count=("seed", "count"),
            mean_checkpoint_count=("checkpoint_count", "mean"),
            mean_initial_mean_cost=("initial_mean_cost", "mean"),
            mean_best_mean_cost=("best_mean_cost", "mean"),
            mean_final_mean_cost=("final_mean_cost", "mean"),
            mean_initial_to_best_improvement_pct=("initial_to_best_improvement_pct", "mean"),
            mean_initial_to_final_improvement_pct=("initial_to_final_improvement_pct", "mean"),
            mean_final_above_best_pct=("final_above_best_pct", "mean"),
            mean_best_step=("best_step", "mean"),
            best_is_final_count=("best_is_final", "sum"),
        )
        .sort_values("method")
        .reset_index(drop=True)
    )
    return detail_df, summary_df


def compute_cigre_sac_convergence() -> tuple[pd.DataFrame, pd.DataFrame]:
    canonical_seed_roots = [
        REPO_ROOT / "models" / "cigre_d4_multiseed_910k" / "seed42",
        REPO_ROOT / "models" / "cigre_d4_multiseed_910k" / "seed43",
        REPO_ROOT / "models" / "cigre_d4_multiseed_910k" / "seed44",
        REPO_ROOT / "models" / "cigre_d4_multiseed_910k" / "seed45",
        REPO_ROOT / "models" / "cigre_d4_multiseed_910k" / "seed46",
        REPO_ROOT / "models" / "cigre_d4_multiseed_910k" / "seed47",
    ]
    detail_rows: list[dict] = []
    for seed_root in canonical_seed_roots:
        seed = int(seed_root.name.replace("seed", ""))
        for method in ("pbm", "ebm"):
            phase_paths = sorted(
                seed_root.glob(f"{method}_seed{seed}_910000_train2023_eval2024_phase*.csv.monitor.csv"),
                key=lambda path: int(path.stem.split("_phase")[-1].split(".")[0]),
            )
            phase_frames: list[pd.DataFrame] = []
            for phase_path in phase_paths:
                phase_str = phase_path.stem.split("_phase")[-1].split(".")[0]
                phase_index = int(phase_str)
                phase_df = _load_monitor_csv(phase_path)
                if phase_df.empty:
                    continue
                phase_df = phase_df.copy()
                phase_df["phase_index"] = phase_index
                phase_df["episode_index_in_phase"] = np.arange(1, len(phase_df) + 1)
                phase_df["episode_cost_per_day"] = -phase_df["r"].astype(float) / np.maximum(
                    phase_df["l"].astype(float) / 24.0,
                    1e-9,
                )
                phase_frames.append(phase_df)

            if not phase_frames:
                continue

            merged = pd.concat(phase_frames, ignore_index=True)
            merged["episode_index"] = np.arange(1, len(merged) + 1)
            initial_row = merged.iloc[0]
            best_idx = merged["episode_cost_per_day"].astype(float).idxmin()
            best_row = merged.loc[best_idx]
            final_row = merged.iloc[-1]
            tail = merged["episode_cost_per_day"].astype(float).tail(min(5, len(merged)))
            initial_cost = float(initial_row["episode_cost_per_day"])
            best_cost = float(best_row["episode_cost_per_day"])
            final_cost = float(final_row["episode_cost_per_day"])
            detail_rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "phase_count": int(len(phase_frames)),
                    "episode_count": int(len(merged)),
                    "initial_phase_index": int(initial_row["phase_index"]),
                    "best_phase_index": int(best_row["phase_index"]),
                    "final_phase_index": int(final_row["phase_index"]),
                    "initial_episode_index": int(initial_row["episode_index"]),
                    "best_episode_index": int(best_row["episode_index"]),
                    "final_episode_index": int(final_row["episode_index"]),
                    "initial_cost_per_day": initial_cost,
                    "best_cost_per_day": best_cost,
                    "final_cost_per_day": final_cost,
                    "initial_to_best_improvement_pct": 100.0 * (initial_cost - best_cost) / max(abs(initial_cost), 1e-9),
                    "initial_to_final_improvement_pct": 100.0 * (initial_cost - final_cost) / max(abs(initial_cost), 1e-9),
                    "final_above_best_pct": 100.0 * (final_cost - best_cost) / max(abs(best_cost), 1e-9),
                    "tail_mean_cost_per_day": float(tail.mean()),
                    "tail_std_cost_per_day": float(tail.std(ddof=0)),
                    "best_is_final": bool(int(best_idx) == int(len(merged) - 1)),
                    "source": ";".join(str(path.relative_to(REPO_ROOT)) for path in phase_paths),
                }
            )

    detail_df = pd.DataFrame(detail_rows).sort_values(["method", "seed"]).reset_index(drop=True)
    summary_df = (
        detail_df.groupby("method", as_index=False)
        .agg(
            seed_count=("seed", "count"),
            mean_episode_count=("episode_count", "mean"),
            mean_initial_cost_per_day=("initial_cost_per_day", "mean"),
            mean_best_cost_per_day=("best_cost_per_day", "mean"),
            mean_final_cost_per_day=("final_cost_per_day", "mean"),
            mean_initial_to_best_improvement_pct=("initial_to_best_improvement_pct", "mean"),
            mean_initial_to_final_improvement_pct=("initial_to_final_improvement_pct", "mean"),
            mean_final_above_best_pct=("final_above_best_pct", "mean"),
            mean_best_episode_index=("best_episode_index", "mean"),
            best_is_final_count=("best_is_final", "sum"),
        )
        .sort_values("method")
        .reset_index(drop=True)
    )
    return detail_df, summary_df


def compute_agent_appendix_summary() -> pd.DataFrame:
    rows: list[dict] = []

    same_protocol_reports = {
        "SAC": {
            "family": "SAC-family",
            "path": REPO_ROOT / "results" / "paper" / "res" / "seed45" / "report.json",
            "scope": "matched_res_seed45_mainline",
            "note": "Residential released mainline SAC seed-45 package under the final matched protocol.",
        },
        "TQC": {
            "family": "SAC-family",
            "path": REPO_ROOT / "results" / "_tmp_res45_tqc_10k_short" / "report.json",
            "scope": "matched_res_seed45_short10k",
            "note": "Residential 10k fair-budget TQC rerun with validation selection and action regularization.",
        },
        "DDPG": {
            "family": "DDPG-family",
            "path": REPO_ROOT / "results" / "_tmp_res45_ddpg_10k_short" / "report.json",
            "scope": "matched_res_seed45_short10k",
            "note": "Residential 10k fair-budget DDPG rerun with validation selection and action regularization.",
        },
        "TD3": {
            "family": "DDPG-family",
            "path": REPO_ROOT / "results" / "_tmp_res45_td3_10k_rescue" / "report.json",
            "scope": "matched_res_seed45_rescue10k",
            "note": "Residential 10k TD3 rescue rerun with fair training, validation selection, and action regularization; PBM branch recovers nontrivial dispatch while EBM remains near-idle.",
        },
    }
    for agent, spec in same_protocol_reports.items():
        path = Path(spec["path"])
        report = _load_json(path)
        rows.append(
            {
                "evidence_scope": str(spec["scope"]),
                "family": str(spec["family"]),
                "agent": agent,
                "seed_count": 1,
                "seed_label": "45",
                "pbm_cost": float(report["pbm_cost"]),
                "ebm_cost": float(report["ebm_cost"]),
                "gap_pct": float(report["gap_pct"]),
                "window_win_rate_pct": 100.0 * float(report["window_win_rate"]),
                "note": str(spec["note"]),
                "source": str(path.relative_to(REPO_ROOT)),
            }
        )

    return pd.DataFrame(rows)


def compute_cigre_agent_scout_summary() -> pd.DataFrame:
    scout_specs = [
        {
            "family": "DDPG-family",
            "agent": "TD3",
            "steps": 20_000,
            "seed": 44,
            "path": REPO_ROOT / "results" / "_tmp_cigre_td3_20k_postfix" / "report.json",
            "note": "Post-fix 20k MG-CIGRE TD3 scout with validation selection and action regularization; stable and weakly PBM-favorable, but below the 1.5% practical threshold.",
        },
        {
            "family": "DDPG-family",
            "agent": "TD3",
            "steps": 20_000,
            "seed": 46,
            "path": REPO_ROOT / "results" / "_tmp_cigre_td3_20k_seed46_postfix" / "report.json",
            "note": "Post-fix 20k MG-CIGRE TD3 seed-46 rerun; stable, but validation selection falls back to an idle checkpoint and the annual PBM/EBM gap is exactly zero.",
        },
        {
            "family": "DDPG-family",
            "agent": "TD3",
            "steps": 20_000,
            "seed": 47,
            "path": REPO_ROOT / "results" / "_tmp_cigre_td3_20k_seed47_postfix" / "report.json",
            "note": "Post-fix 20k MG-CIGRE TD3 seed-47 rerun; stable but EBM-selected dispatch is materially stronger, producing a negative PBM gap.",
        },
        {
            "family": "DDPG-family",
            "agent": "DDPG",
            "steps": 20_000,
            "seed": 44,
            "path": REPO_ROOT / "results" / "_tmp_cigre_ddpg_20k_postfix" / "report.json",
            "note": "Post-fix 20k MG-CIGRE DDPG scout with validation selection and action regularization; stable but effectively flat under PBM evaluation.",
        },
        {
            "family": "SAC-family",
            "agent": "TQC",
            "steps": 20_000,
            "seed": 44,
            "path": REPO_ROOT / "results" / "_tmp_cigre_tqc_20k_postfix" / "report.json",
            "note": "Post-fix 20k MG-CIGRE TQC scout with validation selection and action regularization; stable and no longer NaN, but near-idle and effectively flat.",
        },
        {
            "family": "SAC-family",
            "agent": "TQC",
            "steps": 20_000,
            "seed": 47,
            "path": REPO_ROOT / "results" / "_tmp_cigre_tqc_20k_seed47_postfix" / "report.json",
            "note": "Post-fix 20k MG-CIGRE TQC seed-47 rerun; stable and clearly PBM-favorable, showing that non-SAC positive support can appear on CIGRE but is seed-sensitive.",
        },
    ]

    columns = [
        "family",
        "agent",
        "steps",
        "seed",
        "pbm_cost",
        "ebm_cost",
        "gap_pct",
        "threshold_met",
        "pbm_validation_failed_windows",
        "ebm_validation_failed_windows",
        "pbm_clip_ratio_pct",
        "ebm_clip_ratio_pct",
        "pbm_p_actual_abs_p95_kw",
        "ebm_p_actual_abs_p95_kw",
        "note",
        "source",
    ]
    rows: list[dict] = []
    for spec in scout_specs:
        path = Path(spec["path"])
        report = _load_json(path)
        pbm_training_meta = dict(report.get("pbm_training_meta", {}))
        ebm_training_meta = dict(report.get("ebm_training_meta", {}))
        pbm_selected = dict(pbm_training_meta.get("selected_checkpoint", {}))
        ebm_selected = dict(ebm_training_meta.get("selected_checkpoint", {}))
        rows.append(
            {
                "family": spec["family"],
                "agent": spec["agent"],
                "steps": int(spec["steps"]),
                "seed": int(spec["seed"]),
                "pbm_cost": float(report["pbm_cost"]),
                "ebm_cost": float(report["ebm_cost"]),
                "gap_pct": float(report["gap_pct"]),
                "threshold_met": bool(report.get("threshold_met", False)),
                "pbm_validation_failed_windows": int(pbm_selected.get("validation_failed_windows", 0)),
                "ebm_validation_failed_windows": int(ebm_selected.get("validation_failed_windows", 0)),
                "pbm_clip_ratio_pct": 100.0 * float(report["pbm_clip_ratio"]),
                "ebm_clip_ratio_pct": 100.0 * float(report["ebm_clip_ratio"]),
                "pbm_p_actual_abs_p95_kw": float(report["pbm_p_actual_abs_p95_kw"]),
                "ebm_p_actual_abs_p95_kw": float(report["ebm_p_actual_abs_p95_kw"]),
                "note": spec["note"],
                "source": str(path.relative_to(REPO_ROOT)),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def compute_cigre_sac_budget_trend_summary() -> pd.DataFrame:
    budget_specs = [
        {
            "steps": 5_000,
            "scope": "postfix_short_budget",
            "path": REPO_ROOT / "results" / "_tmp_cigre_sac_5k_valsel_postfix" / "report.json",
            "note": "Very short post-fix sanity check; budget is too small to produce a positive PBM edge.",
        },
        {
            "steps": 20_000,
            "scope": "postfix_short_budget",
            "path": REPO_ROOT / "results" / "_tmp_cigre_sac_20k_valsel_postfix" / "report.json",
            "note": "Short-budget post-fix sanity check; PBM-favorable direction already clears the 1.5% threshold.",
        },
        {
            "steps": 30_000,
            "scope": "postfix_mid_budget",
            "path": REPO_ROOT / "results" / "_tmp_cigre_sac_30k_valsel_postfix" / "report.json",
            "note": "Mid-budget post-fix sanity check; PBM-favorable direction remains clear and approaches the released package seed-44 gap.",
        },
        {
            "steps": 910_000,
            "scope": "released_seed44_reference",
            "path": REPO_ROOT / "results" / "paper" / "cigre" / "seed44" / "report.json",
            "note": "Released seed-44 reference from the main MG-CIGRE package.",
        },
    ]

    rows: list[dict] = []
    for spec in budget_specs:
        path = Path(spec["path"])
        report = _load_json(path)
        pbm_training_meta = dict(report.get("pbm_training_meta", {}))
        ebm_training_meta = dict(report.get("ebm_training_meta", {}))
        pbm_selected = dict(pbm_training_meta.get("selected_checkpoint", {}))
        ebm_selected = dict(ebm_training_meta.get("selected_checkpoint", {}))
        rows.append(
            {
                "agent": str(report.get("agent", "sac")).upper(),
                "seed": int(report["seed"]),
                "steps": int(spec["steps"]),
                "scope": str(spec["scope"]),
                "pbm_cost": float(report["pbm_cost"]),
                "ebm_cost": float(report["ebm_cost"]),
                "gap_pct": float(report["gap_pct"]),
                "threshold_met": bool(report.get("threshold_met", False)),
                "pbm_validation_failed_windows": int(pbm_selected.get("validation_failed_windows", 0)),
                "ebm_validation_failed_windows": int(ebm_selected.get("validation_failed_windows", 0)),
                "pbm_clip_ratio_pct": 100.0 * float(report["pbm_clip_ratio"]),
                "ebm_clip_ratio_pct": 100.0 * float(report["ebm_clip_ratio"]),
                "pbm_p_actual_abs_p95_kw": float(report["pbm_p_actual_abs_p95_kw"]),
                "ebm_p_actual_abs_p95_kw": float(report["ebm_p_actual_abs_p95_kw"]),
                "note": str(spec["note"]),
                "source": str(path.relative_to(REPO_ROOT)),
            }
        )
    return pd.DataFrame(rows)


def compute_res_mechanism_summary() -> pd.DataFrame:
    rows: list[dict] = []
    for seed_dir in sorted((PAPER_DIR / "res").glob("seed*")):
        seed = int(seed_dir.name.replace("seed", ""))
        summary_df = pd.read_csv(seed_dir / "summary.csv")
        for _, row in summary_df.iterrows():
            label = str(row["label"])
            method = "pbm" if "pbm" in label.lower() else "ebm"
            timeseries_path = next(seed_dir.glob(f"{method}_seed*_timeseries.csv"))
            ts = pd.read_csv(timeseries_path)
            upper_price_threshold = float(ts["Price_per_kWh"].quantile(0.75))
            rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "clip_ratio_pct": 100.0 * float(row["clip_ratio"]),
                    "mean_abs_action_gap_norm": float(np.mean(np.abs(ts["Battery_Action_Raw"] - ts["Battery_Action_Applied"]))),
                    "p95_cmd_kw": float(row["p_cmd_abs_p95_kw"]),
                    "p95_applied_kw": float(row["p_actual_abs_p95_kw"]),
                    "annual_battery_loss_kwh": float(row["battery_loss_kwh_sum"]),
                    "soc_ge_90_pct": 100.0 * float((ts["SOC"] >= 0.90).mean()),
                    "upper_price_discharge_share_pct": 100.0 * float(
                        ((ts["Price_per_kWh"] >= upper_price_threshold) & (ts["Battery_Power_kW"] > 0.05)).mean()
                    ),
                }
            )
    detail_df = pd.DataFrame(rows).sort_values(["method", "seed"]).reset_index(drop=True)
    detail_df.to_csv(OUTPUT_DIR / "res_mechanism_detail.csv", index=False)
    summary = (
        detail_df.groupby("method", as_index=False)
        .agg(
            seed_count=("seed", "count"),
            mean_clip_ratio_pct=("clip_ratio_pct", "mean"),
            mean_abs_action_gap_norm=("mean_abs_action_gap_norm", "mean"),
            mean_p95_cmd_kw=("p95_cmd_kw", "mean"),
            mean_p95_applied_kw=("p95_applied_kw", "mean"),
            mean_annual_battery_loss_kwh=("annual_battery_loss_kwh", "mean"),
            mean_soc_ge_90_pct=("soc_ge_90_pct", "mean"),
            mean_upper_price_discharge_share_pct=("upper_price_discharge_share_pct", "mean"),
        )
        .sort_values("method")
        .reset_index(drop=True)
    )
    return summary


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

    res_validation_detail_df, res_validation_summary_df = compute_res_validation_convergence()
    res_validation_detail_df.to_csv(OUTPUT_DIR / "res_validation_convergence_detail.csv", index=False)
    res_validation_summary_df.to_csv(OUTPUT_DIR / "res_validation_convergence_summary.csv", index=False)

    cigre_sac_detail_df, cigre_sac_summary_df = compute_cigre_sac_convergence()
    cigre_sac_detail_df.to_csv(OUTPUT_DIR / "cigre_sac_convergence_detail.csv", index=False)
    cigre_sac_summary_df.to_csv(OUTPUT_DIR / "cigre_sac_convergence_summary.csv", index=False)

    agent_df = compute_agent_appendix_summary()
    agent_df.to_csv(OUTPUT_DIR / "agent_appendix_summary.csv", index=False)

    cigre_agent_df = compute_cigre_agent_scout_summary()
    cigre_agent_df.to_csv(OUTPUT_DIR / "cigre_agent_scout_summary.csv", index=False)

    cigre_sac_budget_df = compute_cigre_sac_budget_trend_summary()
    cigre_sac_budget_df.to_csv(OUTPUT_DIR / "cigre_sac_budget_trend_summary.csv", index=False)

    res_mechanism_df = compute_res_mechanism_summary()
    res_mechanism_df.to_csv(OUTPUT_DIR / "res_mechanism_summary.csv", index=False)

    print("Saved appendix analysis files:")
    print(f"  {OUTPUT_DIR / 'case_statistical_inference.csv'}")
    print(f"  {OUTPUT_DIR / 'case_statistical_inference.json'}")
    print(f"  {OUTPUT_DIR / 'res_fixed_window_summary.csv'}")
    print(f"  {OUTPUT_DIR / 'res_validation_convergence_detail.csv'}")
    print(f"  {OUTPUT_DIR / 'res_validation_convergence_summary.csv'}")
    print(f"  {OUTPUT_DIR / 'cigre_sac_convergence_detail.csv'}")
    print(f"  {OUTPUT_DIR / 'cigre_sac_convergence_summary.csv'}")
    print(f"  {OUTPUT_DIR / 'agent_appendix_summary.csv'}")
    print(f"  {OUTPUT_DIR / 'cigre_agent_scout_summary.csv'}")
    print(f"  {OUTPUT_DIR / 'cigre_sac_budget_trend_summary.csv'}")
    print(f"  {OUTPUT_DIR / 'res_mechanism_detail.csv'}")
    print(f"  {OUTPUT_DIR / 'res_mechanism_summary.csv'}")


if __name__ == "__main__":
    main()
