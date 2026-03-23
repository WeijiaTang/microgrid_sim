"""Grid search for residential D9 demand-charge tuning."""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_EXPERIMENTS = PROJECT_ROOT / "experiments" / "run_experiments.py"


GROUP_TARGET = "pbm_gt_ebm_gt_idle"
GROUP_PBM_IDLE = "pbm_gt_idle"
GROUP_NEAR_TIE = "pbm_near_ebm"
GROUP_OTHER = "other"


def parse_float_grid(raw: str) -> list[float]:
    values = [part.strip() for part in str(raw).split(",") if part.strip()]
    return [float(value) for value in values]


def parse_extra_args(raw: str) -> list[str]:
    values = [part.strip() for part in str(raw).split(";;") if part.strip()]
    args: list[str] = []
    for value in values:
        args.extend(value.split())
    return args


def classify_report(report: dict, near_tie_gap_pct: float) -> str:
    if bool(report.get("target_hit", False)):
        return GROUP_TARGET
    if bool(report.get("pbm_beats_idle", False)) and bool(report.get("ebm_beats_idle", False)):
        if abs(float(report.get("gap_pct", 0.0))) <= float(near_tie_gap_pct):
            return GROUP_NEAR_TIE
        return GROUP_PBM_IDLE
    if abs(float(report.get("gap_pct", 0.0))) <= float(near_tie_gap_pct):
        return GROUP_NEAR_TIE
    return GROUP_OTHER


def group_priority(group: str) -> int:
    return {
        GROUP_TARGET: 3,
        GROUP_PBM_IDLE: 2,
        GROUP_NEAR_TIE: 1,
        GROUP_OTHER: 0,
    }.get(group, -1)


def candidate_sort_tuple(report: dict) -> tuple:
    return (
        group_priority(str(report.get("group_label", GROUP_OTHER))),
        int(bool(report.get("pbm_beats_ebm", False))),
        int(bool(report.get("pbm_beats_idle", False))),
        int(bool(report.get("ebm_beats_idle", False))),
        float(report.get("pbm_margin_vs_ebm", 0.0)),
        float(report.get("pbm_margin_vs_idle", 0.0)),
        -abs(float(report.get("gap_pct", 0.0))),
        -float(report.get("pbm_cost", float("inf"))),
    )


def within_group_sort_tuple(report: dict) -> tuple:
    group = str(report.get("group_label", GROUP_OTHER))
    pbm_margin_vs_ebm = float(report.get("pbm_margin_vs_ebm", 0.0))
    pbm_margin_vs_idle = float(report.get("pbm_margin_vs_idle", 0.0))
    abs_gap_pct = abs(float(report.get("gap_pct", 0.0)))
    pbm_cost = float(report.get("pbm_cost", float("inf")))
    if group == GROUP_TARGET:
        return (pbm_margin_vs_ebm, pbm_margin_vs_idle, -pbm_cost, -abs_gap_pct)
    if group == GROUP_PBM_IDLE:
        return (pbm_margin_vs_ebm, pbm_margin_vs_idle, -abs_gap_pct, -pbm_cost)
    if group == GROUP_NEAR_TIE:
        return (pbm_margin_vs_idle, pbm_margin_vs_ebm, -abs_gap_pct, -pbm_cost)
    return (pbm_margin_vs_ebm, pbm_margin_vs_idle, -abs_gap_pct, -pbm_cost)


def retrain_sort_tuple(report: dict) -> tuple:
    return (
        group_priority(str(report.get("group_label", GROUP_OTHER))),
        float(report.get("pbm_margin_vs_ebm", 0.0)),
        float(report.get("pbm_margin_vs_idle", 0.0)),
        -abs(float(report.get("gap_pct", 0.0))),
        -float(report.get("pbm_cost", float("inf"))),
    )


def retrain_reason(report: dict) -> str:
    group = str(report.get("group_label", GROUP_OTHER))
    if group == GROUP_TARGET:
        return "PBM > EBM > idle; retrain for stability"
    if group == GROUP_PBM_IDLE:
        return "PBM beats idle but trails EBM; retrain near target"
    if group == GROUP_NEAR_TIE:
        return "PBM ~= EBM; retrain to confirm a flip"
    return "Outside target zone; keep as fallback control"


def run_combo(
    python_exe: str,
    combo_id: str,
    output_root: Path,
    models_root: Path,
    steps: int,
    seed: int,
    monthly_demand_charge_per_kw: float,
    monthly_demand_charge_threshold_kw: float,
    pbm_r_int_scale: float,
    pbm_step_multiplier: float,
    extra_args: list[str],
    near_tie_gap_pct: float,
    skip_completed: bool,
) -> dict:
    run_output_dir = output_root / combo_id
    run_models_dir = models_root / combo_id
    run_output_dir.mkdir(parents=True, exist_ok=True)
    run_models_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_output_dir / "report.json"
    if skip_completed and report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        command = [
            python_exe,
            str(RUN_EXPERIMENTS),
            "residential-d4-run",
            "--steps",
            str(int(steps)),
            "--seed",
            str(int(seed)),
            "--monthly-demand-charge-per-kw",
            str(float(monthly_demand_charge_per_kw)),
            "--monthly-demand-charge-threshold-kw",
            str(float(monthly_demand_charge_threshold_kw)),
            "--pbm-r-int-scale",
            str(float(pbm_r_int_scale)),
            "--pbm-step-multiplier",
            str(float(pbm_step_multiplier)),
            "--output-dir",
            str(run_output_dir),
            "--models-dir",
            str(run_models_dir),
            *extra_args,
        ]
        completed = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True)
        (run_output_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
        (run_output_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(f"Combo {combo_id} failed with exit code {completed.returncode}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
    report["combo_id"] = combo_id
    report["output_dir"] = str(run_output_dir)
    report["models_dir"] = str(run_models_dir)
    report["pbm_beats_ebm"] = bool(report["pbm_cost"] < report["ebm_cost"])
    report["pbm_beats_idle"] = bool(report["pbm_cost"] < report["idle_cost"])
    report["ebm_beats_idle"] = bool(report["ebm_cost"] < report["idle_cost"])
    report["target_hit"] = bool(report["pbm_beats_ebm"] and report["pbm_beats_idle"] and report["ebm_beats_idle"])
    report["pbm_margin_vs_ebm"] = float(report["ebm_cost"] - report["pbm_cost"])
    report["pbm_margin_vs_idle"] = float(report["idle_cost"] - report["pbm_cost"])
    report["abs_gap_pct"] = abs(float(report.get("gap_pct", 0.0)))
    report["group_label"] = classify_report(report, near_tie_gap_pct=near_tie_gap_pct)
    report["group_priority"] = group_priority(report["group_label"])
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Grid search monthly demand charge and PBM resistance scale for residential D9")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--output-root", type=str, default=str(PROJECT_ROOT / "results" / "residential_d9_grid"))
    parser.add_argument("--models-root", type=str, default=str(PROJECT_ROOT / "models" / "residential_d9_grid"))
    parser.add_argument("--monthly-demand-charge-per-kw-grid", type=str, default="24,30,35")
    parser.add_argument("--monthly-demand-charge-threshold-kw-grid", type=str, default="15,16,18")
    parser.add_argument("--pbm-r-int-scale-grid", type=str, default="10,12,14")
    parser.add_argument("--pbm-step-multiplier-grid", type=str, default="1,2,4")
    parser.add_argument("--near-tie-gap-pct", type=float, default=1.0)
    parser.add_argument("--top-k-per-group", type=int, default=5)
    parser.add_argument("--extra-args", type=str, default="")
    parser.add_argument("--skip-completed", action="store_true")
    args = parser.parse_args(argv)

    output_root = Path(args.output_root)
    models_root = Path(args.models_root)
    output_root.mkdir(parents=True, exist_ok=True)
    models_root.mkdir(parents=True, exist_ok=True)

    charge_grid = parse_float_grid(args.monthly_demand_charge_per_kw_grid)
    threshold_grid = parse_float_grid(args.monthly_demand_charge_threshold_kw_grid)
    rint_grid = parse_float_grid(args.pbm_r_int_scale_grid)
    step_multiplier_grid = parse_float_grid(args.pbm_step_multiplier_grid)
    extra_args = parse_extra_args(args.extra_args)

    results: list[dict] = []
    combo_iter = itertools.product(charge_grid, threshold_grid, rint_grid, step_multiplier_grid)
    for index, (charge, threshold, rint_scale, step_multiplier) in enumerate(combo_iter, start=1):
        combo_id = f"c{index:02d}_dc{charge:g}_th{threshold:g}_r{rint_scale:g}_ps{step_multiplier:g}"
        print(f"[{index}] Running {combo_id} ...")
        report = run_combo(
            python_exe=sys.executable,
            combo_id=combo_id,
            output_root=output_root,
            models_root=models_root,
            steps=args.steps,
            seed=args.seed,
            monthly_demand_charge_per_kw=charge,
            monthly_demand_charge_threshold_kw=threshold,
            pbm_r_int_scale=rint_scale,
            pbm_step_multiplier=step_multiplier,
            extra_args=extra_args,
            near_tie_gap_pct=args.near_tie_gap_pct,
            skip_completed=bool(args.skip_completed),
        )
        results.append(report)
        print(json.dumps({
            "combo_id": combo_id,
            "group_label": report["group_label"],
            "pbm_cost": report["pbm_cost"],
            "ebm_cost": report["ebm_cost"],
            "idle_cost": report["idle_cost"],
            "gap_pct": report["gap_pct"],
            "target_hit": report["target_hit"],
        }, ensure_ascii=False))

    ranking = sorted(results, key=candidate_sort_tuple, reverse=True)
    results_df = pd.DataFrame(ranking)
    results_df.to_csv(output_root / "grid_results.csv", index=False)

    grouped_rows = {
        GROUP_TARGET: sorted([row for row in results if row["group_label"] == GROUP_TARGET], key=within_group_sort_tuple, reverse=True),
        GROUP_PBM_IDLE: sorted([row for row in results if row["group_label"] == GROUP_PBM_IDLE], key=within_group_sort_tuple, reverse=True),
        GROUP_NEAR_TIE: sorted([row for row in results if row["group_label"] == GROUP_NEAR_TIE], key=within_group_sort_tuple, reverse=True),
        GROUP_OTHER: sorted([row for row in results if row["group_label"] == GROUP_OTHER], key=within_group_sort_tuple, reverse=True),
    }

    grouped_dfs: dict[str, pd.DataFrame] = {}
    for label, rows in grouped_rows.items():
        ranked_rows = []
        for group_rank, row in enumerate(rows, start=1):
            ranked_row = dict(row)
            ranked_row["group_rank"] = group_rank
            ranked_row["retrain_reason"] = retrain_reason(ranked_row)
            ranked_rows.append(ranked_row)
        grouped_dfs[label] = pd.DataFrame(ranked_rows)

    target_df = grouped_dfs[GROUP_TARGET]
    pbm_idle_df = grouped_dfs[GROUP_PBM_IDLE]
    near_tie_df = grouped_dfs[GROUP_NEAR_TIE]
    other_df = grouped_dfs[GROUP_OTHER]

    target_df.to_csv(output_root / "grid_group_target.csv", index=False)
    pbm_idle_df.to_csv(output_root / "grid_group_pbm_idle.csv", index=False)
    near_tie_df.to_csv(output_root / "grid_group_near_tie.csv", index=False)
    other_df.to_csv(output_root / "grid_group_other.csv", index=False)

    top_k = max(int(args.top_k_per_group), 1)
    candidate_rows: list[dict] = []
    for label in (GROUP_TARGET, GROUP_PBM_IDLE, GROUP_NEAR_TIE):
        candidate_rows.extend(grouped_rows[label][:top_k])
    candidate_rows = sorted(candidate_rows, key=retrain_sort_tuple, reverse=True)
    retrain_ranked_rows = []
    for retrain_rank, row in enumerate(candidate_rows, start=1):
        retrain_row = dict(row)
        retrain_row["retrain_rank"] = retrain_rank
        retrain_row["retrain_reason"] = retrain_reason(retrain_row)
        retrain_ranked_rows.append(retrain_row)
    candidates_df = pd.DataFrame(retrain_ranked_rows)
    candidates_df.to_csv(output_root / "grid_candidates.csv", index=False)
    candidates_df.to_csv(output_root / "grid_retrain_candidates.csv", index=False)

    grouped_top = {
        GROUP_TARGET: target_df.head(top_k).to_dict(orient="records"),
        GROUP_PBM_IDLE: pbm_idle_df.head(top_k).to_dict(orient="records"),
        GROUP_NEAR_TIE: near_tie_df.head(top_k).to_dict(orient="records"),
    }
    summary = {
        "steps": int(args.steps),
        "seed": int(args.seed),
        "near_tie_gap_pct": float(args.near_tie_gap_pct),
        "pbm_step_multiplier_grid": step_multiplier_grid,
        "num_combos": int(len(results)),
        "num_target_hits": int(len(target_df)),
        "num_pbm_gt_idle": int(len(pbm_idle_df)),
        "num_pbm_near_ebm": int(len(near_tie_df)),
        "best_combo": ranking[0] if ranking else None,
        "best_retrain_candidates": candidates_df.head(top_k).to_dict(orient="records"),
        "grouped_top_candidates": grouped_top,
    }
    (output_root / "grid_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
