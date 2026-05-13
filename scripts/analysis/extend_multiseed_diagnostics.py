#!/usr/bin/env python3
"""Run and aggregate paper-facing multiseed diagnostics for CIGRE and IEEE33."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "scripts" / "analysis" / "short_cross_fidelity_probe.py"

DEFAULT_OLD_SEEDS = [42, 52, 62]
DEFAULT_NEW_SEEDS = [72, 82, 92]

CIGRE_EXISTING_DETAIL = REPO_ROOT / "results" / "diagnostics" / "cigre_key_protocols_multiseed_detail.csv"
IEEE_EXISTING_DETAIL = REPO_ROOT / "results" / "diagnostics" / "ieee33_symmetric_20k_multiseed_detail_20260423.csv"
GATE_DWELL_THRESHOLD = 0.05


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    output_dir_name: str
    cli_args: list[str]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extend paper-facing multiseed diagnostics.")
    parser.add_argument(
        "--mode",
        choices=("run", "aggregate", "all"),
        default="all",
        help="Run new seed experiments, aggregate diagnostics, or do both.",
    )
    parser.add_argument(
        "--new-seeds",
        type=str,
        default="72,82,92",
        help="Comma-separated new seeds to add on top of the existing 42/52/62 set.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="results/diagnostics/seed_extension_20260427",
        help="Directory for new run folders and aggregated 6-seed CSVs.",
    )
    parser.add_argument(
        "--overwrite-canonical",
        action="store_true",
        help="Also overwrite the canonical paper-facing CIGRE summary/detail CSVs after aggregation.",
    )
    return parser


def _parse_seeds(raw: str) -> list[int]:
    seeds: list[int] = []
    for token in str(raw).split(","):
        token = token.strip()
        if token:
            seeds.append(int(token))
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def _seed_tag(seeds: list[int]) -> str:
    return "_".join(str(seed) for seed in seeds)


def _common_probe_args(*, seeds: list[int], case: str) -> list[str]:
    args = [
        "--cases",
        case,
        "--regimes",
        "network_stress",
        "--reward-profile",
        "paper_balanced",
        "--agent",
        "sac",
        "--train-steps",
        "20000",
        "--days",
        "30",
        "--train-year",
        "2023",
        "--eval-year",
        "2024",
        "--train-episode-days",
        "30",
        "--eval-days",
        "30",
        "--train-random-start-within-year",
        "--action-smoothing-coef",
        "0.5",
        "--action-max-delta",
        "0.1",
        "--action-rate-penalty",
        "0.05",
        "--battery-feasibility-aware",
        "--battery-infeasible-penalty",
        "-1.0",
        "--symmetric-battery-action",
        "--seeds",
        ",".join(str(seed) for seed in seeds),
        "--device",
        "cpu",
    ]
    if case == "cigre":
        args.extend(
            [
                "--eval-steps",
                "2880",
                "--train-validation-days",
                "7",
                "--train-validation-offset-days-within-year",
                "0,91,182,273",
                "--train-validation-checkpoint-every",
                "2500",
                "--train-validation-metric",
                "health_objective",
                "--train-validation-terminal-penalty-weight",
                "1.0",
                "--train-validation-boundary-dwell-weight",
                "20000",
                "--train-validation-infeasible-dwell-weight",
                "20000",
                "--train-validation-peak-reserve-weight",
                "0.0",
                "--train-validation-peak-discharge-limit-threshold",
                "0.25",
            ]
        )
    elif case == "ieee33":
        args.extend(
            [
                "--eval-full-horizon",
                "--train-validation-days",
                "7",
                "--train-validation-offset-days-within-year",
                "0,91,182,273",
                "--train-validation-checkpoint-every",
                "5000",
                "--train-validation-metric",
                "health_objective",
                "--train-validation-terminal-penalty-weight",
                "1.0",
                "--train-validation-boundary-dwell-weight",
                "20000",
                "--train-validation-infeasible-dwell-weight",
                "20000",
                "--train-validation-peak-reserve-weight",
                "10000.0",
                "--train-validation-peak-discharge-limit-threshold",
                "0.25",
            ]
        )
    else:
        raise ValueError(f"Unsupported case '{case}'.")
    return args


def build_experiments(*, seeds: list[int], output_root: Path) -> list[ExperimentSpec]:
    seed_tag = _seed_tag(seeds)
    cigre_common = _common_probe_args(seeds=seeds, case="cigre")
    ieee_common = _common_probe_args(seeds=seeds, case="ieee33")
    return [
        ExperimentSpec(
            name="cigre_simple_dualtest",
            output_dir_name=f"cigre_simple_dualtest_{seed_tag}",
            cli_args=cigre_common + ["--train-models", "simple", "--test-models", "simple,thevenin_full"],
        ),
        ExperimentSpec(
            name="cigre_mixed_full",
            output_dir_name=f"cigre_mixed_full_{seed_tag}",
            cli_args=cigre_common + ["--train-models", "simple+thevenin", "--test-models", "thevenin_full"],
        ),
        ExperimentSpec(
            name="cigre_full_full",
            output_dir_name=f"cigre_full_full_{seed_tag}",
            cli_args=cigre_common + ["--train-models", "thevenin_full", "--test-models", "thevenin_full"],
        ),
        ExperimentSpec(
            name="ieee33_simple",
            output_dir_name=f"ieee33_simple_20k_{seed_tag}",
            cli_args=ieee_common + ["--train-models", "simple", "--test-models", "simple"],
        ),
        ExperimentSpec(
            name="ieee33_rint_only",
            output_dir_name=f"ieee33_rint_only_20k_{seed_tag}",
            cli_args=ieee_common + ["--train-models", "thevenin_rint_only", "--test-models", "thevenin_rint_only"],
        ),
        ExperimentSpec(
            name="ieee33_full",
            output_dir_name=f"ieee33_full_20k_{seed_tag}",
            cli_args=ieee_common + ["--train-models", "thevenin_full", "--test-models", "thevenin_full"],
        ),
    ]


def _resolve_summary_csv(output_root: Path, spec: ExperimentSpec) -> Path | None:
    primary = output_root / spec.output_dir_name / "summary.csv"
    if primary.exists():
        return primary
    recovered = output_root / spec.output_dir_name / "eval_recovered" / "summary.csv"
    if recovered.exists():
        return recovered
    return None


def _gate_mask_from_columns(
    *,
    savings_vs_none: pd.Series,
    upper_dwell: pd.Series,
    lower_dwell: pd.Series,
    infeasible_dwell: pd.Series,
) -> pd.Series:
    """Repository-wide reasonableness gate from the documented <0.05 dwell rule."""

    return (
        (savings_vs_none.astype(float) > 0.0)
        & (upper_dwell.astype(float) < GATE_DWELL_THRESHOLD)
        & (lower_dwell.astype(float) < GATE_DWELL_THRESHOLD)
        & (infeasible_dwell.astype(float) < GATE_DWELL_THRESHOLD)
    )


def run_experiments(*, specs: list[ExperimentSpec], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        output_dir = output_root / spec.output_dir_name
        cmd = [sys.executable, str(RUNNER), *spec.cli_args, "--output-dir", str(output_dir)]
        print(f"\n[run] {spec.name}")
        print(" ".join(cmd))
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def _cigre_protocol(row: pd.Series) -> str | None:
    train_model = str(row["train_model"]).strip().lower()
    test_model = str(row["test_model"]).strip().lower()
    if train_model == "simple" and test_model == "simple":
        return "simple_to_simple"
    if train_model == "simple" and test_model in {"thevenin", "thevenin_full"}:
        return "simple_to_full"
    if train_model == "simple+thevenin" and test_model in {"thevenin", "thevenin_full"}:
        return "mixed_simple+thevenin_to_full"
    if train_model in {"thevenin", "thevenin_full"} and test_model in {"thevenin", "thevenin_full"}:
        return "full_to_full"
    return None


def _cigre_baseline_objective(existing_detail: pd.DataFrame) -> float:
    sample = existing_detail.iloc[0]
    return float(sample["objective_cost"]) + float(sample["savings_vs_none"])


def _aggregate_cigre(*, specs: list[ExperimentSpec], output_root: Path, new_seeds: set[int]) -> tuple[Path, Path]:
    existing_detail = pd.read_csv(CIGRE_EXISTING_DETAIL)
    baseline_objective = _cigre_baseline_objective(existing_detail)
    new_frames: list[pd.DataFrame] = []
    for spec in specs:
        if not spec.name.startswith("cigre_"):
            continue
        summary_csv = _resolve_summary_csv(output_root=output_root, spec=spec)
        if summary_csv is None:
            print(f"[skip] missing CIGRE summary: {output_root / spec.output_dir_name}")
            continue
        frame = pd.read_csv(summary_csv)
        if frame.empty:
            continue
        frame["protocol"] = frame.apply(_cigre_protocol, axis=1)
        frame = frame[frame["protocol"].notna()].copy()
        if frame.empty:
            continue
        frame = frame[frame["seed"].astype(int).isin(new_seeds)].copy()
        if frame.empty:
            continue
        frame["objective_cost"] = frame["final_cumulative_objective_cost"].astype(float)
        frame["savings_vs_none"] = baseline_objective - frame["objective_cost"]
        frame["final_soc"] = frame["final_soc"].astype(float)
        frame["throughput_kwh"] = frame["total_battery_throughput_kwh"].astype(float)
        frame["upper_dwell"] = frame["soc_upper_dwell_fraction"].astype(float)
        frame["lower_dwell"] = frame["soc_lower_dwell_fraction"].astype(float)
        frame["infeasible_dwell"] = frame["infeasible_action_dwell_fraction"].astype(float)
        frame["reasonable_dispatch_gate"] = _gate_mask_from_columns(
            savings_vs_none=frame["savings_vs_none"],
            upper_dwell=frame["upper_dwell"],
            lower_dwell=frame["lower_dwell"],
            infeasible_dwell=frame["infeasible_dwell"],
        )
        new_frames.append(
            frame[
                [
                    "protocol",
                    "seed",
                    "objective_cost",
                    "savings_vs_none",
                    "final_soc",
                    "throughput_kwh",
                    "upper_dwell",
                    "lower_dwell",
                    "infeasible_dwell",
                    "reasonable_dispatch_gate",
                ]
            ].copy()
        )

    if not new_frames:
        raise ValueError("No new CIGRE rows were collected from the new-seed summaries.")

    combined = pd.concat([existing_detail, *new_frames], ignore_index=True)
    combined["seed"] = combined["seed"].astype(int)
    combined = combined.drop_duplicates(subset=["protocol", "seed"], keep="last")
    combined = combined.sort_values(["protocol", "seed"], kind="stable").reset_index(drop=True)
    total_seeds = int(combined["seed"].nunique())

    summary = (
        combined.groupby("protocol", dropna=False)
        .agg(
            seeds=("seed", "count"),
            gate_passes=("reasonable_dispatch_gate", lambda s: int(pd.Series(s).astype(bool).sum())),
            mean_objective_cost=("objective_cost", "mean"),
            mean_savings_vs_none=("savings_vs_none", "mean"),
            min_savings_vs_none=("savings_vs_none", "min"),
            max_savings_vs_none=("savings_vs_none", "max"),
            mean_final_soc=("final_soc", "mean"),
            mean_throughput_kwh=("throughput_kwh", "mean"),
            mean_upper_dwell=("upper_dwell", "mean"),
            mean_lower_dwell=("lower_dwell", "mean"),
            mean_infeasible_dwell=("infeasible_dwell", "mean"),
        )
        .reset_index()
    )
    summary["gate_pass_rate"] = summary["gate_passes"].astype(float) / summary["seeds"].astype(float)
    order = [
        "full_to_full",
        "mixed_simple+thevenin_to_full",
        "simple_to_full",
        "simple_to_simple",
    ]
    summary["protocol"] = pd.Categorical(summary["protocol"], categories=order, ordered=True)
    summary = summary.sort_values("protocol", kind="stable").reset_index(drop=True)
    summary["protocol"] = summary["protocol"].astype(str)

    detail_path = output_root / f"cigre_key_protocols_multiseed_detail_{total_seeds}seeds_20260428.csv"
    summary_path = output_root / f"cigre_key_protocols_multiseed_summary_{total_seeds}seeds_20260428.csv"
    combined.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    return detail_path, summary_path


def _ieee_family(row: pd.Series) -> str | None:
    train_model = str(row["train_model"]).strip().lower()
    test_model = str(row["test_model"]).strip().lower()
    if train_model != test_model:
        return None
    if train_model == "simple":
        return "simple"
    if train_model == "thevenin_rint_only":
        return "rint_only"
    if train_model == "thevenin_full":
        return "full"
    return None


def _ieee_baseline_objective(existing_detail: pd.DataFrame) -> float:
    sample = existing_detail.iloc[0]
    return float(sample["final_cumulative_objective_cost"]) + float(sample["obj_savings_vs_none"])


def _aggregate_ieee(*, specs: list[ExperimentSpec], output_root: Path, new_seeds: set[int]) -> tuple[Path, Path]:
    existing_detail = pd.read_csv(IEEE_EXISTING_DETAIL)
    baseline_objective = _ieee_baseline_objective(existing_detail)
    new_frames: list[pd.DataFrame] = []
    for spec in specs:
        if not spec.name.startswith("ieee33_"):
            continue
        summary_csv = _resolve_summary_csv(output_root=output_root, spec=spec)
        if summary_csv is None:
            print(f"[skip] missing IEEE33 summary: {output_root / spec.output_dir_name}")
            continue
        frame = pd.read_csv(summary_csv)
        if frame.empty:
            continue
        frame = frame[frame["seed"].astype(int).isin(new_seeds)].copy()
        if frame.empty:
            continue
        frame["family"] = frame.apply(_ieee_family, axis=1)
        frame = frame[frame["family"].notna()].copy()
        if frame.empty:
            continue
        frame["source_path"] = summary_csv.as_posix()
        frame["obj_savings_vs_none"] = baseline_objective - frame["final_cumulative_objective_cost"].astype(float)
        frame["gate_pass"] = _gate_mask_from_columns(
            savings_vs_none=frame["obj_savings_vs_none"],
            upper_dwell=frame["soc_upper_dwell_fraction"],
            lower_dwell=frame["soc_lower_dwell_fraction"],
            infeasible_dwell=frame["infeasible_action_dwell_fraction"],
        ).astype(int)
        new_frames.append(frame[existing_detail.columns].copy())

    if not new_frames:
        raise ValueError("No new IEEE33 rows were collected from the new-seed summaries.")

    combined = pd.concat([existing_detail, *new_frames], ignore_index=True)
    combined["seed"] = combined["seed"].astype(int)
    combined = combined.drop_duplicates(subset=["family", "seed"], keep="last")
    combined = combined.sort_values(["family", "seed"], kind="stable").reset_index(drop=True)
    total_seeds = int(combined["seed"].nunique())

    family_summary = (
        combined.groupby("family", dropna=False)
        .agg(
            seeds=("seed", "count"),
            gate_passes=("gate_pass", "sum"),
            mean_objective_cost=("final_cumulative_objective_cost", "mean"),
            mean_obj_savings_vs_none=("obj_savings_vs_none", "mean"),
            min_obj_savings_vs_none=("obj_savings_vs_none", "min"),
            max_obj_savings_vs_none=("obj_savings_vs_none", "max"),
            mean_final_soc=("final_soc", "mean"),
            mean_throughput_kwh=("total_battery_throughput_kwh", "mean"),
            mean_upper_dwell=("soc_upper_dwell_fraction", "mean"),
            mean_lower_dwell=("soc_lower_dwell_fraction", "mean"),
            mean_infeasible_dwell=("infeasible_action_dwell_fraction", "mean"),
            mean_peak_limit_ratio=("peak_price_mean_discharge_limit_ratio", "mean"),
            mean_peak_low_limit_dwell=("peak_price_low_discharge_limit_dwell_fraction", "mean"),
        )
        .reset_index()
    )
    family_summary["gate_pass_rate"] = family_summary["gate_passes"].astype(float) / family_summary["seeds"].astype(float)
    order = ["simple", "rint_only", "full"]
    family_summary["family"] = pd.Categorical(family_summary["family"], categories=order, ordered=True)
    family_summary = family_summary.sort_values("family", kind="stable").reset_index(drop=True)
    family_summary["family"] = family_summary["family"].astype(str)

    detail_path = output_root / f"ieee33_symmetric_20k_multiseed_detail_{total_seeds}seeds_20260428.csv"
    summary_path = output_root / f"ieee33_symmetric_20k_family_summary_{total_seeds}seeds_20260428.csv"
    combined.to_csv(detail_path, index=False)
    family_summary.to_csv(summary_path, index=False)
    return detail_path, summary_path


def aggregate_outputs(*, specs: list[ExperimentSpec], output_root: Path, new_seeds: list[int], overwrite_canonical: bool) -> dict[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    new_seed_set = set(int(seed) for seed in new_seeds)
    cigre_detail, cigre_summary = _aggregate_cigre(specs=specs, output_root=output_root, new_seeds=new_seed_set)
    ieee_detail, ieee_summary = _aggregate_ieee(specs=specs, output_root=output_root, new_seeds=new_seed_set)

    written = {
        "cigre_detail": cigre_detail,
        "cigre_summary": cigre_summary,
        "ieee_detail": ieee_detail,
        "ieee_summary": ieee_summary,
    }
    if overwrite_canonical:
        pd.read_csv(cigre_detail).to_csv(CIGRE_EXISTING_DETAIL, index=False)
        pd.read_csv(cigre_summary).to_csv(REPO_ROOT / "results" / "diagnostics" / "cigre_key_protocols_multiseed_summary.csv", index=False)
        written["cigre_detail_canonical"] = CIGRE_EXISTING_DETAIL
        written["cigre_summary_canonical"] = REPO_ROOT / "results" / "diagnostics" / "cigre_key_protocols_multiseed_summary.csv"
    return written


def main() -> int:
    args = build_parser().parse_args()
    new_seeds = _parse_seeds(args.new_seeds)
    output_root = REPO_ROOT / str(args.output_root)
    specs = build_experiments(seeds=new_seeds, output_root=output_root)

    if args.mode in {"run", "all"}:
        run_experiments(specs=specs, output_root=output_root)
    if args.mode in {"aggregate", "all"}:
        written = aggregate_outputs(
            specs=specs,
            output_root=output_root,
            new_seeds=new_seeds,
            overwrite_canonical=bool(args.overwrite_canonical),
        )
        for label, path in written.items():
            print(f"[write] {label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
