#!/usr/bin/env python3
# Ref: docs/spec/task.md (Task-ID: SPEC-SAFE-OFFLINE-RL-001)

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from microgrid_sim.training.offline_dataset import export_trajectory_glob_to_offline_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export trajectory CSV files into a unified offline RL dataset.")
    parser.add_argument(
        "--trajectory-glob",
        type=str,
        required=True,
        help="Glob pattern relative to the repo root, e.g. results/myrun/trajectories/*.csv",
    )
    parser.add_argument(
        "--controller-source",
        type=str,
        required=True,
        help="Source tag for the dataset, e.g. sac_plain / heuristic / oracle",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory relative to the repo root.",
    )
    parser.add_argument("--case", type=str, default="", help="Optional explicit case tag override for all matched trajectories.")
    parser.add_argument("--regime", type=str, default="", help="Optional explicit regime tag override for all matched trajectories.")
    parser.add_argument(
        "--battery-model",
        type=str,
        default="",
        help="Optional explicit battery_model tag override for all matched trajectories.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    base_dir = Path.cwd()
    trajectory_paths = sorted(base_dir.glob(str(args.trajectory_glob)))
    if not trajectory_paths:
        trajectory_paths = sorted(REPO_ROOT.glob(str(args.trajectory_glob)))
    if not trajectory_paths:
        raise FileNotFoundError(f"No trajectory CSVs matched glob: {args.trajectory_glob}")
    output_dir_arg = Path(str(args.output_dir))
    output_dir = output_dir_arg if output_dir_arg.is_absolute() else (base_dir / output_dir_arg)
    combined, metadata_rows = export_trajectory_glob_to_offline_dataset(
        trajectory_paths,
        output_dir=output_dir,
        controller_source=str(args.controller_source),
        case=(str(args.case).strip() or None),
        regime=(str(args.regime).strip() or None),
        battery_model=(str(args.battery_model).strip() or None),
    )
    print(f"Matched trajectories: {len(trajectory_paths)}")
    print(f"Combined transitions: {len(combined)}")
    print(f"Saved offline dataset directory: {output_dir}")
    if metadata_rows:
        print(
            "Sources: "
            + ", ".join(
                f"{row.case}/{row.regime}/{row.battery_model}:{row.transition_count}"
                for row in metadata_rows
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
