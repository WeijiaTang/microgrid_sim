# Ref: docs/spec/task.md (Task-ID: SPEC-SAFE-OFFLINE-RL-001)

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


OFFLINE_REDUCED_OBS_COLUMNS: tuple[str, ...] = (
    "soc",
    "temperature_c",
    "load_w",
    "pv_w",
    "price",
    "grid_import_mw",
    "grid_export_mw",
    "min_bus_voltage_pu",
    "max_bus_voltage_pu",
    "max_line_loading_pct",
    "mean_line_loading_pct",
    "battery_power_mw",
    "battery_action_applied",
    "battery_action_feasible_low",
    "battery_action_feasible_high",
    "battery_charge_fraction_feasible",
    "battery_discharge_fraction_feasible",
    "battery_action_infeasible_gap",
    "battery_internal_clip_gap_w",
    "discharge_limit_ratio",
    "rule_based_action_hint",
    "rule_guidance_mix",
    "terminal_soc_target",
    "terminal_soc_tolerance",
    "terminal_soc_deviation",
    "peak_reserve_shortfall",
    "p_max_w",
    "p_max_trend_w",
)


REQUIRED_TRAJECTORY_COLUMNS: tuple[str, ...] = (
    "step",
    "reward",
    "soc",
    "battery_action_applied",
    "grid_import_mw",
    "min_bus_voltage_pu",
    "max_line_loading_pct",
    "cumulative_cost",
    "terminal_soc_penalty",
    "battery_action_infeasible_gap",
)


@dataclass(frozen=True)
class OfflineDatasetMetadata:
    case: str
    regime: str
    battery_model: str
    controller_source: str
    source_path: str
    row_count: int
    transition_count: int
    observation_columns: tuple[str, ...]


def _require_columns(frame: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Trajectory is missing required columns: {missing}")


def _infer_case_from_path(path: Path) -> str:
    text = path.as_posix().lower()
    if "ieee33" in text:
        return "ieee33"
    if "cigre" in text:
        return "cigre"
    return "unknown"


def _infer_regime_from_path(path: Path) -> str:
    tokens = path.stem.lower().split("_")
    candidates = ("network_stress", "high_load", "high_pv", "tight_soc", "base")
    joined = "_".join(tokens)
    for candidate in candidates:
        if candidate in joined:
            return candidate
    return "unknown"


def _infer_battery_model_from_path(path: Path) -> str:
    lowered = path.stem.lower()
    if "thevenin_full" in lowered:
        return "thevenin_full"
    if "thevenin_loss_only" in lowered:
        return "thevenin_loss_only"
    if "thevenin" in lowered:
        return "thevenin"
    if "simple" in lowered:
        return "simple"
    return "unknown"


def _json_array_from_row(row: pd.Series, columns: tuple[str, ...]) -> str:
    values = [float(row[column]) if pd.notna(row[column]) else 0.0 for column in columns]
    return json.dumps(values, separators=(",", ":"))


def trajectory_to_offline_transitions(
    trajectory: pd.DataFrame,
    *,
    case: str,
    regime: str,
    battery_model: str,
    controller_source: str,
    observation_columns: tuple[str, ...] = OFFLINE_REDUCED_OBS_COLUMNS,
) -> pd.DataFrame:
    _require_columns(trajectory, REQUIRED_TRAJECTORY_COLUMNS)
    missing_obs = [column for column in observation_columns if column not in trajectory.columns]
    if missing_obs:
        raise ValueError(f"Trajectory is missing observation columns: {missing_obs}")
    if trajectory.empty:
        return pd.DataFrame(
            columns=[
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
        )

    rows: list[dict[str, object]] = []
    total_rows = int(len(trajectory))
    for idx in range(total_rows - 1):
        current = trajectory.iloc[idx]
        nxt = trajectory.iloc[idx + 1]
        rows.append(
            {
                "case": str(case),
                "regime": str(regime),
                "battery_model": str(battery_model),
                "controller_source": str(controller_source),
                "step": int(current["step"]),
                "obs": _json_array_from_row(current, observation_columns),
                "next_obs": _json_array_from_row(nxt, observation_columns),
                "action": float(current["battery_action_applied"]),
                "reward": float(current["reward"]),
                "done": int(idx == total_rows - 2),
                "soc": float(current["soc"]),
                "grid_import_mw": float(current["grid_import_mw"]),
                "min_bus_voltage_pu": float(current["min_bus_voltage_pu"]),
                "max_line_loading_pct": float(current["max_line_loading_pct"]),
                "cumulative_cost": float(current["cumulative_cost"]),
                "terminal_penalty": float(current["terminal_soc_penalty"]),
                "infeasible_gap": float(current["battery_action_infeasible_gap"]),
                "source_row_count": int(total_rows),
            }
        )
    return pd.DataFrame(rows)


def export_trajectory_csv_to_offline_dataset(
    trajectory_csv: Path,
    *,
    output_dir: Path,
    controller_source: str,
    case: str | None = None,
    regime: str | None = None,
    battery_model: str | None = None,
    observation_columns: tuple[str, ...] = OFFLINE_REDUCED_OBS_COLUMNS,
) -> tuple[pd.DataFrame, OfflineDatasetMetadata]:
    frame = pd.read_csv(trajectory_csv)
    resolved_case = str(case or _infer_case_from_path(trajectory_csv))
    resolved_regime = str(regime or _infer_regime_from_path(trajectory_csv))
    resolved_battery_model = str(battery_model or _infer_battery_model_from_path(trajectory_csv))
    transitions = trajectory_to_offline_transitions(
        frame,
        case=resolved_case,
        regime=resolved_regime,
        battery_model=resolved_battery_model,
        controller_source=controller_source,
        observation_columns=observation_columns,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / f"{trajectory_csv.stem}_offline.csv"
    output_json = output_dir / f"{trajectory_csv.stem}_offline.meta.json"
    transitions.to_csv(output_csv, index=False)
    metadata = OfflineDatasetMetadata(
        case=resolved_case,
        regime=resolved_regime,
        battery_model=resolved_battery_model,
        controller_source=str(controller_source),
        source_path=str(trajectory_csv),
        row_count=int(len(frame)),
        transition_count=int(len(transitions)),
        observation_columns=tuple(observation_columns),
    )
    output_json.write_text(json.dumps(metadata.__dict__, indent=2), encoding="utf-8")
    return transitions, metadata


def export_trajectory_glob_to_offline_dataset(
    trajectory_paths: Iterable[Path],
    *,
    output_dir: Path,
    controller_source: str,
    case: str | None = None,
    regime: str | None = None,
    battery_model: str | None = None,
    observation_columns: tuple[str, ...] = OFFLINE_REDUCED_OBS_COLUMNS,
) -> tuple[pd.DataFrame, list[OfflineDatasetMetadata]]:
    combined_frames: list[pd.DataFrame] = []
    metadata_rows: list[OfflineDatasetMetadata] = []
    for trajectory_path in trajectory_paths:
        transitions, metadata = export_trajectory_csv_to_offline_dataset(
            trajectory_path,
            output_dir=output_dir,
            controller_source=controller_source,
            case=case,
            regime=regime,
            battery_model=battery_model,
            observation_columns=observation_columns,
        )
        combined_frames.append(transitions)
        metadata_rows.append(metadata)
    combined = pd.concat(combined_frames, ignore_index=True) if combined_frames else pd.DataFrame()
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        combined.to_csv(output_dir / "combined_offline_dataset.csv", index=False)
        (output_dir / "combined_offline_dataset.meta.json").write_text(
            json.dumps([row.__dict__ for row in metadata_rows], indent=2),
            encoding="utf-8",
        )
    return combined, metadata_rows


def decode_observation_json(encoded: str) -> np.ndarray:
    return np.asarray(json.loads(str(encoded)), dtype=np.float32)
