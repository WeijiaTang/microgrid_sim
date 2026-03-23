"""Simple external dataset reader for PV, load, and price series."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..paths import PROJECT_ROOT


DATA_ROLE_FILENAMES: dict[str, tuple[str, ...]] = {
    "load": ("load.csv", "Load_Power.csv", "Load_Power_2024.csv", "Load_Power_2023.csv"),
    "pv": (
        "pv.csv",
        "PV_Power_yearly.csv",
        "PV_Power_yearly_los_angeles_2024.csv",
        "PV_Power_yearly_los_angeles_2023.csv",
    ),
    "price": ("price.csv", "TOU_Price.csv", "price_profile.csv"),
    "wind": ("wind.csv", "WT_Power.csv", "Wind_Power.csv", "wind_power.csv"),
    "other": ("other.csv", "P_other.csv", "Other_Power.csv"),
    "net": ("net.csv", "P_net.csv", "Net_Load.csv"),
}

CASE_DIR_CANDIDATES: dict[str, tuple[str, ...]] = {
    "mg_res": ("mg_res", "MG-RES", "MGRES", "residential", "res"),
    "mg_cigre": ("mg_cigre", "MG-CIGRE", "CIGRE", "cigre"),
}


def _find_numeric_column(frame: pd.DataFrame) -> pd.Series:
    for column in frame.columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().any():
            return values.dropna()
    raise ValueError("No numeric column found in dataset file")


def read_numeric_series(path: str | Path) -> np.ndarray:
    """Read a single time series from CSV.

    Supports either:
    - a single numeric column
    - multiple columns where one column is numeric (e.g. ``timestamp,value``)
    """

    csv_path = Path(path)
    try:
        frame = pd.read_csv(csv_path, header=None)
    except (pd.errors.ParserError, MemoryError):
        frame = pd.read_csv(csv_path, header=None, engine="python")
    if frame.empty:
        raise ValueError(f"Dataset file is empty: {csv_path}")
    values = _find_numeric_column(frame)
    if values.empty:
        raise ValueError(f"Dataset file has no usable numeric values: {csv_path}")
    return values.to_numpy(dtype=float)


def resolve_bundle_files(data_dir: str | Path, required_roles: tuple[str, ...] = ("load", "pv", "price")) -> dict[str, Path]:
    """Resolve dataset files by logical role from a user-supplied directory."""

    root = Path(data_dir)
    resolved: dict[str, Path] = {}
    for role in required_roles:
        for filename in DATA_ROLE_FILENAMES.get(role, (f"{role}.csv",)):
            candidate = root / filename
            if candidate.exists():
                resolved[role] = candidate.resolve()
                break
    return resolved


def resolve_case_dir(data_root: str | Path, case_key: str) -> Path:
    """Resolve the directory holding datasets for a specific paper case."""

    root = Path(data_root).resolve()
    for dirname in CASE_DIR_CANDIDATES.get(case_key, (case_key,)):
        candidate = root / dirname
        if candidate.exists() and candidate.is_dir():
            return candidate
    return root


def _optional_case_dirs(data_root: str | Path, case_key: str, primary_case_dir: Path) -> list[Path]:
    """Return fallback directories for case-specific optional roles.

    This supports a common private-data workflow where users keep closed
    ``load/pv/price`` files outside the repository, while the repository stores
    only open/reconstructed auxiliary files such as ``other.csv`` or ``net.csv``.
    """

    candidates: list[Path] = []
    root = Path(data_root).resolve()

    for dirname in CASE_DIR_CANDIDATES.get(case_key, (case_key,)):
        candidate = (root / dirname).resolve()
        if candidate.exists() and candidate.is_dir() and candidate != primary_case_dir:
            candidates.append(candidate)

    repo_case_dir = (PROJECT_ROOT / "data" / case_key).resolve()
    if repo_case_dir.exists() and repo_case_dir.is_dir() and repo_case_dir != primary_case_dir:
        candidates.append(repo_case_dir)

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            unique.append(candidate)
            seen.add(key)
    return unique


def _trim_or_tile(values: np.ndarray, total_hours: int) -> np.ndarray:
    if len(values) == total_hours:
        return values.astype(float, copy=False)
    if len(values) > total_hours:
        return values[:total_hours].astype(float, copy=False)
    repeats = int(np.ceil(total_hours / max(len(values), 1)))
    return np.tile(values, repeats)[:total_hours].astype(float, copy=False)


def read_dataset_bundle(
    data_dir: str | Path,
    total_hours: Optional[int] = None,
    required_roles: tuple[str, ...] = ("load", "pv", "price"),
) -> dict[str, np.ndarray | str]:
    """Read a directory of user datasets into a normalized bundle."""

    files = resolve_bundle_files(data_dir, required_roles=required_roles)
    missing = [role for role in required_roles if role not in files]
    if missing:
        raise FileNotFoundError(
            f"Missing dataset files for roles: {', '.join(missing)} in {Path(data_dir).resolve()}"
        )

    bundle: dict[str, np.ndarray | str] = {
        "source": str(Path(data_dir).resolve()),
        "data_dir": str(Path(data_dir).resolve()),
    }
    for role, path in files.items():
        values = read_numeric_series(path)
        if total_hours is not None:
            values = _trim_or_tile(values, total_hours)
        bundle[role] = values
    return bundle


def read_case_dataset(
    data_root: str | Path,
    case_key: str,
    total_hours: Optional[int] = None,
) -> dict[str, np.ndarray | str]:
    """Read a paper case dataset with case-aware role requirements.

    - ``mg_res`` requires ``load/pv/price``
    - ``mg_cigre`` requires ``load/pv/price`` and either ``other`` or ``net``
      (the optional ``wind`` / ``other`` / ``net`` roles may come from a fallback case directory)
    """

    case_dir = resolve_case_dir(data_root, case_key)
    bundle = read_dataset_bundle(case_dir, total_hours=total_hours, required_roles=("load", "pv", "price"))

    if case_key == "mg_cigre":
        optional = resolve_bundle_files(case_dir, required_roles=("wind", "other", "net"))
        optional_source_dir = case_dir
        if "wind" not in optional and "other" not in optional and "net" not in optional:
            for fallback_dir in _optional_case_dirs(data_root, case_key=case_key, primary_case_dir=case_dir):
                optional = resolve_bundle_files(fallback_dir, required_roles=("wind", "other", "net"))
                if "wind" in optional or "other" in optional or "net" in optional:
                    optional_source_dir = fallback_dir
                    break

        if optional_source_dir != case_dir and "net" in optional:
            filtered = {"net": optional["net"]}
            if "wind" in optional:
                filtered["wind"] = optional["wind"]
            optional = filtered

        if "wind" in optional:
            wind = read_numeric_series(optional["wind"])
            bundle["wind"] = _trim_or_tile(wind, total_hours) if total_hours is not None else wind

        if "other" in optional:
            other = read_numeric_series(optional["other"])
            bundle["other"] = _trim_or_tile(other, total_hours) if total_hours is not None else other
        elif "net" in optional:
            net = read_numeric_series(optional["net"])
            bundle["net"] = _trim_or_tile(net, total_hours) if total_hours is not None else net
        else:
            raise FileNotFoundError(
                "MG-CIGRE reproduction requires either 'other.csv' or 'net.csv' "
                f"in {case_dir} or a fallback case directory"
            )
        bundle["optional_source_dir"] = str(optional_source_dir)

    bundle["case_dir"] = str(case_dir)
    return bundle
