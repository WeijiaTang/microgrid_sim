"""Project path helpers for the current paper repository layout."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
NETWORK_DATA_ROOT = DATA_ROOT / "network"
LEGACY_DATA_ROOT = DATA_ROOT / "legacy"
LEGACY_AGGREGATED_DATA_ROOT = LEGACY_DATA_ROOT / "aggregated"
LEGACY_YEARLY_DATA_ROOT = LEGACY_DATA_ROOT / "yearly"
DEFAULT_DATASET_FILES = ("load", "pv", "price")


def _env_data_dir() -> Optional[Path]:
    value = os.getenv("MICROGRID_SIM_DATA_DIR")
    if not value:
        return None
    return Path(value).expanduser().resolve()


def candidate_data_dirs() -> list[Path]:
    candidates: list[Path] = []
    env_path = _env_data_dir()
    if env_path is not None:
        candidates.append(env_path)
    candidates.append(DATA_ROOT)
    return [path.resolve() for path in candidates if path.exists()]


def resolve_dataset_file(data_dir: Path, role: str) -> Optional[Path]:
    from .io.reader import resolve_bundle_files

    resolved = resolve_bundle_files(data_dir, required_roles=(role,))
    return resolved.get(role)


def resolve_data_dir(required_files: Sequence[str] = DEFAULT_DATASET_FILES) -> Optional[Path]:
    for path in candidate_data_dirs():
        if all(resolve_dataset_file(path, role) is not None for role in required_files):
            return path
    for path in candidate_data_dirs():
        return path
    return None
