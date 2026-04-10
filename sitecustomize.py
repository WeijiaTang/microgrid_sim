"""Project-local path bootstrap for src-layout commands.

This keeps ``uv run python -m microgrid_sim...`` working without relying on the
removed legacy experiment entrypoints.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"

if SRC_ROOT.is_dir():
    src_path = str(SRC_ROOT)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
