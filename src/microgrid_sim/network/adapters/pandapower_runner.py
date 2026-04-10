"""Centralized pandapower power flow execution."""

from __future__ import annotations

from typing import Any

import pandapower as pp
from pandapower.auxiliary import LoadflowNotConverged


def run_power_flow(net) -> dict[str, Any]:
    """Run a power flow and return convergence metadata."""

    options = dict(getattr(net, "user_pf_options", {}) or {})
    # Keep solver behavior quiet and deterministic in environments where numba is not installed.
    options.setdefault("numba", False)
    try:
        pp.runpp(net, **options)
    except LoadflowNotConverged as exc:
        return {
            "converged": False,
            "failed": True,
            "failure_reason": str(exc),
        }
    return {
        "converged": bool(net.converged),
        "failed": False,
        "failure_reason": "",
    }
