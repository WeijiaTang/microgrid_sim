"""Extract normalized network metrics from a solved pandapower network."""

from __future__ import annotations

import numpy as np


def extract_network_metrics(net) -> dict[str, float]:
    vm_pu = net.res_bus["vm_pu"].to_numpy(dtype=float) if len(net.res_bus) else np.zeros(0, dtype=float)
    line_loading = (
        net.res_line["loading_percent"].to_numpy(dtype=float)
        if len(net.res_line) and "loading_percent" in net.res_line
        else np.zeros(0, dtype=float)
    )
    trafo_loading = (
        net.res_trafo["loading_percent"].to_numpy(dtype=float)
        if len(net.res_trafo) and "loading_percent" in net.res_trafo
        else np.zeros(0, dtype=float)
    )
    ext_grid_p_mw = (
        net.res_ext_grid["p_mw"].to_numpy(dtype=float)
        if len(net.res_ext_grid) and "p_mw" in net.res_ext_grid
        else np.zeros(0, dtype=float)
    )
    line_current_ka = (
        net.res_line["i_from_ka"].to_numpy(dtype=float)
        if len(net.res_line) and "i_from_ka" in net.res_line
        else np.zeros(0, dtype=float)
    )
    return {
        "min_bus_voltage_pu": float(np.min(vm_pu)) if vm_pu.size else 1.0,
        "max_bus_voltage_pu": float(np.max(vm_pu)) if vm_pu.size else 1.0,
        "max_line_loading_pct": float(np.max(line_loading)) if line_loading.size else 0.0,
        "max_line_current_ka": float(np.max(line_current_ka)) if line_current_ka.size else 0.0,
        "mean_line_loading_pct": float(np.mean(line_loading)) if line_loading.size else 0.0,
        "max_transformer_loading_pct": float(np.max(trafo_loading)) if trafo_loading.size else 0.0,
        "slack_active_power_mw": float(np.sum(ext_grid_p_mw)) if ext_grid_p_mw.size else 0.0,
    }
