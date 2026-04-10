"""Builder for a modified IEEE 33-bus distribution network."""

from __future__ import annotations

import pandapower as pp
import pandapower.networks as pn

IEEE33_DESS_POWER_MW = 0.5
IEEE33_DESS_ENERGY_MWH = 1.0
IEEE33_BATTERY_BUS = 32
IEEE33_PV_BUSES = (24, 29, 30)
IEEE33_PV_DISTRIBUTION_WEIGHTS = (0.15, 0.30, 0.55)
IEEE33_LINE_LIMIT_MAIN_KA = 0.23
IEEE33_LINE_LIMIT_LATERAL_KA = 0.16
IEEE33_LINE_LIMIT_TAIL_KA = 0.11
IEEE33_HEAVY_DISTRESSED_LOAD_BUSES = {28, 29, 30, 31, 32}
IEEE33_MODERATE_DISTRESSED_LOAD_BUSES = {23, 24, 25, 26, 27}


def _assign_ieee33_thermal_limits(net) -> None:
    """Add study-oriented line ampacity assumptions for congestion analysis."""

    for idx in net.line.index:
        from_bus = int(net.line.at[idx, "from_bus"])
        to_bus = int(net.line.at[idx, "to_bus"])
        downstream = max(from_bus, to_bus)
        if downstream >= 28:
            limit = IEEE33_LINE_LIMIT_TAIL_KA
        elif downstream >= 18:
            limit = IEEE33_LINE_LIMIT_LATERAL_KA
        else:
            limit = IEEE33_LINE_LIMIT_MAIN_KA
        net.line.at[idx, "max_i_ka"] = limit


def _stress_ieee33_load_distribution(net) -> None:
    """Shift more demand toward weak downstream buses to expose congestion value."""

    for idx in net.load.index:
        bus = int(net.load.at[idx, "bus"])
        if bus in IEEE33_HEAVY_DISTRESSED_LOAD_BUSES:
            scale = 1.45
        elif bus in IEEE33_MODERATE_DISTRESSED_LOAD_BUSES:
            scale = 1.15
        else:
            continue
        net.load.at[idx, "p_mw"] = float(net.load.at[idx, "p_mw"]) * scale
        net.load.at[idx, "q_mvar"] = float(net.load.at[idx, "q_mvar"]) * scale


def build_modified_ieee33_network():
    """Build a modified IEEE 33-bus network with DER attachment slots."""

    net = pn.case33bw()
    _assign_ieee33_thermal_limits(net)
    _stress_ieee33_load_distribution(net)
    battery_bus = IEEE33_BATTERY_BUS
    pv_buses = IEEE33_PV_BUSES
    storage_idx = pp.create_storage(
        net,
        bus=battery_bus,
        p_mw=0.0,
        max_e_mwh=IEEE33_DESS_ENERGY_MWH,
        soc_percent=50.0,
        min_e_mwh=0.0,
        name="battery_storage",
        max_p_mw=IEEE33_DESS_POWER_MW,
        min_p_mw=-IEEE33_DESS_POWER_MW,
    )
    for idx, bus in enumerate(pv_buses, start=1):
        pp.create_sgen(net, bus=bus, p_mw=0.0, q_mvar=0.0, name=f"pv_injection_{idx}")
    net.user_pf_options = {"init": "auto"}
    net.user_metadata = {
        "benchmark": "Modified IEEE 33-bus",
        "derived_from": "pandapower.networks.case33bw",
        "modifications": [
            "Added study-oriented line thermal limits to replace placeholder ampacity values",
            "Shifted load stress toward mid-to-tail buses with stronger weighting on the deepest branch",
            "Added battery storage slot at bus 33 (0-based index 32)",
            "Added controllable PV injection slots at buses 25, 30, and 31",
        ],
        "battery_bus_index": battery_bus,
        "pv_bus_indices": pv_buses,
        "pv_distribution_weights": IEEE33_PV_DISTRIBUTION_WEIGHTS,
        "storage_index": int(storage_idx),
        "storage_power_mw": IEEE33_DESS_POWER_MW,
        "storage_energy_mwh": IEEE33_DESS_ENERGY_MWH,
        "storage_role": "distribution_scale_dess",
        "line_limit_main_ka": IEEE33_LINE_LIMIT_MAIN_KA,
        "line_limit_lateral_ka": IEEE33_LINE_LIMIT_LATERAL_KA,
        "line_limit_tail_ka": IEEE33_LINE_LIMIT_TAIL_KA,
        "heavy_distressed_load_buses": tuple(sorted(IEEE33_HEAVY_DISTRESSED_LOAD_BUSES)),
        "moderate_distressed_load_buses": tuple(sorted(IEEE33_MODERATE_DISTRESSED_LOAD_BUSES)),
    }
    return net
