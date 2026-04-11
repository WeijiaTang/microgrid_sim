"""Builder for the IEEE 33-bus distribution network."""

from __future__ import annotations

import pandapower as pp
import pandapower.networks as pn

IEEE33_DESS_POWER_MW = 0.5
IEEE33_DESS_ENERGY_MWH = 0.896
IEEE33_BATTERY_BUS = 32
IEEE33_PV_BUSES = (24, 29, 30)
IEEE33_PV_DISTRIBUTION_WEIGHTS = (0.15, 0.30, 0.55)
IEEE33_LINE_LIMIT_KA = 0.25


def _assign_uniform_ieee33_thermal_limits(net) -> None:
    """Replace placeholder ampacity values with a uniform screening limit."""

    net.line.loc[:, "max_i_ka"] = IEEE33_LINE_LIMIT_KA


def build_ieee33_network():
    """Build the standard IEEE 33-bus feeder with DER attachment slots."""

    net = pn.case33bw()
    _assign_uniform_ieee33_thermal_limits(net)
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
        "benchmark": "IEEE 33-bus",
        "derived_from": "pandapower.networks.case33bw",
        "modifications": [
            "Replaced placeholder line ampacity values with a uniform 0.25 kA thermal screening limit",
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
        "line_limit_ka": IEEE33_LINE_LIMIT_KA,
    }
    return net
