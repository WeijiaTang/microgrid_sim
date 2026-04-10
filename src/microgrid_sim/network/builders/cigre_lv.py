"""Builder for the CIGRE European LV benchmark network."""

from __future__ import annotations

import pandapower as pp
import pandapower.networks as pn

CIGRE_LV_BESS_POWER_MW = 0.1
CIGRE_LV_BESS_ENERGY_MWH = 0.2
CIGRE_LV_BATTERY_BUS_NAME = "Bus R18"
CIGRE_LV_PV_BUS_NAMES = ("Bus R11", "Bus R15", "Bus R17")
CIGRE_LV_PV_DISTRIBUTION_WEIGHTS = (0.20, 0.30, 0.50)


def _find_bus_index_by_name(net, name: str) -> int:
    matches = net.bus.index[net.bus["name"] == name].tolist()
    if not matches:
        raise KeyError(f"Bus named '{name}' not found in CIGRE European LV network")
    return int(matches[0])


def build_cigre_european_lv_network():
    """Build a modified CIGRE European LV network with battery and PV slots."""

    net = pn.create_cigre_network_lv()
    battery_bus = _find_bus_index_by_name(net, CIGRE_LV_BATTERY_BUS_NAME)
    pv_buses = tuple(_find_bus_index_by_name(net, name) for name in CIGRE_LV_PV_BUS_NAMES)
    storage_idx = pp.create_storage(
        net,
        bus=battery_bus,
        p_mw=0.0,
        max_e_mwh=CIGRE_LV_BESS_ENERGY_MWH,
        soc_percent=50.0,
        min_e_mwh=0.0,
        name="battery_storage",
        max_p_mw=CIGRE_LV_BESS_POWER_MW,
        min_p_mw=-CIGRE_LV_BESS_POWER_MW,
    )
    for idx, bus in enumerate(pv_buses, start=1):
        pp.create_sgen(net, bus=bus, p_mw=0.0, q_mvar=0.0, name=f"pv_injection_{idx}")
    net.user_pf_options = {"init": "auto"}
    net.user_metadata = {
        "benchmark": "CIGRE European LV",
        "derived_from": "pandapower.networks.create_cigre_network_lv",
        "modifications": [
            "Moved the community BESS to the residential feeder tail at Bus R18",
            "Added distributed controllable PV injection slots at Bus R11, Bus R15, and Bus R17",
        ],
        "battery_bus_index": battery_bus,
        "pv_bus_indices": pv_buses,
        "pv_distribution_weights": CIGRE_LV_PV_DISTRIBUTION_WEIGHTS,
        "storage_index": int(storage_idx),
        "storage_power_mw": CIGRE_LV_BESS_POWER_MW,
        "storage_energy_mwh": CIGRE_LV_BESS_ENERGY_MWH,
        "storage_role": "community_scale_lv_bess",
    }
    return net
