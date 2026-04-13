"""Builder for the CIGRE European LV benchmark network."""

from __future__ import annotations

import pandapower as pp
import pandapower.networks as pn

CIGRE_LV_BESS_POWER_MW = 0.2
CIGRE_LV_BESS_ENERGY_MWH = 0.3584
CIGRE_LV_BATTERY_BUS_NAME = "Bus R11"
CIGRE_LV_PV_BUS_NAMES = ("Bus R11", "Bus R15", "Bus R17")
CIGRE_LV_PV_DISTRIBUTION_WEIGHTS = (0.20, 0.30, 0.50)


def _find_bus_index_by_name(net, name: str) -> int:
    matches = net.bus.index[net.bus["name"] == name].tolist()
    if not matches:
        raise KeyError(f"Bus named '{name}' not found in CIGRE European LV network")
    return int(matches[0])


def build_cigre_european_lv_network(
    *,
    battery_bus_name: str = CIGRE_LV_BATTERY_BUS_NAME,
    pv_bus_names: tuple[str, ...] = CIGRE_LV_PV_BUS_NAMES,
    storage_power_mw: float = CIGRE_LV_BESS_POWER_MW,
    storage_energy_mwh: float = CIGRE_LV_BESS_ENERGY_MWH,
):
    """Build a modified CIGRE European LV network with battery and PV slots."""

    net = pn.create_cigre_network_lv()
    battery_bus = _find_bus_index_by_name(net, str(battery_bus_name))
    pv_buses = tuple(_find_bus_index_by_name(net, name) for name in tuple(pv_bus_names))
    storage_idx = pp.create_storage(
        net,
        bus=battery_bus,
        p_mw=0.0,
        max_e_mwh=float(storage_energy_mwh),
        soc_percent=50.0,
        min_e_mwh=0.0,
        name="battery_storage",
        max_p_mw=float(storage_power_mw),
        min_p_mw=-float(storage_power_mw),
    )
    for idx, bus in enumerate(pv_buses, start=1):
        pp.create_sgen(net, bus=bus, p_mw=0.0, q_mvar=0.0, name=f"pv_injection_{idx}")
    net.user_pf_options = {"init": "auto"}
    net.user_metadata = {
        "benchmark": "CIGRE European LV",
        "derived_from": "pandapower.networks.create_cigre_network_lv",
        "modifications": [
            f"Configured community BESS at {battery_bus_name}",
            f"Added distributed controllable PV injection slots at {', '.join(pv_bus_names)}",
        ],
        "battery_bus_index": battery_bus,
        "pv_bus_indices": pv_buses,
        "pv_distribution_weights": CIGRE_LV_PV_DISTRIBUTION_WEIGHTS,
        "storage_index": int(storage_idx),
        "storage_power_mw": float(storage_power_mw),
        "storage_energy_mwh": float(storage_energy_mwh),
        "storage_role": "community_scale_lv_bess",
    }
    return net
