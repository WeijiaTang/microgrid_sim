from microgrid_sim.network.builders.cigre_lv import build_cigre_european_lv_network


def test_cigre_builder_returns_network_with_required_elements():
    net = build_cigre_european_lv_network()
    assert len(net.bus) > 0
    assert len(net.line) > 0
    assert len(net.ext_grid) > 0
    assert len(net.storage) > 0
    assert len(net.sgen) >= 3
    assert net.user_metadata["benchmark"] == "CIGRE European LV"
    assert net.user_metadata["storage_role"] == "community_scale_lv_bess"
    assert net.user_metadata["storage_energy_mwh"] == 0.2
    assert net.user_metadata["storage_power_mw"] == 0.1
    assert float(net.storage.iloc[0]["max_e_mwh"]) == 0.2
    assert net.user_metadata["battery_bus_index"] == 19
    assert net.user_metadata["pv_bus_indices"] == (12, 16, 18)
