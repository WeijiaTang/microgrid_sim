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
    assert net.user_metadata["storage_energy_mwh"] == 0.3584
    assert net.user_metadata["storage_power_mw"] == 0.2
    assert float(net.storage.iloc[0]["max_e_mwh"]) == 0.3584
    assert net.user_metadata["battery_bus_index"] == 12
    assert net.user_metadata["pv_bus_indices"] == (12, 16, 18)


def test_cigre_builder_honors_configured_storage_location_and_size():
    net = build_cigre_european_lv_network(
        battery_bus_name="Bus R15",
        pv_bus_names=("Bus R11", "Bus R17"),
        storage_power_mw=0.2,
        storage_energy_mwh=0.3,
    )

    assert net.user_metadata["battery_bus_index"] == 16
    assert net.user_metadata["pv_bus_indices"] == (12, 18)
    assert net.user_metadata["storage_power_mw"] == 0.2
    assert net.user_metadata["storage_energy_mwh"] == 0.3
    assert float(net.storage.iloc[0]["max_p_mw"]) == 0.2
    assert float(net.storage.iloc[0]["max_e_mwh"]) == 0.3
