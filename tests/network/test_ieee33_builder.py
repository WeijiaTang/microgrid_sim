from microgrid_sim.network.builders.ieee33 import build_ieee33_network


def test_ieee33_builder_returns_network_with_der_slots():
    net = build_ieee33_network()
    assert len(net.bus) >= 33
    assert len(net.line) > 0
    assert len(net.ext_grid) > 0
    assert len(net.storage) > 0
    assert len(net.sgen) >= 3
    assert net.user_metadata["benchmark"] == "IEEE 33-bus"
    assert net.user_metadata["storage_role"] == "distribution_scale_dess"
    assert net.user_metadata["storage_energy_mwh"] == 0.896
    assert net.user_metadata["storage_power_mw"] == 0.5
    assert float(net.storage.iloc[0]["max_e_mwh"]) == 0.896
    assert net.user_metadata["battery_bus_index"] == 32
    assert net.user_metadata["pv_bus_indices"] == (24, 29, 30)
    assert net.user_metadata["line_limit_ka"] == 0.25
    assert float(net.line["max_i_ka"].min()) == 0.25
