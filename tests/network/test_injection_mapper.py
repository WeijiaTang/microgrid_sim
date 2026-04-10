import numpy as np

from microgrid_sim.network.adapters.injection_mapper import apply_power_injections, initialize_injection_state
from microgrid_sim.network.builders.ieee33_modified import build_modified_ieee33_network


def test_apply_power_injections_preserves_load_reactive_ratio():
    net = build_modified_ieee33_network()
    state = initialize_injection_state(net)

    original_ratios = np.divide(
        net.load["q_mvar"].to_numpy(dtype=float),
        np.maximum(net.load["p_mw"].to_numpy(dtype=float), 1e-9),
    )
    apply_power_injections(net, state, load_w=2_000_000.0, pv_w=200_000.0, battery_power_w=100_000.0)

    updated_ratios = np.divide(
        net.load["q_mvar"].to_numpy(dtype=float),
        np.maximum(net.load["p_mw"].to_numpy(dtype=float), 1e-9),
    )
    assert np.allclose(updated_ratios, original_ratios, rtol=1e-6, atol=1e-9)

