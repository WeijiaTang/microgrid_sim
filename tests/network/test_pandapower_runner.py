import pandapower

from microgrid_sim.network.adapters.pandapower_runner import run_power_flow
from microgrid_sim.network.builders.cigre_lv import build_cigre_european_lv_network


def test_pandapower_is_importable():
    assert pandapower.__version__


def test_run_power_flow_returns_convergence_metadata():
    net = build_cigre_european_lv_network()
    result = run_power_flow(net)
    assert "converged" in result
    assert result["converged"] is True
