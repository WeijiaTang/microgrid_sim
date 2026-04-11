from pathlib import Path

import numpy as np

from microgrid_sim.cases import CIGREEuropeanLVConfig, IEEE33Config
from microgrid_sim.data.network_profiles import load_network_profiles
from microgrid_sim.envs.network_microgrid import NetworkMicrogridEnv


def test_network_profile_regimes_shift_load_and_pv_levels():
    base_cfg = CIGREEuropeanLVConfig(simulation_days=1, seed=42, regime="base")
    high_load_cfg = CIGREEuropeanLVConfig(simulation_days=1, seed=42, regime="high_load")
    high_pv_cfg = CIGREEuropeanLVConfig(simulation_days=1, seed=42, regime="high_pv")

    base = load_network_profiles(base_cfg, total_hours=24)
    high_load = load_network_profiles(high_load_cfg, total_hours=24)
    high_pv = load_network_profiles(high_pv_cfg, total_hours=24)

    assert float(high_load.load_w.mean()) > float(base.load_w.mean())
    assert float(high_pv.pv_w.mean()) > float(base.pv_w.mean())


def test_tight_soc_regime_resets_battery_in_low_soc_band():
    env = NetworkMicrogridEnv(CIGREEuropeanLVConfig(simulation_days=1, seed=7, regime="tight_soc"))
    _, info = env.reset(seed=7)
    assert 0.15 <= float(info["soc"]) <= 0.30


def test_ieee33_network_stress_is_stronger_than_base_in_load_and_price():
    base_cfg = IEEE33Config(simulation_days=1, seed=42, regime="base")
    stress_cfg = IEEE33Config(simulation_days=1, seed=42, regime="network_stress")

    base = load_network_profiles(base_cfg, total_hours=24)
    stress = load_network_profiles(stress_cfg, total_hours=24)
    stressed_hours = slice(16, 24)

    assert float(stress.load_w.mean()) > float(base.load_w.mean()) * 1.30
    assert float(stress.pv_w.mean()) < float(base.pv_w.mean()) * 0.85
    assert float(stress.price[stressed_hours].mean()) > float(base.price[stressed_hours].mean()) * 1.10


def test_network_profiles_can_load_from_canonical_case_directory(tmp_path: Path):
    case_dir = tmp_path / "network" / "ieee33"
    case_dir.mkdir(parents=True)
    (case_dir / "load.csv").write_text("100\n200\n300\n", encoding="utf-8")
    (case_dir / "pv.csv").write_text("10\n20\n30\n", encoding="utf-8")
    (case_dir / "price.csv").write_text("0.1\n0.2\n0.3\n", encoding="utf-8")

    cfg = IEEE33Config(simulation_days=1, seed=42, regime="base", data_dir=str(tmp_path), tou_price_spread_multiplier=1.0)
    profiles = load_network_profiles(cfg, total_hours=6)

    assert np.allclose(profiles.load_w, np.array([100, 200, 300, 100, 200, 300], dtype=float))
    assert np.allclose(profiles.pv_w, np.array([10, 20, 30, 10, 20, 30], dtype=float))
    assert np.allclose(profiles.price, np.array([0.1, 0.2, 0.3, 0.1, 0.2, 0.3], dtype=float))
