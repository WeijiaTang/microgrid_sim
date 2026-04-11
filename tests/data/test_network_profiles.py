from pathlib import Path

import numpy as np

from microgrid_sim.cases import CIGREEuropeanLVConfig, IEEE33Config
from microgrid_sim.data.network_profiles import load_network_profiles
from microgrid_sim.envs.network_microgrid import NetworkMicrogridEnv
from microgrid_sim.io.reader import read_numeric_series
from microgrid_sim.paths import NETWORK_DATA_ROOT


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
    assert float(stress.price[stressed_hours].mean()) > float(base.price[stressed_hours].mean()) * 1.05


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


def test_repo_bundles_canonical_network_case_datasets():
    expected = {
        "cigre_eu_lv": {"load_peak": 120_000.0, "pv_peak": 18_000.0},
        "ieee33": {"load_peak": 4_000_000.0, "pv_peak": 450_000.0},
    }

    for case_dirname, peaks in expected.items():
        case_dir = NETWORK_DATA_ROOT / case_dirname
        assert case_dir.is_dir(), f"Missing canonical network dataset directory: {case_dir}"

        load = read_numeric_series(case_dir / "load.csv")
        pv = read_numeric_series(case_dir / "pv.csv")
        price = read_numeric_series(case_dir / "price.csv")

        assert len(load) == 365 * 24
        assert len(pv) == 365 * 24
        assert len(price) == 365 * 24
        assert np.isclose(float(load.max()), peaks["load_peak"])
        assert np.isclose(float(pv.max()), peaks["pv_peak"])
        assert float(price.min()) >= 0.0
