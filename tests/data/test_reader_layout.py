from pathlib import Path

import numpy as np

from microgrid_sim.io.reader import read_case_dataset


def test_read_case_dataset_uses_legacy_aggregated_layout(tmp_path: Path):
    case_dir = tmp_path / "legacy" / "aggregated" / "mg_res"
    case_dir.mkdir(parents=True)
    (case_dir / "load.csv").write_text("1\n2\n3\n", encoding="utf-8")
    (case_dir / "pv.csv").write_text("4\n5\n6\n", encoding="utf-8")
    (case_dir / "price.csv").write_text("0.1\n0.2\n0.3\n", encoding="utf-8")

    bundle = read_case_dataset(tmp_path, case_key="mg_res", total_hours=5)

    assert np.allclose(bundle["load"], np.array([1, 2, 3, 1, 2], dtype=float))
    assert np.allclose(bundle["pv"], np.array([4, 5, 6, 4, 5], dtype=float))
    assert np.allclose(bundle["price"], np.array([0.1, 0.2, 0.3, 0.1, 0.2], dtype=float))


def test_read_case_dataset_uses_legacy_optional_cigre_roles(tmp_path: Path):
    case_dir = tmp_path / "legacy" / "aggregated" / "mg_cigre"
    case_dir.mkdir(parents=True)
    (case_dir / "load.csv").write_text("10\n20\n", encoding="utf-8")
    (case_dir / "pv.csv").write_text("1\n2\n", encoding="utf-8")
    (case_dir / "price.csv").write_text("0.5\n0.6\n", encoding="utf-8")
    (case_dir / "other.csv").write_text("3\n4\n", encoding="utf-8")

    bundle = read_case_dataset(tmp_path, case_key="mg_cigre", total_hours=4)

    assert np.allclose(bundle["other"], np.array([3, 4, 3, 4], dtype=float))
