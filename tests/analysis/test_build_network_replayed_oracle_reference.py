from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.build_network_replayed_oracle_reference import _canonical_case_key, normalized_action_for_requested_power


class _FakeBattery:
    def __init__(self, bounds: tuple[float, float]):
        self._bounds = bounds

    def power_command_bounds(self, dt: float):
        del dt
        return self._bounds


class _FakeConfig:
    dt_seconds = 900.0


class _FakeEnv:
    def __init__(self, bounds: tuple[float, float]):
        self.battery = _FakeBattery(bounds)
        self.config = _FakeConfig()


def test_canonical_case_key_accepts_reference_case_key_alias():
    assert _canonical_case_key("ieee33_network") == "ieee33"
    assert _canonical_case_key("CIGRE European LV") == "cigre"


def test_normalized_action_for_requested_power_uses_current_env_bounds():
    env = _FakeEnv(bounds=(-200.0, 100.0))

    assert normalized_action_for_requested_power(env, 50.0) == 0.5
    assert normalized_action_for_requested_power(env, -50.0) == -0.25
    assert normalized_action_for_requested_power(env, 500.0) == 1.0
    assert normalized_action_for_requested_power(env, -500.0) == -1.0
