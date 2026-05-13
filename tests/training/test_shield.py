# Ref: docs/spec/task.md (Task-ID: SPEC-SAFE-OFFLINE-RL-001)

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from microgrid_sim.training.shield import BatteryActionShieldConfig, inventory_teacher_action, shield_battery_action


class _DummyBattery:
    def __init__(self, *, soc: float, charge_limit_w: float, discharge_limit_w: float):
        self.soc = float(soc)
        self.params = type(
            "Params",
            (),
            {
                "soc_init": 0.5,
                "soc_min": 0.2,
                "soc_max": 0.9,
                "p_charge_max": 1000.0,
                "p_discharge_max": 1000.0,
                "nominal_energy_wh": 4000.0,
            },
        )()
        self._charge_limit_w = float(charge_limit_w)
        self._discharge_limit_w = float(discharge_limit_w)

    def power_command_bounds(self, dt: float):
        del dt
        return -self._charge_limit_w, self._discharge_limit_w


class _DummyEnv:
    def __init__(
        self,
        *,
        soc: float = 0.5,
        price: float = 0.4,
        current_step: int = 0,
        total_steps: int = 96,
        charge_limit_w: float = 1000.0,
        discharge_limit_w: float = 1000.0,
        terminal_soc_target: float = 0.5,
        terminal_soc_tolerance: float = 0.05,
        peak_price: float = 0.5,
    ):
        self.current_step = int(current_step)
        self.total_steps = int(total_steps)
        self.battery = _DummyBattery(soc=soc, charge_limit_w=charge_limit_w, discharge_limit_w=discharge_limit_w)
        self.config = type(
            "Config",
            (),
            {
                "dt_seconds": 3600.0,
                "battery_params": self.battery.params,
                "terminal_soc_target": float(terminal_soc_target),
                "terminal_soc_tolerance": float(terminal_soc_tolerance),
                "reward": type("Reward", (), {"peak_price": float(peak_price)})(),
            },
        )()
        self._profiles = type("Profiles", (), {"price": np.full(self.total_steps, float(price), dtype=float)})()


def test_shield_is_noop_for_healthy_action():
    env = _DummyEnv(soc=0.55, price=0.4)
    decision = shield_battery_action(
        np.array([0.15], dtype=np.float32),
        unwrapped_env=env,
        config=BatteryActionShieldConfig(),
        action_space_low=np.array([-1.0], dtype=np.float32),
        action_space_high=np.array([1.0], dtype=np.float32),
    )

    assert decision.applied is False
    assert decision.post_action == pytest.approx(0.15)


def test_shield_suppresses_peak_discharge_when_reserve_is_low():
    env = _DummyEnv(soc=0.30, price=0.6, discharge_limit_w=150.0, peak_price=0.5, terminal_soc_target=0.30)
    decision = shield_battery_action(
        np.array([0.8], dtype=np.float32),
        unwrapped_env=env,
        config=BatteryActionShieldConfig(reserve_discharge_min_fraction=0.25),
        action_space_low=np.array([-1.0], dtype=np.float32),
        action_space_high=np.array([1.0], dtype=np.float32),
    )

    assert decision.reserve_active is True
    assert decision.post_action == pytest.approx(0.0)
    assert decision.applied is True


def test_shield_pulls_back_from_upper_boundary():
    env = _DummyEnv(soc=0.89, price=0.4, terminal_soc_target=0.89)
    decision = shield_battery_action(
        np.array([-0.7], dtype=np.float32),
        unwrapped_env=env,
        config=BatteryActionShieldConfig(hard_pullback_action=0.2),
        action_space_low=np.array([-1.0], dtype=np.float32),
        action_space_high=np.array([1.0], dtype=np.float32),
    )

    assert decision.boundary_active is True
    assert decision.post_action >= 0.2


def test_shield_assists_terminal_soc_closure_near_episode_end():
    env = _DummyEnv(
        soc=0.35,
        price=0.4,
        current_step=90,
        total_steps=96,
        charge_limit_w=1000.0,
        discharge_limit_w=1000.0,
        terminal_soc_target=0.55,
        terminal_soc_tolerance=0.01,
    )
    decision = shield_battery_action(
        np.array([0.0], dtype=np.float32),
        unwrapped_env=env,
        config=BatteryActionShieldConfig(terminal_closure_horizon_fraction=0.35),
        action_space_low=np.array([-1.0], dtype=np.float32),
        action_space_high=np.array([1.0], dtype=np.float32),
    )

    assert decision.terminal_active is True
    assert decision.closure_mix > 0.0
    assert decision.post_action < 0.0


def test_inventory_teacher_surfaces_terminal_inventory_guidance():
    env = _DummyEnv(
        soc=0.34,
        price=0.4,
        current_step=92,
        total_steps=96,
        terminal_soc_target=0.56,
        terminal_soc_tolerance=0.01,
    )
    teacher = inventory_teacher_action(
        unwrapped_env=env,
        config=BatteryActionShieldConfig(),
        action_space_low=np.array([-1.0], dtype=np.float32),
        action_space_high=np.array([1.0], dtype=np.float32),
    )

    assert teacher.active is True
    assert teacher.terminal_active is True
    assert teacher.weight > 1.0
    assert float(teacher.action[0]) < 0.0


def test_terminal_closure_is_inactive_outside_tail_horizon_even_with_large_soc_gap():
    env = _DummyEnv(
        soc=0.30,
        price=0.4,
        current_step=10,
        total_steps=96,
        terminal_soc_target=0.60,
        terminal_soc_tolerance=0.01,
    )
    shield_decision = shield_battery_action(
        np.array([0.0], dtype=np.float32),
        unwrapped_env=env,
        config=BatteryActionShieldConfig(terminal_closure_horizon_fraction=0.35),
        action_space_low=np.array([-1.0], dtype=np.float32),
        action_space_high=np.array([1.0], dtype=np.float32),
    )
    teacher = inventory_teacher_action(
        unwrapped_env=env,
        config=BatteryActionShieldConfig(terminal_closure_horizon_fraction=0.35),
        action_space_low=np.array([-1.0], dtype=np.float32),
        action_space_high=np.array([1.0], dtype=np.float32),
    )

    assert shield_decision.terminal_active is False
    assert shield_decision.closure_mix == pytest.approx(0.0)
    assert teacher.terminal_active is False


def test_inventory_teacher_prefers_charging_in_valley_when_inventory_is_low():
    env = _DummyEnv(
        soc=0.28,
        price=0.10,
        current_step=12,
        total_steps=96,
        terminal_soc_target=0.50,
        terminal_soc_tolerance=0.02,
    )
    env.config.reward.valley_price = 0.15
    env.config.reward.peak_price = 0.50
    teacher = inventory_teacher_action(
        unwrapped_env=env,
        config=BatteryActionShieldConfig(),
        action_space_low=np.array([-1.0], dtype=np.float32),
        action_space_high=np.array([1.0], dtype=np.float32),
    )

    assert teacher.active is True
    assert float(teacher.action[0]) < 0.0
    assert teacher.weight > 1.0


def test_inventory_teacher_recovers_reserve_in_peak_price_when_discharge_headroom_is_low():
    env = _DummyEnv(
        soc=0.32,
        price=0.60,
        current_step=48,
        total_steps=96,
        charge_limit_w=1000.0,
        discharge_limit_w=150.0,
        terminal_soc_target=0.40,
        terminal_soc_tolerance=0.02,
        peak_price=0.50,
    )
    teacher = inventory_teacher_action(
        unwrapped_env=env,
        config=BatteryActionShieldConfig(reserve_discharge_min_fraction=0.25),
        action_space_low=np.array([-1.0], dtype=np.float32),
        action_space_high=np.array([1.0], dtype=np.float32),
    )

    assert teacher.active is True
    assert teacher.reserve_active is True
    assert float(teacher.action[0]) <= 0.0


def test_inventory_teacher_marks_peak_value_conversion_inside_midband_as_active():
    env = _DummyEnv(
        soc=0.58,
        price=0.60,
        current_step=48,
        total_steps=96,
        charge_limit_w=1000.0,
        discharge_limit_w=1000.0,
        terminal_soc_target=0.50,
        terminal_soc_tolerance=0.05,
        peak_price=0.50,
    )
    teacher = inventory_teacher_action(
        unwrapped_env=env,
        config=BatteryActionShieldConfig(reserve_discharge_min_fraction=0.25),
        action_space_low=np.array([-1.0], dtype=np.float32),
        action_space_high=np.array([1.0], dtype=np.float32),
    )

    assert teacher.active is True
    assert teacher.reserve_active is False
    assert teacher.boundary_active is False
    assert teacher.terminal_active is False
    assert teacher.peak_value_active is True
    assert teacher.valley_value_active is False
    assert float(teacher.action[0]) > 0.10
    assert teacher.weight > 1.5


def test_inventory_teacher_keeps_near_target_peak_action_non_material():
    env = _DummyEnv(
        soc=0.54,
        price=0.60,
        current_step=48,
        total_steps=96,
        charge_limit_w=1000.0,
        discharge_limit_w=1000.0,
        terminal_soc_target=0.50,
        terminal_soc_tolerance=0.05,
        peak_price=0.50,
    )
    teacher = inventory_teacher_action(
        unwrapped_env=env,
        config=BatteryActionShieldConfig(reserve_discharge_min_fraction=0.25),
        action_space_low=np.array([-1.0], dtype=np.float32),
        action_space_high=np.array([1.0], dtype=np.float32),
    )

    assert teacher.active is False
    assert teacher.peak_value_active is False
    assert teacher.valley_value_active is False
    assert float(teacher.action[0]) > 0.10
    assert teacher.weight == pytest.approx(0.0)


def test_inventory_teacher_keeps_near_target_valley_action_non_material():
    env = _DummyEnv(
        soc=0.46,
        price=0.10,
        current_step=12,
        total_steps=96,
        terminal_soc_target=0.50,
        terminal_soc_tolerance=0.05,
    )
    env.config.reward.valley_price = 0.15
    env.config.reward.peak_price = 0.50
    teacher = inventory_teacher_action(
        unwrapped_env=env,
        config=BatteryActionShieldConfig(),
        action_space_low=np.array([-1.0], dtype=np.float32),
        action_space_high=np.array([1.0], dtype=np.float32),
    )

    assert teacher.active is False
    assert teacher.peak_value_active is False
    assert teacher.valley_value_active is False
    assert float(teacher.action[0]) < -0.10
    assert teacher.weight == pytest.approx(0.0)
