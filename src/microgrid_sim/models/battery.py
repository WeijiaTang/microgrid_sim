"""Battery models for PBM/EBM comparison."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _soc_energy_room_wh(params: BatteryParams, soc: float) -> tuple[float, float]:
    nominal_energy_wh = float(params.nominal_energy_wh)
    available_discharge_wh = max((float(soc) - float(params.soc_min)) * nominal_energy_wh, 0.0)
    available_charge_wh = max((float(params.soc_max) - float(soc)) * nominal_energy_wh, 0.0)
    return available_charge_wh, available_discharge_wh


@dataclass
class BatteryParams:
    """Battery pack and lookup parameters."""

    cell_capacity_ah: float = 280.0
    num_cells_series: int = 250
    num_cells_parallel: int = 1
    nominal_energy_kwh: float | None = None
    soc_min: float = 0.10
    soc_max: float = 0.90
    soc_init: float = 0.50
    p_charge_max: float = 50_000.0
    p_discharge_max: float = 50_000.0
    eta_charge: float = 1.00
    eta_discharge: float = 1.00
    soc_breakpoints: np.ndarray | None = None
    ocv_values: np.ndarray | None = None
    ocv_charge_values: np.ndarray | None = None
    ocv_discharge_values: np.ndarray | None = None
    r_int_values: np.ndarray | None = None
    thermal_dynamics_enabled: bool = True
    ambient_temperature_c: float = 25.0
    temperature_init_c: float = 25.0
    reference_temperature_c: float = 25.0
    r_int_temp_coeff_per_c: float = 0.015
    min_r_int_temp_factor: float = 0.70
    max_r_int_temp_factor: float = 2.50
    thermal_resistance_k_per_w: float = 1.50
    thermal_capacitance_j_per_k: float = 250_000.0
    temperature_min_c: float = -10.0
    temperature_max_c: float = 60.0
    low_soc_r_int_boost_enabled: bool = False
    low_soc_r_int_boost_threshold: float = 0.30
    low_soc_r_int_boost_factor: float = 1.0
    low_soc_r_int_boost_exponent: float = 1.0
    power_stress_r_int_boost_enabled: bool = False
    power_stress_r_int_boost_start_fraction: float = 0.35
    power_stress_r_int_boost_factor: float = 1.0
    power_stress_r_int_boost_exponent: float = 1.0
    paper_soc_nonlinearity_enabled: bool = False
    paper_soc_nonlinearity_gain: float = 0.0
    paper_soc_discharge_nonlinearity_gain: float | None = None
    paper_soc_charge_nonlinearity_gain: float | None = None
    paper_soc_nonlinearity_reference_soc: float = 0.55
    paper_soc_nonlinearity_charge_shift: float = 1.10
    paper_soc_nonlinearity_floor: float = 0.15
    paper_soc_nonlinearity_exponent: float = 1.0
    ocv_hysteresis_enabled: bool = False
    ocv_hysteresis_transition_tau_seconds: float = 5_400.0
    ocv_hysteresis_relaxation_tau_seconds: float = 21_600.0
    ocv_hysteresis_deadband_a: float = 2.0
    rc_branch_1_resistance_values: np.ndarray | None = None
    rc_branch_1_capacitance_values: np.ndarray | None = None
    rc_branch_2_resistance_values: np.ndarray | None = None
    rc_branch_2_capacitance_values: np.ndarray | None = None

    def __post_init__(self):
        if self.soc_breakpoints is None:
            self.soc_breakpoints = np.array([0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]) / 100.0
        if self.ocv_values is None:
            self.ocv_values = np.array([
                2.50,
                2.90,
                3.05,
                3.20,
                3.22,
                3.24,
                3.26,
                3.27,
                3.29,
                3.32,
                3.38,
                3.50,
                3.65,
            ])
        if self.r_int_values is None:
            self.r_int_values = np.array([
                0.36,
                0.32,
                0.29,
                0.26,
                0.24,
                0.23,
                0.22,
                0.23,
                0.24,
                0.26,
                0.29,
                0.32,
                0.36,
            ]) / 1000.0
        if self.ocv_charge_values is None:
            self.ocv_charge_values = np.asarray(self.ocv_values, dtype=float).copy()
        if self.ocv_discharge_values is None:
            self.ocv_discharge_values = np.asarray(self.ocv_values, dtype=float).copy()
        self.soc_breakpoints = np.asarray(self.soc_breakpoints, dtype=float).copy()
        self.ocv_values = np.asarray(self.ocv_values, dtype=float).copy()
        self.ocv_charge_values = np.asarray(self.ocv_charge_values, dtype=float).copy()
        self.ocv_discharge_values = np.asarray(self.ocv_discharge_values, dtype=float).copy()
        self.r_int_values = np.asarray(self.r_int_values, dtype=float).copy()
        if self.rc_branch_1_resistance_values is not None:
            self.rc_branch_1_resistance_values = np.asarray(self.rc_branch_1_resistance_values, dtype=float).copy()
        if self.rc_branch_1_capacitance_values is not None:
            self.rc_branch_1_capacitance_values = np.asarray(self.rc_branch_1_capacitance_values, dtype=float).copy()
        if self.rc_branch_2_resistance_values is not None:
            self.rc_branch_2_resistance_values = np.asarray(self.rc_branch_2_resistance_values, dtype=float).copy()
        if self.rc_branch_2_capacitance_values is not None:
            self.rc_branch_2_capacitance_values = np.asarray(self.rc_branch_2_capacitance_values, dtype=float).copy()
        self._validate()

    @property
    def nominal_energy_wh(self) -> float:
        if self.nominal_energy_kwh is not None:
            return self.nominal_energy_kwh * 1000.0
        return self.cell_capacity_ah * 3.2 * self.num_cells_series * self.num_cells_parallel

    @property
    def cell_stack_nominal_energy_wh(self) -> float:
        return self.cell_capacity_ah * 3.2 * self.num_cells_series * self.num_cells_parallel

    @property
    def nominal_energy_mismatch_fraction(self) -> float:
        if self.nominal_energy_kwh is None:
            return 0.0
        reference_wh = max(self.cell_stack_nominal_energy_wh, 1e-9)
        return abs(self.nominal_energy_wh - reference_wh) / reference_wh

    def _validate_lookup(self, name: str, values: np.ndarray, expected_len: int) -> None:
        if values.ndim != 1:
            raise ValueError(f"{name} must be a 1D array")
        if len(values) != expected_len:
            raise ValueError(f"{name} length {len(values)} does not match soc_breakpoints length {expected_len}")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain only finite values")

    def _validate_rc_pair(self, resistance_name: str, resistance_values: np.ndarray | None, capacitance_name: str, capacitance_values: np.ndarray | None) -> None:
        if resistance_values is None and capacitance_values is None:
            return
        if resistance_values is None or capacitance_values is None:
            raise ValueError(f"{resistance_name} and {capacitance_name} must both be provided")
        expected_len = len(self.soc_breakpoints)
        self._validate_lookup(resistance_name, resistance_values, expected_len)
        self._validate_lookup(capacitance_name, capacitance_values, expected_len)
        if np.any(resistance_values <= 0.0):
            raise ValueError(f"{resistance_name} must be strictly positive")
        if np.any(capacitance_values <= 0.0):
            raise ValueError(f"{capacitance_name} must be strictly positive")

    def _validate(self) -> None:
        if self.cell_capacity_ah <= 0.0:
            raise ValueError("cell_capacity_ah must be positive")
        if self.num_cells_series <= 0:
            raise ValueError("num_cells_series must be positive")
        if self.num_cells_parallel <= 0:
            raise ValueError("num_cells_parallel must be positive")
        if self.nominal_energy_kwh is not None and self.nominal_energy_kwh <= 0.0:
            raise ValueError("nominal_energy_kwh must be positive when provided")
        if self.p_charge_max < 0.0 or self.p_discharge_max < 0.0:
            raise ValueError("power limits must be non-negative")
        if self.eta_charge <= 0.0 or self.eta_discharge <= 0.0:
            raise ValueError("charge and discharge efficiencies must be positive")
        if not 0.0 <= self.soc_min < self.soc_max <= 1.0:
            raise ValueError("SOC bounds must satisfy 0 <= soc_min < soc_max <= 1")
        if not self.soc_min <= self.soc_init <= self.soc_max:
            raise ValueError("soc_init must lie within [soc_min, soc_max]")
        if self.soc_breakpoints.ndim != 1:
            raise ValueError("soc_breakpoints must be a 1D array")
        if len(self.soc_breakpoints) < 2:
            raise ValueError("soc_breakpoints must contain at least two points")
        if not np.all(np.isfinite(self.soc_breakpoints)):
            raise ValueError("soc_breakpoints must contain only finite values")
        if np.any(np.diff(self.soc_breakpoints) <= 0.0):
            raise ValueError("soc_breakpoints must be strictly increasing")
        if self.soc_breakpoints[0] < 0.0 or self.soc_breakpoints[-1] > 1.0:
            raise ValueError("soc_breakpoints must stay within [0, 1]")
        expected_len = len(self.soc_breakpoints)
        self._validate_lookup("ocv_values", self.ocv_values, expected_len)
        self._validate_lookup("ocv_charge_values", self.ocv_charge_values, expected_len)
        self._validate_lookup("ocv_discharge_values", self.ocv_discharge_values, expected_len)
        self._validate_lookup("r_int_values", self.r_int_values, expected_len)
        if np.any(self.ocv_values <= 0.0) or np.any(self.ocv_charge_values <= 0.0) or np.any(self.ocv_discharge_values <= 0.0):
            raise ValueError("OCV lookup values must be strictly positive")
        if np.any(self.r_int_values <= 0.0):
            raise ValueError("r_int_values must be strictly positive")
        if self.r_int_temp_coeff_per_c < 0.0:
            raise ValueError("r_int_temp_coeff_per_c must be non-negative")
        if self.min_r_int_temp_factor <= 0.0:
            raise ValueError("min_r_int_temp_factor must be strictly positive")
        if self.max_r_int_temp_factor < self.min_r_int_temp_factor:
            raise ValueError("max_r_int_temp_factor must be >= min_r_int_temp_factor")
        if self.thermal_resistance_k_per_w <= 0.0:
            raise ValueError("thermal_resistance_k_per_w must be strictly positive")
        if self.thermal_capacitance_j_per_k <= 0.0:
            raise ValueError("thermal_capacitance_j_per_k must be strictly positive")
        if self.temperature_min_c >= self.temperature_max_c:
            raise ValueError("temperature_min_c must be smaller than temperature_max_c")
        for name, value in (
            ("ambient_temperature_c", self.ambient_temperature_c),
            ("temperature_init_c", self.temperature_init_c),
            ("reference_temperature_c", self.reference_temperature_c),
        ):
            if not self.temperature_min_c <= value <= self.temperature_max_c:
                raise ValueError(f"{name} must lie within [temperature_min_c, temperature_max_c]")
        self._validate_rc_pair(
            "rc_branch_1_resistance_values",
            self.rc_branch_1_resistance_values,
            "rc_branch_1_capacitance_values",
            self.rc_branch_1_capacitance_values,
        )
        self._validate_rc_pair(
            "rc_branch_2_resistance_values",
            self.rc_branch_2_resistance_values,
            "rc_branch_2_capacitance_values",
            self.rc_branch_2_capacitance_values,
        )
        if self.nominal_energy_mismatch_fraction > 0.05:
            raise ValueError(
                "nominal_energy_kwh is inconsistent with the configured cell stack energy by more than 5%"
            )


@dataclass
class BatteryStepResult:
    actual_power: float
    effective_power: float
    soc: float
    current: float
    voltage: float
    efficiency: float
    power_loss: float
    soc_violation: float
    v_ocv: float
    v_ocv_base: float
    v_ocv_charge: float
    v_ocv_discharge: float
    r_int: float
    r_int_base: float
    r_int_temp_factor: float
    r_int_soc_factor: float
    r_int_power_factor: float
    r_int_paper_soc_factor: float
    ocv_hysteresis_blend: float
    ocv_hysteresis_offset: float
    polarization_voltage: float
    polarization_power: float
    rc_branch_1_voltage: float
    rc_branch_2_voltage: float
    p_max: float
    temperature_c: float

    def as_dict(self) -> dict:
        return {
            "soc": self.soc,
            "current": self.current,
            "voltage": self.voltage,
            "efficiency": self.efficiency,
            "power_loss": self.power_loss,
            "soc_violation": self.soc_violation,
            "v_ocv": self.v_ocv,
            "v_ocv_base": self.v_ocv_base,
            "v_ocv_charge": self.v_ocv_charge,
            "v_ocv_discharge": self.v_ocv_discharge,
            "r_int": self.r_int,
            "r_int_base": self.r_int_base,
            "r_int_temp_factor": self.r_int_temp_factor,
            "r_int_soc_factor": self.r_int_soc_factor,
            "r_int_power_factor": self.r_int_power_factor,
            "r_int_paper_soc_factor": self.r_int_paper_soc_factor,
            "ocv_hysteresis_blend": self.ocv_hysteresis_blend,
            "ocv_hysteresis_offset": self.ocv_hysteresis_offset,
            "polarization_voltage": self.polarization_voltage,
            "polarization_power": self.polarization_power,
            "rc_branch_1_voltage": self.rc_branch_1_voltage,
            "rc_branch_2_voltage": self.rc_branch_2_voltage,
            "p_max": self.p_max,
            "effective_power": self.effective_power,
            "temperature_c": self.temperature_c,
        }


class TheveninBattery:
    """Physics-based battery model using an OCV--Rint pathway."""

    def __init__(self, params: BatteryParams | None = None):
        self.params = params or BatteryParams()
        self.soc = self.params.soc_init
        self.temperature_c = self.params.temperature_init_c
        self.ocv_branch_blend = 0.5
        self.ocv_hysteresis_offset_v = 0.0
        self.v_rc_1 = 0.0
        self.v_rc_2 = 0.0

    def reset(self, soc: float | None = None) -> float:
        self.soc = self.params.soc_init if soc is None else float(soc)
        self.temperature_c = self.params.temperature_init_c
        self.ocv_branch_blend = 0.5
        self.ocv_hysteresis_offset_v = 0.0
        self.v_rc_1 = 0.0
        self.v_rc_2 = 0.0
        return self.soc

    def _interp(self, x: float, fp: np.ndarray) -> float:
        return float(np.interp(x, self.params.soc_breakpoints, fp))

    def get_ocv(self, soc: float) -> float:
        return self._interp(np.clip(soc, 0.0, 1.0), self.params.ocv_values)

    def get_ocv_charge(self, soc: float) -> float:
        return self._interp(np.clip(soc, 0.0, 1.0), np.asarray(self.params.ocv_charge_values, dtype=float))

    def get_ocv_discharge(self, soc: float) -> float:
        return self._interp(np.clip(soc, 0.0, 1.0), np.asarray(self.params.ocv_discharge_values, dtype=float))

    def get_r_int(self, soc: float) -> float:
        return self._interp(np.clip(soc, 0.0, 1.0), self.params.r_int_values)

    def _interp_optional(self, soc: float, values: np.ndarray | None) -> float:
        if values is None:
            return 0.0
        return self._interp(np.clip(soc, 0.0, 1.0), np.asarray(values, dtype=float))

    def _temperature_r_int_factor(self) -> float:
        params = self.params
        if not params.thermal_dynamics_enabled:
            return 1.0
        delta = params.reference_temperature_c - self.temperature_c
        factor = 1.0 + params.r_int_temp_coeff_per_c * delta
        return float(np.clip(factor, params.min_r_int_temp_factor, params.max_r_int_temp_factor))

    def _soc_r_int_factor(self, soc: float) -> float:
        params = self.params
        if not params.low_soc_r_int_boost_enabled:
            return 1.0
        threshold = max(params.low_soc_r_int_boost_threshold, params.soc_min + 1e-6)
        if soc >= threshold:
            return 1.0
        span = max(threshold - params.soc_min, 1e-6)
        depth = np.clip((threshold - soc) / span, 0.0, 1.0)
        factor = 1.0 + (params.low_soc_r_int_boost_factor - 1.0) * (depth ** params.low_soc_r_int_boost_exponent)
        return float(max(1.0, factor))

    def _power_stress_r_int_factor(self, requested_power: float) -> float:
        params = self.params
        if not params.power_stress_r_int_boost_enabled or requested_power <= 0.0:
            return 1.0
        max_power = max(params.p_discharge_max, params.p_charge_max, 1e-9)
        power_fraction = requested_power / max_power
        start = np.clip(params.power_stress_r_int_boost_start_fraction, 0.0, 0.99)
        stress = np.clip((power_fraction - start) / max(1.0 - start, 1e-6), 0.0, 1.0)
        factor = 1.0 + (params.power_stress_r_int_boost_factor - 1.0) * (stress ** params.power_stress_r_int_boost_exponent)
        return float(max(1.0, factor))

    def _paper_soc_r_int_factor(self, soc: float, requested_power: float) -> float:
        params = self.params
        if not params.paper_soc_nonlinearity_enabled or abs(requested_power) <= 1e-9:
            return 1.0
        reference_soc = float(np.clip(params.paper_soc_nonlinearity_reference_soc, params.soc_min + 1e-6, params.soc_max - 1e-6))
        denominator_floor = max(float(params.paper_soc_nonlinearity_floor), 1e-6)
        exponent = max(float(params.paper_soc_nonlinearity_exponent), 1.0)
        if requested_power >= 0.0:
            raw_gain = params.paper_soc_discharge_nonlinearity_gain
            if raw_gain is None:
                raw_gain = params.paper_soc_nonlinearity_gain
            gain = max(float(raw_gain), 0.0)
            if gain <= 0.0:
                return 1.0
            denominator = max(float(soc), denominator_floor)
            reference_denominator = max(reference_soc, denominator_floor)
        else:
            raw_gain = params.paper_soc_charge_nonlinearity_gain
            if raw_gain is None:
                raw_gain = params.paper_soc_nonlinearity_gain
            gain = max(float(raw_gain), 0.0)
            if gain <= 0.0:
                return 1.0
            charge_shift = max(float(params.paper_soc_nonlinearity_charge_shift), params.soc_max + 1e-6)
            denominator = max(charge_shift - float(soc), denominator_floor)
            reference_denominator = max(charge_shift - reference_soc, denominator_floor)
        normalized_stress = max(reference_denominator / denominator - 1.0, 0.0)
        factor = 1.0 + gain * (normalized_stress ** exponent)
        return float(max(1.0, factor))

    def _effective_ocv(self, soc: float) -> tuple[float, float, float, float, float]:
        ocv_base = self.get_ocv(soc) * self.params.num_cells_series
        ocv_charge = self.get_ocv_charge(soc) * self.params.num_cells_series
        ocv_discharge = self.get_ocv_discharge(soc) * self.params.num_cells_series
        if not self.params.ocv_hysteresis_enabled:
            return ocv_base, ocv_base, ocv_charge, ocv_discharge, 0.0
        blend = float(np.clip(self.ocv_branch_blend, 0.0, 1.0))
        ocv_effective = blend * ocv_charge + (1.0 - blend) * ocv_discharge
        return ocv_effective, ocv_base, ocv_charge, ocv_discharge, ocv_effective - ocv_base

    def _update_ocv_hysteresis(self, current: float, dt: float) -> None:
        params = self.params
        if not params.ocv_hysteresis_enabled:
            self.ocv_branch_blend = 0.5
            self.ocv_hysteresis_offset_v = 0.0
            return
        deadband = max(float(params.ocv_hysteresis_deadband_a), 0.0)
        if current > deadband:
            target = 0.0
            tau = max(float(params.ocv_hysteresis_transition_tau_seconds), 1e-6)
        elif current < -deadband:
            target = 1.0
            tau = max(float(params.ocv_hysteresis_transition_tau_seconds), 1e-6)
        else:
            target = 0.5
            tau = max(float(params.ocv_hysteresis_relaxation_tau_seconds), 1e-6)
        alpha = float(np.exp(-dt / tau))
        self.ocv_branch_blend = float(np.clip(target + (self.ocv_branch_blend - target) * alpha, 0.0, 1.0))

    def _rc_branch_pack_params(self, soc: float, branch_index: int) -> tuple[float, float]:
        if branch_index == 1:
            r_values = self.params.rc_branch_1_resistance_values
            c_values = self.params.rc_branch_1_capacitance_values
        else:
            r_values = self.params.rc_branch_2_resistance_values
            c_values = self.params.rc_branch_2_capacitance_values
        r_cell = self._interp_optional(soc, r_values)
        c_cell = self._interp_optional(soc, c_values)
        if r_cell <= 0.0 or c_cell <= 0.0:
            return 0.0, 0.0
        r_pack = r_cell * self.params.num_cells_series / max(self.params.num_cells_parallel, 1)
        c_pack = c_cell * max(self.params.num_cells_parallel, 1) / max(self.params.num_cells_series, 1)
        return float(r_pack), float(c_pack)

    def _update_rc_branch(self, v_prev: float, current: float, r_pack: float, c_pack: float, dt: float) -> float:
        if r_pack <= 0.0 or c_pack <= 0.0:
            return 0.0
        tau = max(r_pack * c_pack, 1e-6)
        alpha = float(np.exp(-dt / tau))
        return float(alpha * v_prev + (1.0 - alpha) * current * r_pack)

    def _update_temperature(self, power_loss: float, dt: float) -> float:
        params = self.params
        if not params.thermal_dynamics_enabled:
            self.temperature_c = params.temperature_init_c
            return self.temperature_c
        cooling_power = (self.temperature_c - params.ambient_temperature_c) / max(params.thermal_resistance_k_per_w, 1e-9)
        d_temp = (power_loss - cooling_power) * dt / max(params.thermal_capacitance_j_per_k, 1e-9)
        self.temperature_c = float(np.clip(self.temperature_c + d_temp, params.temperature_min_c, params.temperature_max_c))
        return self.temperature_c

    def _apply_energy_update(self, effective_power: float, dt: float) -> tuple[float, float]:
        energy_before = self.soc * self.params.nominal_energy_wh
        energy_after = energy_before - effective_power * dt / 3600.0
        soc_raw = energy_after / self.params.nominal_energy_wh
        soc_violation = 0.0
        if soc_raw < self.params.soc_min:
            soc_violation = self.params.soc_min - soc_raw
            soc_raw = self.params.soc_min
        elif soc_raw > self.params.soc_max:
            soc_violation = soc_raw - self.params.soc_max
            soc_raw = self.params.soc_max
        self.soc = soc_raw
        return soc_raw, soc_violation

    def _effective_power_bounds(self, dt: float) -> tuple[float, float]:
        available_charge_wh, available_discharge_wh = _soc_energy_room_wh(self.params, self.soc)
        scale = 3600.0 / max(float(dt), 1e-9)
        min_effective_power = -available_charge_wh * scale
        max_effective_power = available_discharge_wh * scale
        return float(min_effective_power), float(max_effective_power)

    def _solve_power_state(
        self,
        target_power: float,
        v_ocv: float,
        polarization_voltage: float,
        r_int: float,
    ) -> tuple[float, float, float, float, float, float, float]:
        v_headroom = max(v_ocv - polarization_voltage, 1e-6)
        discriminant = max(0.0, v_headroom**2 - 4.0 * r_int * target_power)
        current = (v_headroom - np.sqrt(discriminant)) / (2.0 * r_int) if r_int > 0 else 0.0
        voltage = v_ocv - polarization_voltage - current * r_int
        actual_power = voltage * current
        ohmic_loss = (current**2) * r_int
        polarization_power = current * polarization_voltage
        effective_power = actual_power + ohmic_loss + polarization_power
        power_loss = ohmic_loss + max(polarization_power, 0.0)
        efficiency = abs(actual_power) / abs(effective_power) if abs(effective_power) > 1e-9 else 1.0
        return current, voltage, actual_power, effective_power, power_loss, efficiency, polarization_power

    def step(self, p_cmd: float, dt: float = 3600.0) -> tuple[float, float, dict]:
        params = self.params
        p_cmd = float(np.clip(p_cmd, -params.p_charge_max, params.p_discharge_max))
        v_ocv, v_ocv_base, v_ocv_charge, v_ocv_discharge, ocv_hysteresis_offset = self._effective_ocv(self.soc)
        r_int_base = self.get_r_int(self.soc) * params.num_cells_series / params.num_cells_parallel
        r_int_temp_factor = self._temperature_r_int_factor()
        r_int_soc_factor = self._soc_r_int_factor(self.soc)
        r_int_power_factor = self._power_stress_r_int_factor(abs(p_cmd))
        r_int_paper_soc_factor = self._paper_soc_r_int_factor(self.soc, p_cmd)
        r_int = r_int_base * r_int_temp_factor * r_int_soc_factor * r_int_power_factor * r_int_paper_soc_factor
        rc_branch_1_voltage = float(self.v_rc_1)
        rc_branch_2_voltage = float(self.v_rc_2)
        polarization_voltage = rc_branch_1_voltage + rc_branch_2_voltage
        v_headroom = max(v_ocv - polarization_voltage, 1e-6)
        p_max = (v_headroom**2) / (4.0 * r_int) if r_int > 0 else params.p_discharge_max
        p_target = min(p_cmd, 0.999 * p_max) if p_cmd > 0 else p_cmd
        min_effective_power, max_effective_power = self._effective_power_bounds(dt)
        current, voltage, actual_power, effective_power, power_loss, efficiency, polarization_power = self._solve_power_state(
            p_target,
            v_ocv=v_ocv,
            polarization_voltage=polarization_voltage,
            r_int=r_int,
        )
                                                                                         
                                                                                              
        if effective_power > max_effective_power + 1e-9 and p_target > 0.0:
            low = 0.0
            high = float(p_target)
            for _ in range(32):
                mid = 0.5 * (low + high)
                _, _, _, mid_effective_power, _, _, _ = self._solve_power_state(
                    mid,
                    v_ocv=v_ocv,
                    polarization_voltage=polarization_voltage,
                    r_int=r_int,
                )
                if mid_effective_power > max_effective_power:
                    high = mid
                else:
                    low = mid
            p_target = low
            current, voltage, actual_power, effective_power, power_loss, efficiency, polarization_power = self._solve_power_state(
                p_target,
                v_ocv=v_ocv,
                polarization_voltage=polarization_voltage,
                r_int=r_int,
            )
        elif effective_power < min_effective_power - 1e-9 and p_target < 0.0:
            low = float(p_target)
            high = 0.0
            for _ in range(32):
                mid = 0.5 * (low + high)
                _, _, _, mid_effective_power, _, _, _ = self._solve_power_state(
                    mid,
                    v_ocv=v_ocv,
                    polarization_voltage=polarization_voltage,
                    r_int=r_int,
                )
                if mid_effective_power < min_effective_power:
                    low = mid
                else:
                    high = mid
            p_target = high
            current, voltage, actual_power, effective_power, power_loss, efficiency, polarization_power = self._solve_power_state(
                p_target,
                v_ocv=v_ocv,
                polarization_voltage=polarization_voltage,
                r_int=r_int,
            )
        soc, soc_violation = self._apply_energy_update(effective_power, dt)
        r_rc1, c_rc1 = self._rc_branch_pack_params(self.soc, branch_index=1)
        r_rc2, c_rc2 = self._rc_branch_pack_params(self.soc, branch_index=2)
        self.v_rc_1 = self._update_rc_branch(self.v_rc_1, current, r_rc1, c_rc1, dt)
        self.v_rc_2 = self._update_rc_branch(self.v_rc_2, current, r_rc2, c_rc2, dt)
        self._update_ocv_hysteresis(current, dt)
        self.ocv_hysteresis_offset_v = ocv_hysteresis_offset
        temperature_c = self._update_temperature(power_loss, dt)
        result = BatteryStepResult(
            actual_power=actual_power,
            effective_power=effective_power,
            soc=soc,
            current=current,
            voltage=voltage,
            efficiency=efficiency,
            power_loss=power_loss,
            soc_violation=soc_violation,
            v_ocv=v_ocv,
            v_ocv_base=v_ocv_base,
            v_ocv_charge=v_ocv_charge,
            v_ocv_discharge=v_ocv_discharge,
            r_int=r_int,
            r_int_base=r_int_base,
            r_int_temp_factor=r_int_temp_factor,
            r_int_soc_factor=r_int_soc_factor,
            r_int_power_factor=r_int_power_factor,
            r_int_paper_soc_factor=r_int_paper_soc_factor,
            ocv_hysteresis_blend=self.ocv_branch_blend,
            ocv_hysteresis_offset=ocv_hysteresis_offset,
            polarization_voltage=polarization_voltage,
            polarization_power=polarization_power,
            rc_branch_1_voltage=rc_branch_1_voltage,
            rc_branch_2_voltage=rc_branch_2_voltage,
            p_max=p_max,
            temperature_c=temperature_c,
        )
        return result.actual_power, result.soc, result.as_dict()


class SimpleBattery:
    """Efficiency-based battery model used as the EBM baseline."""

    def __init__(self, params: BatteryParams | None = None):
        self.params = params or BatteryParams()
        self.soc = self.params.soc_init
        self.temperature_c = self.params.temperature_init_c
        self.ocv_branch_blend = 0.5
        self.ocv_hysteresis_offset_v = 0.0
        self.v_rc_1 = 0.0
        self.v_rc_2 = 0.0

    def reset(self, soc: float | None = None) -> float:
        self.soc = self.params.soc_init if soc is None else float(soc)
        self.temperature_c = self.params.temperature_init_c
        self.ocv_branch_blend = 0.5
        self.ocv_hysteresis_offset_v = 0.0
        self.v_rc_1 = 0.0
        self.v_rc_2 = 0.0
        return self.soc

    def step(self, p_cmd: float, dt: float = 3600.0) -> tuple[float, float, dict]:
        params = self.params
        actual_power = float(np.clip(p_cmd, -params.p_charge_max, params.p_discharge_max))
        available_charge_wh, available_discharge_wh = _soc_energy_room_wh(params, self.soc)
        discharge_power_limit = available_discharge_wh * 3600.0 / max(float(dt), 1e-9) * max(float(params.eta_discharge), 0.0)
        charge_power_limit = available_charge_wh * 3600.0 / max(float(dt), 1e-9) / max(float(params.eta_charge), 1e-9)
        actual_power = float(np.clip(actual_power, -charge_power_limit, discharge_power_limit))
        if actual_power >= 0:
            effective_power = actual_power / max(params.eta_discharge, 1e-9)
            efficiency = params.eta_discharge
        else:
            effective_power = actual_power * params.eta_charge
            efficiency = params.eta_charge
        energy_before = self.soc * params.nominal_energy_wh
        energy_after = energy_before - effective_power * dt / 3600.0
        soc_raw = energy_after / params.nominal_energy_wh
        soc_violation = 0.0
        if soc_raw < params.soc_min:
            soc_violation = params.soc_min - soc_raw
            soc_raw = params.soc_min
        elif soc_raw > params.soc_max:
            soc_violation = soc_raw - params.soc_max
            soc_raw = params.soc_max
        self.soc = soc_raw
        nominal_voltage = 3.2 * params.num_cells_series
        current = actual_power / nominal_voltage if nominal_voltage else 0.0
        result = BatteryStepResult(
            actual_power=actual_power,
            effective_power=effective_power,
            soc=self.soc,
            current=current,
            voltage=nominal_voltage,
            efficiency=efficiency,
            power_loss=abs(effective_power - actual_power),
            soc_violation=soc_violation,
            v_ocv=nominal_voltage,
            v_ocv_base=nominal_voltage,
            v_ocv_charge=nominal_voltage,
            v_ocv_discharge=nominal_voltage,
            r_int=0.0,
            r_int_base=0.0,
            r_int_temp_factor=1.0,
            r_int_soc_factor=1.0,
            r_int_power_factor=1.0,
            r_int_paper_soc_factor=1.0,
            ocv_hysteresis_blend=0.5,
            ocv_hysteresis_offset=0.0,
            polarization_voltage=0.0,
            polarization_power=0.0,
            rc_branch_1_voltage=0.0,
            rc_branch_2_voltage=0.0,
            p_max=params.p_discharge_max,
            temperature_c=self.temperature_c,
        )
        return result.actual_power, result.soc, result.as_dict()
