"""Residential MG-RES PBM-vs-EBM experiment with year-split data."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from microgrid_sim.cases import (
    MicrogridConfig,
    RESIDENTIAL_GRID_BATTERY_DEGRADATION_COST_MULTIPLIER,
    RESIDENTIAL_GRID_BATTERY_END_OF_LIFE_FRACTION,
    RESIDENTIAL_GRID_BATTERY_EQUIVALENT_FULL_CYCLES,
    RESIDENTIAL_GRID_MONTHLY_DEMAND_CHARGE_PER_KW,
    RESIDENTIAL_GRID_MONTHLY_DEMAND_CHARGE_THRESHOLD_W,
    RESIDENTIAL_GRID_BATTERY_REPLACEMENT_COST_PER_KWH,
    RESIDENTIAL_GRID_TOU_PRICE_SPREAD_MULTIPLIER,
)
from microgrid_sim.envs import ContinuousActionRegularizationWrapper, DiscreteActionWrapper, MicrogridEnv
from microgrid_sim.rl_utils import (
    SUPPORTED_AGENT_NAMES,
    canonicalize_agent_name,
    create_agent,
    load_agent,
    replay_buffer_size_for,
)

RESIDENTIAL_SAC_ENT_COEF = "auto_0.02"
RESIDENTIAL_SAC_TARGET_ENTROPY_SCALE = 0.35
RESIDENTIAL_GRID_BATTERY_THROUGHPUT_PENALTY_SENTINEL = -1.0
RESIDENTIAL_FULL_YEAR_RANDOM_START_HOURS = (0, 91 * 24, 182 * 24, 273 * 24)
RESIDENTIAL_FULL_YEAR_RANDOM_START_STRIDE_HOURS = 24
RESIDENTIAL_PBM_LOW_SOC_R_INT_THRESHOLD = 0.20
RESIDENTIAL_PBM_LOW_SOC_R_INT_FACTOR = 1.35
RESIDENTIAL_PBM_POWER_STRESS_R_INT_FACTOR = 1.10
RESIDENTIAL_CURRICULUM_WIDE_SOC_MIN = 0.25
RESIDENTIAL_CURRICULUM_FINAL_SOC_MIN = 0.35
RESIDENTIAL_CURRICULUM_WIDE_SOC_MAX = 0.85
RESIDENTIAL_AUTO_FINAL_PHASE_RANDOM_WINDOW = True
RESIDENTIAL_DQN_ACTION_BINS = 21
RESIDENTIAL_VALIDATION_SELECTION_DAYS = 30
RESIDENTIAL_VALIDATION_INTERVAL_STEPS = 5_000


def configure_reproducibility(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        import torch

        if hasattr(torch, "set_num_threads"):
            torch.set_num_threads(1)
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(1)
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        # Training can continue even if a backend does not expose strict-determinism toggles.
        pass


def get_device(force_cpu: bool = False) -> str:
    import torch

    if not force_cpu and torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return "cuda"
    print("Using CPU")
    return "cpu"


def parse_curriculum_days(raw: str, fallback_days: int) -> list[int]:
    values = [int(part.strip()) for part in str(raw).split(",") if part.strip()]
    cleaned = [day for day in values if day > 0]
    return cleaned or [int(fallback_days)]

def parse_int_list(raw: str | None) -> list[int]:
    if raw is None:
        return []
    values = []
    for part in str(raw).split(","):
        token = part.strip()
        if not token:
            continue
        values.append(int(token))
    return values


def parse_hidden_sizes(raw: str | None, fallback: Sequence[int] = (256, 128, 64)) -> list[int]:
    values = [int(size) for size in parse_int_list(raw) if int(size) > 0]
    return values or [int(size) for size in fallback if int(size) > 0]


def allocate_phase_steps(total_steps: int, phase_days: Sequence[int]) -> list[int]:
    if len(phase_days) <= 1:
        return [int(total_steps)]
    weights = np.asarray([max(int(day), 1) for day in phase_days], dtype=float)
    weights /= np.sum(weights)
    steps = np.floor(total_steps * weights).astype(int)
    steps[-1] += int(total_steps - int(np.sum(steps)))
    return [max(int(step), 1) for step in steps]


def build_agent_hyperparams(args: argparse.Namespace) -> dict:
    return {
        "net_arch": parse_hidden_sizes(getattr(args, "policy_net_arch", None)),
        "learning_rate": float(getattr(args, "rl_learning_rate", 3e-4)),
        "learning_starts": max(int(getattr(args, "rl_learning_starts", 1000)), 0),
        "off_policy_batch_size": max(int(getattr(args, "rl_batch_size", 384)), 1),
        "gamma": float(getattr(args, "rl_gamma", 0.985)),
        "tau": float(getattr(args, "rl_tau", 0.003)),
        "sac_ent_coef": str(getattr(args, "sac_ent_coef", RESIDENTIAL_SAC_ENT_COEF)),
        "sac_target_entropy_scale": float(
            getattr(args, "sac_target_entropy_scale", RESIDENTIAL_SAC_TARGET_ENTROPY_SCALE)
        ),
        "td3_action_noise_sigma": max(float(getattr(args, "td3_action_noise_sigma", 0.10)), 0.0),
        "td3_policy_delay": max(int(getattr(args, "td3_policy_delay", 2)), 1),
        "td3_target_policy_noise": max(float(getattr(args, "td3_target_policy_noise", 0.2)), 0.0),
        "td3_target_noise_clip": max(float(getattr(args, "td3_target_noise_clip", 0.5)), 0.0),
    }


def build_action_regularization(args: argparse.Namespace) -> dict:
    smoothing_coef = float(np.clip(float(getattr(args, "action_smoothing_coef", 0.0)), 0.0, 0.995))
    max_delta = max(float(getattr(args, "action_max_delta", 0.0)), 0.0)
    rate_penalty = max(float(getattr(args, "action_rate_penalty", 0.0)), 0.0)
    symmetric_battery_action = bool(getattr(args, "enable_symmetric_battery_action", False))
    enabled = bool(
        symmetric_battery_action
        or smoothing_coef > 1e-9
        or max_delta > 1e-9
        or rate_penalty > 1e-12
    )
    return {
        "enabled": enabled,
        "smoothing_coef": smoothing_coef,
        "max_delta": max_delta,
        "rate_penalty": rate_penalty,
        "symmetric_battery_action": symmetric_battery_action,
    }


def wrap_agent_env(
    env,
    agent_name: str,
    dqn_action_bins: int = RESIDENTIAL_DQN_ACTION_BINS,
    action_regularization: dict | None = None,
):
    agent_name = canonicalize_agent_name(agent_name)
    if agent_name == "dqn":
        return DiscreteActionWrapper(env, action_bins=int(dqn_action_bins))
    regularization = dict(action_regularization or {})
    if regularization.get("enabled", False):
        env = ContinuousActionRegularizationWrapper(
            env,
            smoothing_coef=float(regularization.get("smoothing_coef", 0.0)),
            max_delta=float(regularization.get("max_delta", 0.0)),
            rate_penalty=float(regularization.get("rate_penalty", 0.0)),
            symmetric_battery_action=bool(regularization.get("symmetric_battery_action", False)),
        )
    return env


def resolve_train_reward_override(train_value: float | None, eval_value: float) -> float:
    if train_value is None:
        return float(eval_value)
    return float(train_value)


def scale_battery_params(base_params, power_scale: float, energy_scale: float):
    scaled_power = max(float(power_scale), 1e-6)
    scaled_energy = max(float(energy_scale), 1e-6)
    nominal_energy_kwh = base_params.nominal_energy_wh / 1000.0 * scaled_energy
    return replace(
        base_params,
        nominal_energy_kwh=nominal_energy_kwh,
        p_charge_max=base_params.p_charge_max * scaled_power,
        p_discharge_max=base_params.p_discharge_max * scaled_power,
    )


def resolve_battery_throughput_penalty(
    direct_penalty_per_kwh: float,
    replacement_cost_per_kwh: float,
    equivalent_full_cycles: float,
    end_of_life_fraction: float,
    degradation_cost_multiplier: float,
) -> tuple[float, dict]:
    penalty = float(direct_penalty_per_kwh)
    if penalty >= 0.0:
        return penalty, {
            "mode": "direct",
            "replacement_cost_per_kwh": float(replacement_cost_per_kwh),
            "equivalent_full_cycles": float(equivalent_full_cycles),
            "end_of_life_fraction": float(end_of_life_fraction),
            "degradation_cost_multiplier": float(degradation_cost_multiplier),
        }
    effective_replacement_cost = max(float(replacement_cost_per_kwh), 0.0) * max(float(degradation_cost_multiplier), 0.0)
    # The penalty is charged on absolute charge/discharge throughput, so one full cycle consumes two units of throughput.
    usable_lifetime_throughput = 2.0 * max(float(equivalent_full_cycles), 1e-6) * max(float(end_of_life_fraction), 1e-6)
    resolved_penalty = effective_replacement_cost / usable_lifetime_throughput
    return resolved_penalty, {
        "mode": "derived",
        "replacement_cost_per_kwh": float(replacement_cost_per_kwh),
        "equivalent_full_cycles": float(equivalent_full_cycles),
        "end_of_life_fraction": float(end_of_life_fraction),
        "degradation_cost_multiplier": float(degradation_cost_multiplier),
        "effective_replacement_cost_per_kwh": effective_replacement_cost,
        "usable_lifetime_throughput_per_kwh": usable_lifetime_throughput,
    }


def summarize_storage_design(base_params, power_scale: float, energy_scale: float) -> dict:
    designed = scale_battery_params(base_params, power_scale=power_scale, energy_scale=energy_scale)
    nominal_energy_kwh = designed.nominal_energy_wh / 1000.0
    discharge_power_kw = designed.p_discharge_max / 1000.0
    charge_power_kw = designed.p_charge_max / 1000.0
    c_rate = discharge_power_kw / max(nominal_energy_kwh, 1e-9)
    return {
        "battery_nominal_energy_kwh": float(nominal_energy_kwh),
        "battery_charge_power_kw": float(charge_power_kw),
        "battery_discharge_power_kw": float(discharge_power_kw),
        "battery_discharge_c_rate": float(c_rate),
    }



def battery_debug_summary(params) -> dict:
    r_int_values = np.asarray(getattr(params, "r_int_values", []), dtype=float)
    ocv_values = np.asarray(getattr(params, "ocv_values", []), dtype=float)
    ocv_charge_values = np.asarray(getattr(params, "ocv_charge_values", ocv_values), dtype=float)
    ocv_discharge_values = np.asarray(getattr(params, "ocv_discharge_values", ocv_values), dtype=float)
    rc1_r = np.asarray(getattr(params, "rc_branch_1_resistance_values", []), dtype=float)
    rc2_r = np.asarray(getattr(params, "rc_branch_2_resistance_values", []), dtype=float)
    return {
        "soc_min": float(params.soc_min),
        "soc_max": float(params.soc_max),
        "soc_init": float(params.soc_init),
        "eta_charge": float(params.eta_charge),
        "eta_discharge": float(params.eta_discharge),
        "nominal_energy_kwh": float(params.nominal_energy_wh / 1000.0),
        "p_charge_max_kw": float(params.p_charge_max / 1000.0),
        "p_discharge_max_kw": float(params.p_discharge_max / 1000.0),
        "thermal_dynamics_enabled": bool(getattr(params, "thermal_dynamics_enabled", False)),
        "ambient_temperature_c": float(getattr(params, "ambient_temperature_c", 25.0)),
        "temperature_init_c": float(getattr(params, "temperature_init_c", 25.0)),
        "reference_temperature_c": float(getattr(params, "reference_temperature_c", 25.0)),
        "r_int_temp_coeff_per_c": float(getattr(params, "r_int_temp_coeff_per_c", 0.0)),
        "low_soc_r_int_boost_enabled": bool(getattr(params, "low_soc_r_int_boost_enabled", False)),
        "low_soc_r_int_boost_threshold": float(getattr(params, "low_soc_r_int_boost_threshold", 0.0)),
        "low_soc_r_int_boost_factor": float(getattr(params, "low_soc_r_int_boost_factor", 1.0)),
        "power_stress_r_int_boost_enabled": bool(getattr(params, "power_stress_r_int_boost_enabled", False)),
        "power_stress_r_int_boost_start_fraction": float(getattr(params, "power_stress_r_int_boost_start_fraction", 0.0)),
        "power_stress_r_int_boost_factor": float(getattr(params, "power_stress_r_int_boost_factor", 1.0)),
        "ocv_hysteresis_enabled": bool(getattr(params, "ocv_hysteresis_enabled", False)),
        "ocv_hysteresis_transition_tau_seconds": float(getattr(params, "ocv_hysteresis_transition_tau_seconds", 0.0)),
        "ocv_hysteresis_relaxation_tau_seconds": float(getattr(params, "ocv_hysteresis_relaxation_tau_seconds", 0.0)),
        "ocv_hysteresis_deadband_a": float(getattr(params, "ocv_hysteresis_deadband_a", 0.0)),
        "paper_soc_nonlinearity_enabled": bool(getattr(params, "paper_soc_nonlinearity_enabled", False)),
        "paper_soc_nonlinearity_gain": float(getattr(params, "paper_soc_nonlinearity_gain", 0.0)),
        "paper_soc_discharge_nonlinearity_gain": 0.0 if getattr(params, "paper_soc_discharge_nonlinearity_gain", None) is None else float(getattr(params, "paper_soc_discharge_nonlinearity_gain", 0.0)),
        "paper_soc_charge_nonlinearity_gain": 0.0 if getattr(params, "paper_soc_charge_nonlinearity_gain", None) is None else float(getattr(params, "paper_soc_charge_nonlinearity_gain", 0.0)),
        "paper_soc_nonlinearity_reference_soc": float(getattr(params, "paper_soc_nonlinearity_reference_soc", 0.0)),
        "paper_soc_nonlinearity_charge_shift": float(getattr(params, "paper_soc_nonlinearity_charge_shift", 0.0)),
        "paper_soc_nonlinearity_floor": float(getattr(params, "paper_soc_nonlinearity_floor", 0.0)),
        "ocv_min": float(np.min(ocv_values)) if ocv_values.size else 0.0,
        "ocv_max": float(np.max(ocv_values)) if ocv_values.size else 0.0,
        "ocv_hysteresis_span_max": float(np.max(np.abs(ocv_charge_values - ocv_discharge_values))) if ocv_charge_values.size and ocv_discharge_values.size else 0.0,
        "r_int_min": float(np.min(r_int_values)) if r_int_values.size else 0.0,
        "r_int_max": float(np.max(r_int_values)) if r_int_values.size else 0.0,
        "r_int_mean": float(np.mean(r_int_values)) if r_int_values.size else 0.0,
        "rc_branch_1_enabled": bool(rc1_r.size),
        "rc_branch_1_r_max": float(np.max(rc1_r)) if rc1_r.size else 0.0,
        "rc_branch_2_enabled": bool(rc2_r.size),
        "rc_branch_2_r_max": float(np.max(rc2_r)) if rc2_r.size else 0.0,
    }


def generator_debug_summary(params) -> dict:
    if params is None:
        return {
            "enabled": False,
            "p_min_kw": 0.0,
            "p_max_kw": 0.0,
            "start_threshold_kw": 0.0,
            "cost_a0": 0.0,
            "cost_a1": 0.0,
            "cost_a2": 0.0,
            "low_load_threshold_fraction": 0.0,
            "low_load_cost_scale": 0.0,
            "low_load_cost_exponent": 0.0,
        }
    return {
        "enabled": True,
        "p_min_kw": float(params.p_min_w / 1000.0),
        "p_max_kw": float(params.p_max_w / 1000.0),
        "start_threshold_kw": float(params.start_threshold_w / 1000.0),
        "cost_a0": float(params.cost_a0),
        "cost_a1": float(params.cost_a1),
        "cost_a2": float(params.cost_a2),
        "low_load_threshold_fraction": float(getattr(params, "low_load_threshold_fraction", 0.0)),
        "low_load_cost_scale": float(getattr(params, "low_load_cost_scale", 0.0)),
        "low_load_cost_exponent": float(getattr(params, "low_load_cost_exponent", 0.0)),
    }


def action_trace_summary(result: dict, sample_count: int = 12) -> dict:
    p_cmd = np.asarray(result.get("p_cmd_kw", []), dtype=float)
    p_actual = np.asarray(result.get("p_actual_kw", []), dtype=float)
    soc = np.asarray(result.get("soc", []), dtype=float)
    generator_power = np.asarray(result.get("generator_power_kw", []), dtype=float)
    sample_count = max(int(sample_count), 1)
    summary = {
        "steps_recorded": int(len(p_cmd)),
        "p_cmd_first_kw": p_cmd[:sample_count].round(6).tolist(),
        "p_actual_first_kw": p_actual[:sample_count].round(6).tolist(),
        "soc_first": soc[:sample_count].round(6).tolist(),
        "p_cmd_mean_kw": float(np.mean(p_cmd)) if p_cmd.size else 0.0,
        "p_actual_mean_kw": float(np.mean(p_actual)) if p_actual.size else 0.0,
        "p_cmd_std_kw": float(np.std(p_cmd)) if p_cmd.size else 0.0,
        "p_actual_std_kw": float(np.std(p_actual)) if p_actual.size else 0.0,
        "p_cmd_actual_l1_gap_kw": float(np.mean(np.abs(p_cmd - p_actual))) if p_cmd.size and p_actual.size else 0.0,
    }
    if generator_power.size:
        summary.update(
            {
                "generator_power_first_kw": generator_power[:sample_count].round(6).tolist(),
                "generator_power_mean_kw": float(np.mean(generator_power)),
                "generator_power_std_kw": float(np.std(generator_power)),
            }
        )
    return summary

def tune_residential_pbm_params(
    base_params,
    low_soc_threshold: float,
    low_soc_factor: float,
    power_stress_factor: float,
    r_int_scale: float,
):
    low_soc_factor = max(float(low_soc_factor), 1.0)
    power_stress_factor = max(float(power_stress_factor), 1.0)
    return replace(
        base_params,
        r_int_values=np.asarray(base_params.r_int_values, dtype=float) * max(float(r_int_scale), 1e-6),
        low_soc_r_int_boost_enabled=low_soc_factor > 1.0 + 1e-9,
        low_soc_r_int_boost_threshold=max(float(low_soc_threshold), base_params.soc_min + 1e-6),
        low_soc_r_int_boost_factor=low_soc_factor,
        low_soc_r_int_boost_exponent=1.15,
        power_stress_r_int_boost_enabled=power_stress_factor > 1.0 + 1e-9,
        power_stress_r_int_boost_start_fraction=0.25,
        power_stress_r_int_boost_factor=power_stress_factor,
        power_stress_r_int_boost_exponent=1.25,
    )


def make_optimistic_ebm_params(base_params, soc_min: float, soc_max: float, power_scale: float, efficiency: float):
    optimistic_soc_min = max(0.0, min(float(soc_min), 0.98))
    optimistic_soc_max = min(max(float(soc_max), optimistic_soc_min + 1e-6), 1.0)
    optimistic_soc_init = min(max(0.50, optimistic_soc_min), optimistic_soc_max)
    optimistic_power_scale = max(float(power_scale), 1.0)
    optimistic_efficiency = min(max(float(efficiency), 0.0), 1.0)
    return replace(
        base_params,
        soc_min=optimistic_soc_min,
        soc_max=optimistic_soc_max,
        soc_init=optimistic_soc_init,
        p_charge_max=base_params.p_charge_max * optimistic_power_scale,
        p_discharge_max=base_params.p_discharge_max * optimistic_power_scale,
        eta_charge=max(base_params.eta_charge, optimistic_efficiency),
        eta_discharge=max(base_params.eta_discharge, optimistic_efficiency),
        thermal_dynamics_enabled=False,
    )


def build_config(
    battery_model: str,
    simulation_days: int,
    seed: int,
    data_dir: str | None,
    data_year: int,
    random_episode_start: bool,
    episode_start_hour: int,
    reward_mode: str,
    random_initial_soc: bool,
    initial_soc_min: float,
    initial_soc_max: float,
    full_year_random_start_stride_hours: int,
    peak_import_penalty_per_kw: float,
    peak_import_threshold_kw: float,
    monthly_demand_charge_per_kw: float,
    monthly_demand_charge_threshold_kw: float,
    monthly_peak_increment_penalty_per_kw: float,
    battery_throughput_penalty_per_kwh: float,
    battery_loss_penalty_per_kwh: float,
    battery_stress_penalty_per_kwh: float,
    midday_pv_boost_multiplier: float,
    evening_load_boost_multiplier: float,
    stress_episode_sampling: bool,
    stress_sampling_strength: float,
    battery_power_scale: float,
    battery_energy_scale: float,
    pbm_low_soc_r_int_threshold: float,
    pbm_low_soc_r_int_factor: float,
    pbm_power_stress_r_int_factor: float,
    pbm_r_int_scale: float,
    optimistic_ebm_training: bool,
    optimistic_ebm_soc_min: float,
    optimistic_ebm_soc_max: float,
    optimistic_ebm_power_scale: float,
    optimistic_ebm_efficiency: float,
    optimistic_ebm_soc_penalty_scale: float,
    tou_price_spread_multiplier: float = 1.0,
    grid_slack_enabled: bool = True,
    nse_penalty_per_kwh: float = 0.0,
    curtailment_penalty_per_kwh: float = 0.0,
    grid_import_max_kw: float = float("inf"),
    grid_export_max_kw: float = float("inf"),
    generator_enabled: bool | None = None,
) -> MicrogridConfig:
    config = MicrogridConfig(simulation_days=simulation_days, seed=seed)
    config.observation_stack_steps = 4
    config.data_dir = data_dir
    config.data_year = int(data_year)
    config.random_episode_start = bool(random_episode_start)
    config.episode_start_hour = int(episode_start_hour)
    config.full_year_random_start_stride_hours = max(int(full_year_random_start_stride_hours), 1)
    config.reward_mode = reward_mode
    config.random_initial_soc = bool(random_initial_soc)
    config.initial_soc_min = float(initial_soc_min)
    config.initial_soc_max = float(initial_soc_max)
    config.tou_price_spread_multiplier = max(float(tou_price_spread_multiplier), 0.0)
    config.peak_import_penalty_per_kw = float(peak_import_penalty_per_kw)
    config.peak_import_penalty_threshold_w = float(peak_import_threshold_kw) * 1000.0
    config.monthly_demand_charge_per_kw = max(float(monthly_demand_charge_per_kw), 0.0)
    config.monthly_demand_charge_threshold_w = float(monthly_demand_charge_threshold_kw) * 1000.0
    config.monthly_peak_increment_penalty_per_kw = max(float(monthly_peak_increment_penalty_per_kw), 0.0)
    config.grid_slack_enabled = bool(grid_slack_enabled)
    config.nse_penalty_per_kwh = max(float(nse_penalty_per_kwh), 0.0)
    config.curtailment_penalty_per_kwh = max(float(curtailment_penalty_per_kwh), 0.0)
    config.grid_import_max = float(grid_import_max_kw) * 1000.0 if np.isfinite(float(grid_import_max_kw)) else float("inf")
    config.grid_export_max = float(grid_export_max_kw) * 1000.0 if np.isfinite(float(grid_export_max_kw)) else float("inf")
    if generator_enabled is not None:
        config.generator_enabled = bool(generator_enabled)
        if not config.generator_enabled:
            config.generator_params = None
    config.battery_throughput_penalty_per_kwh = max(float(battery_throughput_penalty_per_kwh), 0.0)
    config.battery_loss_penalty_per_kwh = max(float(battery_loss_penalty_per_kwh), 0.0)
    config.battery_stress_penalty_per_kwh = max(float(battery_stress_penalty_per_kwh), 0.0)
    config.midday_pv_boost_multiplier = float(midday_pv_boost_multiplier)
    config.evening_load_boost_multiplier = float(evening_load_boost_multiplier)
    config.stress_episode_sampling = bool(stress_episode_sampling)
    config.stress_sampling_strength = float(stress_sampling_strength)
    config.battery_params = scale_battery_params(config.battery_params, power_scale=battery_power_scale, energy_scale=battery_energy_scale)
    config.reward = replace(config.reward, w_soc_violation=100.0, reward_min=-5000.0, reward_max=500.0)
    if battery_model == "thevenin":
        config.battery_params = tune_residential_pbm_params(
            config.battery_params,
            low_soc_threshold=pbm_low_soc_r_int_threshold,
            low_soc_factor=pbm_low_soc_r_int_factor,
            power_stress_factor=pbm_power_stress_r_int_factor,
            r_int_scale=pbm_r_int_scale,
        )
    if battery_model == "simple" and optimistic_ebm_training:
        config.battery_params = make_optimistic_ebm_params(
            config.battery_params,
            soc_min=optimistic_ebm_soc_min,
            soc_max=optimistic_ebm_soc_max,
            power_scale=optimistic_ebm_power_scale,
            efficiency=optimistic_ebm_efficiency,
        )
        config.reward = replace(
            config.reward,
            w_soc_violation=config.reward.w_soc_violation * max(float(optimistic_ebm_soc_penalty_scale), 0.0),
        )
    return config


def create_env(
    config: MicrogridConfig,
    battery_model: str,
    monitor_file: Path | None = None,
    agent_name: str = "sac",
    dqn_action_bins: int = RESIDENTIAL_DQN_ACTION_BINS,
    action_regularization: dict | None = None,
):
    def _factory():
        cfg = copy.deepcopy(config)
        cfg.battery_model = battery_model
        env = MicrogridEnv(config=cfg)
        env = wrap_agent_env(
            env,
            agent_name=agent_name,
            dqn_action_bins=dqn_action_bins,
            action_regularization=action_regularization,
        )
        env.reset(seed=int(cfg.seed))
        env.action_space.seed(int(cfg.seed))
        env.observation_space.seed(int(cfg.seed))
        filename = None
        if monitor_file is not None:
            monitor_file.parent.mkdir(parents=True, exist_ok=True)
            filename = str(monitor_file)
        return Monitor(env, filename=filename)

    return DummyVecEnv([_factory])


def evaluate_validation_windows(
    agent: BaseAlgorithm,
    agent_name: str,
    validation_label: str,
    validation_days: int,
    validation_start_hours: Sequence[int],
    seed: int,
    data_dir: str | None,
    data_year: int,
    reward_mode: str,
    env_battery_model: str,
    full_year_random_start_stride_hours: int,
    peak_import_penalty_per_kw: float,
    peak_import_threshold_kw: float,
    monthly_demand_charge_per_kw: float,
    monthly_demand_charge_threshold_kw: float,
    battery_throughput_penalty_per_kwh: float,
    battery_loss_penalty_per_kwh: float,
    battery_stress_penalty_per_kwh: float,
    midday_pv_boost_multiplier: float,
    evening_load_boost_multiplier: float,
    battery_power_scale: float,
    battery_energy_scale: float,
    pbm_low_soc_r_int_threshold: float,
    pbm_low_soc_r_int_factor: float,
    pbm_power_stress_r_int_factor: float,
    pbm_r_int_scale: float,
    optimistic_ebm_training: bool,
    optimistic_ebm_soc_min: float,
    optimistic_ebm_soc_max: float,
    optimistic_ebm_power_scale: float,
    optimistic_ebm_efficiency: float,
    optimistic_ebm_soc_penalty_scale: float,
    tou_price_spread_multiplier: float,
    grid_slack_enabled: bool,
    nse_penalty_per_kwh: float,
    curtailment_penalty_per_kwh: float,
    grid_import_max_kw: float,
    grid_export_max_kw: float,
    generator_enabled: bool | None,
    dqn_action_bins: int = RESIDENTIAL_DQN_ACTION_BINS,
    action_regularization: dict | None = None,
) -> dict:
    validation_rows: list[dict] = []
    for start_hour in validation_start_hours:
        result = evaluate_rollout(
            agent,
            agent_name,
            f"{validation_label}_sh{int(start_hour)}",
            validation_days,
            seed,
            data_dir,
            data_year,
            int(start_hour),
            reward_mode,
            env_battery_model,
            full_year_random_start_stride_hours,
            peak_import_penalty_per_kw,
            peak_import_threshold_kw,
            monthly_demand_charge_per_kw,
            monthly_demand_charge_threshold_kw,
            battery_throughput_penalty_per_kwh,
            battery_loss_penalty_per_kwh,
            battery_stress_penalty_per_kwh,
            midday_pv_boost_multiplier,
            evening_load_boost_multiplier,
            battery_power_scale,
            battery_energy_scale,
            pbm_low_soc_r_int_threshold,
            pbm_low_soc_r_int_factor,
            pbm_power_stress_r_int_factor,
            pbm_r_int_scale,
            optimistic_ebm_training=optimistic_ebm_training,
            optimistic_ebm_soc_min=optimistic_ebm_soc_min,
            optimistic_ebm_soc_max=optimistic_ebm_soc_max,
            optimistic_ebm_power_scale=optimistic_ebm_power_scale,
            optimistic_ebm_efficiency=optimistic_ebm_efficiency,
            optimistic_ebm_soc_penalty_scale=optimistic_ebm_soc_penalty_scale,
            tou_price_spread_multiplier=tou_price_spread_multiplier,
            grid_slack_enabled=grid_slack_enabled,
            nse_penalty_per_kwh=nse_penalty_per_kwh,
            curtailment_penalty_per_kwh=curtailment_penalty_per_kwh,
            grid_import_max_kw=grid_import_max_kw,
            grid_export_max_kw=grid_export_max_kw,
            generator_enabled=generator_enabled,
            record_timeseries=False,
            dqn_action_bins=dqn_action_bins,
            action_regularization=action_regularization,
        )
        validation_rows.append(
            {
                "start_hour": int(start_hour),
                "total_cost": float(result["total_cost"]),
                "max_grid_import_kw": float(result.get("max_grid_import_kw", 0.0)),
                "p_actual_abs_p95_kw": float(result.get("p_actual_abs_p95_kw", 0.0)),
            }
        )
    costs = [float(row["total_cost"]) for row in validation_rows]
    return {
        "validation_mean_cost": float(np.mean(costs)) if costs else float("inf"),
        "validation_worst_cost": float(np.max(costs)) if costs else float("inf"),
        "validation_best_cost": float(np.min(costs)) if costs else float("inf"),
        "validation_window_costs": json.dumps(validation_rows, ensure_ascii=False),
    }


def train_agent(
    agent_name: str,
    battery_model: str,
    steps: int,
    train_days: int,
    seed: int,
    data_dir: str | None,
    data_year: int,
    random_episode_start: bool,
    episode_start_hour: int,
    reward_mode: str,
    random_initial_soc: bool,
    initial_soc_min: float,
    initial_soc_max: float,
    full_year_random_start_stride_hours: int,
    peak_import_penalty_per_kw: float,
    peak_import_threshold_kw: float,
    monthly_demand_charge_per_kw: float,
    monthly_demand_charge_threshold_kw: float,
    monthly_peak_increment_penalty_per_kw: float,
    battery_throughput_penalty_per_kwh: float,
    battery_loss_penalty_per_kwh: float,
    battery_stress_penalty_per_kwh: float,
    midday_pv_boost_multiplier: float,
    evening_load_boost_multiplier: float,
    stress_episode_sampling: bool,
    stress_sampling_strength: float,
    curriculum_days: Sequence[int],
    battery_power_scale: float,
    battery_energy_scale: float,
    pbm_low_soc_r_int_threshold: float,
    pbm_low_soc_r_int_factor: float,
    pbm_power_stress_r_int_factor: float,
    pbm_r_int_scale: float,
    optimistic_ebm_training: bool,
    optimistic_ebm_soc_min: float,
    optimistic_ebm_soc_max: float,
    optimistic_ebm_power_scale: float,
    optimistic_ebm_efficiency: float,
    optimistic_ebm_soc_penalty_scale: float,
    model_path: Path,
    tou_price_spread_multiplier: float = 1.0,
    grid_slack_enabled: bool = True,
    nse_penalty_per_kwh: float = 0.0,
    curtailment_penalty_per_kwh: float = 0.0,
    grid_import_max_kw: float = float("inf"),
    grid_export_max_kw: float = float("inf"),
    generator_enabled: bool | None = None,
    force_cpu: bool = False,
    dqn_action_bins: int = RESIDENTIAL_DQN_ACTION_BINS,
    validation_selection_enabled: bool = False,
    validation_days: int = RESIDENTIAL_VALIDATION_SELECTION_DAYS,
    validation_start_hours: Sequence[int] = (),
    validation_interval_steps: int = RESIDENTIAL_VALIDATION_INTERVAL_STEPS,
    validation_eval_battery_model: str = "thevenin",
    agent_hyperparams: dict | None = None,
    action_regularization: dict | None = None,
) -> tuple[BaseAlgorithm, dict]:
    configure_reproducibility(seed)
    agent_name = canonicalize_agent_name(agent_name)
    device = get_device(force_cpu=bool(force_cpu))
    phase_days = [int(day) for day in curriculum_days if int(day) > 0] or [int(train_days)]
    phase_steps = allocate_phase_steps(int(steps), phase_days)
    validation_selection_enabled = bool(validation_selection_enabled)
    validation_days = max(int(validation_days), 1)
    validation_interval_steps = max(int(validation_interval_steps), 1)
    validation_hours = [int(hour) for hour in validation_start_hours if int(hour) >= 0] or [
        int(hour) for hour in RESIDENTIAL_FULL_YEAR_RANDOM_START_HOURS
    ]
    validation_history: list[dict] = []
    best_validation_record: dict | None = None
    best_validation_mean_cost = float("inf")
    total_steps_trained = 0
    agent: BaseAlgorithm | None = None
    best_model_path = model_path.parent / f"{model_path.stem}_best.zip"
    last_model_path = model_path.parent / f"{model_path.stem}_last.zip"
    first_learn = True
    for phase_index, (phase_day_count, phase_step_count) in enumerate(zip(phase_days, phase_steps), start=1):
        phase_is_final = phase_index == len(phase_days)
        auto_final_phase_random_window = bool(
            RESIDENTIAL_AUTO_FINAL_PHASE_RANDOM_WINDOW and len(phase_days) > 1 and phase_is_final
        )
        phase_initial_soc_min = float(initial_soc_min)
        phase_initial_soc_max = float(initial_soc_max)
        if random_initial_soc and len(phase_days) > 1:
            phase_initial_soc_max = max(phase_initial_soc_max, RESIDENTIAL_CURRICULUM_WIDE_SOC_MAX)
        if random_initial_soc and len(phase_days) > 1 and phase_is_final:
            phase_initial_soc_min = min(phase_initial_soc_min, RESIDENTIAL_CURRICULUM_FINAL_SOC_MIN)
        if random_initial_soc and len(phase_days) > 1 and phase_day_count < train_days:
            phase_initial_soc_min = min(phase_initial_soc_min, RESIDENTIAL_CURRICULUM_WIDE_SOC_MIN)
        phase_random_episode_start = bool(
            random_episode_start
            or phase_day_count < train_days
            or auto_final_phase_random_window
        )
        phase_config = build_config(
            battery_model=battery_model,
            simulation_days=phase_day_count,
            seed=seed,
            data_dir=data_dir,
            data_year=data_year,
            random_episode_start=phase_random_episode_start,
            episode_start_hour=episode_start_hour,
            reward_mode=reward_mode,
            random_initial_soc=random_initial_soc,
            initial_soc_min=phase_initial_soc_min,
            initial_soc_max=phase_initial_soc_max,
            full_year_random_start_stride_hours=full_year_random_start_stride_hours,
            peak_import_penalty_per_kw=peak_import_penalty_per_kw,
            peak_import_threshold_kw=peak_import_threshold_kw,
            monthly_demand_charge_per_kw=monthly_demand_charge_per_kw,
            monthly_demand_charge_threshold_kw=monthly_demand_charge_threshold_kw,
            monthly_peak_increment_penalty_per_kw=monthly_peak_increment_penalty_per_kw,
            battery_throughput_penalty_per_kwh=battery_throughput_penalty_per_kwh,
            battery_loss_penalty_per_kwh=battery_loss_penalty_per_kwh,
            battery_stress_penalty_per_kwh=battery_stress_penalty_per_kwh,
            midday_pv_boost_multiplier=midday_pv_boost_multiplier,
            evening_load_boost_multiplier=evening_load_boost_multiplier,
            stress_episode_sampling=stress_episode_sampling,
            stress_sampling_strength=stress_sampling_strength,
            battery_power_scale=battery_power_scale,
            battery_energy_scale=battery_energy_scale,
            pbm_low_soc_r_int_threshold=pbm_low_soc_r_int_threshold,
            pbm_low_soc_r_int_factor=pbm_low_soc_r_int_factor,
            pbm_power_stress_r_int_factor=pbm_power_stress_r_int_factor,
            pbm_r_int_scale=pbm_r_int_scale,
            optimistic_ebm_training=optimistic_ebm_training,
            optimistic_ebm_soc_min=optimistic_ebm_soc_min,
            optimistic_ebm_soc_max=optimistic_ebm_soc_max,
            optimistic_ebm_power_scale=optimistic_ebm_power_scale,
            optimistic_ebm_efficiency=optimistic_ebm_efficiency,
            optimistic_ebm_soc_penalty_scale=optimistic_ebm_soc_penalty_scale,
            tou_price_spread_multiplier=tou_price_spread_multiplier,
            grid_slack_enabled=grid_slack_enabled,
            nse_penalty_per_kwh=nse_penalty_per_kwh,
            curtailment_penalty_per_kwh=curtailment_penalty_per_kwh,
            grid_import_max_kw=grid_import_max_kw,
            grid_export_max_kw=grid_export_max_kw,
            generator_enabled=generator_enabled,
        )
        if phase_config.random_episode_start and phase_day_count >= train_days:
            # Use the configured stride (daily by default) so the full-year phase can
            # sample broadly across the year instead of overfitting to a few anchor starts.
            phase_config.full_year_random_start_hours = ()
        phase_env = create_env(
            phase_config,
            battery_model=battery_model,
            monitor_file=model_path.parent / f"{model_path.stem}_phase{phase_index}.csv",
            agent_name=agent_name,
            dqn_action_bins=int(dqn_action_bins),
            action_regularization=action_regularization,
        )
        if agent is None:
            obs_dim = int(np.prod(getattr(phase_env.observation_space, "shape", (0,)) or (0,)))
            if hasattr(phase_env.action_space, "n"):
                action_dim = int(getattr(phase_env.action_space, "n", 0))
            else:
                action_dim = int(np.prod(getattr(phase_env.action_space, "shape", (0,)) or (0,)))
            replay_buffer_size = int(replay_buffer_size_for(agent_name, steps))
            print(f"  Replay buffer: {replay_buffer_size:,} (obs_dim={obs_dim}, action_dim={action_dim})")
            agent = create_agent(
                agent_name=agent_name,
                env=phase_env,
                total_steps=steps,
                seed=seed,
                device=device,
                sac_ent_coef=RESIDENTIAL_SAC_ENT_COEF,
                sac_target_entropy_scale=RESIDENTIAL_SAC_TARGET_ENTROPY_SCALE,
                agent_hyperparams=agent_hyperparams,
            )
        else:
            agent.set_env(phase_env)
        remaining_phase_steps = int(phase_step_count)
        phase_steps_trained = 0
        phase_chunk_index = 0
        while remaining_phase_steps > 0:
            chunk_steps = remaining_phase_steps if not validation_selection_enabled else min(remaining_phase_steps, validation_interval_steps)
            phase_chunk_index += 1
            print(
                f"Training {agent_name}:{battery_model} phase {phase_index}/{len(phase_days)} "
                f"chunk {phase_chunk_index} for {chunk_steps:,}/{phase_step_count:,} steps "
                f"on {phase_day_count}d windows (year {data_year}) ..."
            )
            agent.learn(total_timesteps=chunk_steps, progress_bar=True, reset_num_timesteps=first_learn)
            first_learn = False
            remaining_phase_steps -= int(chunk_steps)
            phase_steps_trained += int(chunk_steps)
            total_steps_trained += int(chunk_steps)
            if validation_selection_enabled:
                validation_record = evaluate_validation_windows(
                    agent=agent,
                    agent_name=agent_name,
                    validation_label=f"{model_path.stem}_phase{phase_index}_chunk{phase_chunk_index}",
                    validation_days=validation_days,
                    validation_start_hours=validation_hours,
                    seed=seed,
                    data_dir=data_dir,
                    data_year=data_year,
                    reward_mode=reward_mode,
                    env_battery_model=validation_eval_battery_model,
                    full_year_random_start_stride_hours=full_year_random_start_stride_hours,
                    peak_import_penalty_per_kw=peak_import_penalty_per_kw,
                    peak_import_threshold_kw=peak_import_threshold_kw,
                    monthly_demand_charge_per_kw=monthly_demand_charge_per_kw,
                    monthly_demand_charge_threshold_kw=monthly_demand_charge_threshold_kw,
                    battery_throughput_penalty_per_kwh=battery_throughput_penalty_per_kwh,
                    battery_loss_penalty_per_kwh=battery_loss_penalty_per_kwh,
                    battery_stress_penalty_per_kwh=battery_stress_penalty_per_kwh,
                    midday_pv_boost_multiplier=midday_pv_boost_multiplier,
                    evening_load_boost_multiplier=evening_load_boost_multiplier,
                    battery_power_scale=battery_power_scale,
                    battery_energy_scale=battery_energy_scale,
                    pbm_low_soc_r_int_threshold=pbm_low_soc_r_int_threshold,
                    pbm_low_soc_r_int_factor=pbm_low_soc_r_int_factor,
                    pbm_power_stress_r_int_factor=pbm_power_stress_r_int_factor,
                    pbm_r_int_scale=pbm_r_int_scale,
                    optimistic_ebm_training=optimistic_ebm_training,
                    optimistic_ebm_soc_min=optimistic_ebm_soc_min,
                    optimistic_ebm_soc_max=optimistic_ebm_soc_max,
                    optimistic_ebm_power_scale=optimistic_ebm_power_scale,
                    optimistic_ebm_efficiency=optimistic_ebm_efficiency,
                    optimistic_ebm_soc_penalty_scale=optimistic_ebm_soc_penalty_scale,
                    tou_price_spread_multiplier=tou_price_spread_multiplier,
                    grid_slack_enabled=grid_slack_enabled,
                    nse_penalty_per_kwh=nse_penalty_per_kwh,
                    curtailment_penalty_per_kwh=curtailment_penalty_per_kwh,
                    grid_import_max_kw=grid_import_max_kw,
                    grid_export_max_kw=grid_export_max_kw,
                    generator_enabled=generator_enabled,
                    dqn_action_bins=dqn_action_bins,
                    action_regularization=action_regularization,
                )
                validation_record.update(
                    {
                        "phase_index": int(phase_index),
                        "phase_day_count": int(phase_day_count),
                        "phase_chunk_index": int(phase_chunk_index),
                        "phase_steps_trained": int(phase_steps_trained),
                        "total_steps_trained": int(total_steps_trained),
                    }
                )
                validation_history.append(validation_record)
                print(
                    f"  Validation ({validation_eval_battery_model}, {validation_days}d x {len(validation_hours)} windows): "
                    f"mean={validation_record['validation_mean_cost']:.2f}, "
                    f"worst={validation_record['validation_worst_cost']:.2f}"
                )
                if float(validation_record["validation_mean_cost"]) < float(best_validation_mean_cost) - 1e-9:
                    best_validation_mean_cost = float(validation_record["validation_mean_cost"])
                    best_validation_record = dict(validation_record)
                    agent.save(str(best_model_path))
                    print(f"  New best checkpoint saved: mean cost {best_validation_mean_cost:.2f}")
        phase_env.close()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if validation_selection_enabled:
        agent.save(str(last_model_path))
        if best_validation_record is not None and best_model_path.exists():
            agent = load_agent(agent_name, str(best_model_path), device=device)
        if validation_history:
            pd.DataFrame(validation_history).to_csv(model_path.parent / f"{model_path.stem}_validation.csv", index=False)
    agent.save(str(model_path))
    training_meta = {
        "validation_selection_enabled": bool(validation_selection_enabled),
        "validation_days": int(validation_days),
        "validation_interval_steps": int(validation_interval_steps),
        "validation_start_hours": [int(hour) for hour in validation_hours],
        "validation_eval_battery_model": str(validation_eval_battery_model),
        "selected_checkpoint": best_validation_record,
        "validation_history_rows": int(len(validation_history)),
        "best_model_path": str(best_model_path) if best_validation_record is not None else "",
        "last_model_path": str(last_model_path) if validation_selection_enabled else "",
    }
    return agent, training_meta


def evaluate_rollout(
    agent: BaseAlgorithm,
    agent_name: str,
    label: str,
    eval_days: int,
    seed: int,
    data_dir: str | None,
    data_year: int,
    episode_start_hour: int,
    reward_mode: str,
    env_battery_model: str,
    full_year_random_start_stride_hours: int,
    peak_import_penalty_per_kw: float,
    peak_import_threshold_kw: float,
    monthly_demand_charge_per_kw: float,
    monthly_demand_charge_threshold_kw: float,
    battery_throughput_penalty_per_kwh: float,
    battery_loss_penalty_per_kwh: float,
    battery_stress_penalty_per_kwh: float,
    midday_pv_boost_multiplier: float,
    evening_load_boost_multiplier: float,
    battery_power_scale: float,
    battery_energy_scale: float,
    pbm_low_soc_r_int_threshold: float,
    pbm_low_soc_r_int_factor: float,
    pbm_power_stress_r_int_factor: float,
    pbm_r_int_scale: float,
    optimistic_ebm_training: bool = False,
    optimistic_ebm_soc_min: float = 0.0,
    optimistic_ebm_soc_max: float = 1.0,
    optimistic_ebm_power_scale: float = 1.0,
    optimistic_ebm_efficiency: float = 1.0,
    optimistic_ebm_soc_penalty_scale: float = 1.0,
    tou_price_spread_multiplier: float = 1.0,
    grid_slack_enabled: bool = True,
    nse_penalty_per_kwh: float = 0.0,
    curtailment_penalty_per_kwh: float = 0.0,
    grid_import_max_kw: float = float("inf"),
    grid_export_max_kw: float = float("inf"),
    generator_enabled: bool | None = None,
    record_timeseries: bool = True,
    dqn_action_bins: int = RESIDENTIAL_DQN_ACTION_BINS,
    action_regularization: dict | None = None,
) -> dict:
    agent_name = canonicalize_agent_name(agent_name)
    config = build_config(
        battery_model=env_battery_model,
        simulation_days=eval_days,
        seed=seed,
        data_dir=data_dir,
        data_year=data_year,
        random_episode_start=False,
        episode_start_hour=episode_start_hour,
        reward_mode=reward_mode,
        random_initial_soc=False,
        initial_soc_min=0.0,
        initial_soc_max=1.0,
        full_year_random_start_stride_hours=full_year_random_start_stride_hours,
        peak_import_penalty_per_kw=peak_import_penalty_per_kw,
        peak_import_threshold_kw=peak_import_threshold_kw,
        monthly_demand_charge_per_kw=monthly_demand_charge_per_kw,
        monthly_demand_charge_threshold_kw=monthly_demand_charge_threshold_kw,
        monthly_peak_increment_penalty_per_kw=0.0,
        battery_throughput_penalty_per_kwh=battery_throughput_penalty_per_kwh,
        battery_loss_penalty_per_kwh=battery_loss_penalty_per_kwh,
        battery_stress_penalty_per_kwh=battery_stress_penalty_per_kwh,
        midday_pv_boost_multiplier=midday_pv_boost_multiplier,
        evening_load_boost_multiplier=evening_load_boost_multiplier,
        stress_episode_sampling=False,
        stress_sampling_strength=0.0,
        battery_power_scale=battery_power_scale,
        battery_energy_scale=battery_energy_scale,
        pbm_low_soc_r_int_threshold=pbm_low_soc_r_int_threshold,
        pbm_low_soc_r_int_factor=pbm_low_soc_r_int_factor,
        pbm_power_stress_r_int_factor=pbm_power_stress_r_int_factor,
        pbm_r_int_scale=pbm_r_int_scale,
        optimistic_ebm_training=optimistic_ebm_training,
        optimistic_ebm_soc_min=optimistic_ebm_soc_min,
        optimistic_ebm_soc_max=optimistic_ebm_soc_max,
        optimistic_ebm_power_scale=optimistic_ebm_power_scale,
        optimistic_ebm_efficiency=optimistic_ebm_efficiency,
        optimistic_ebm_soc_penalty_scale=optimistic_ebm_soc_penalty_scale,
        tou_price_spread_multiplier=tou_price_spread_multiplier,
        grid_slack_enabled=grid_slack_enabled,
        nse_penalty_per_kwh=nse_penalty_per_kwh,
        curtailment_penalty_per_kwh=curtailment_penalty_per_kwh,
        grid_import_max_kw=grid_import_max_kw,
        grid_export_max_kw=grid_export_max_kw,
        generator_enabled=generator_enabled,
    )
    config.battery_model = env_battery_model
    env = MicrogridEnv(config=config)
    env = wrap_agent_env(
        env,
        agent_name=agent_name,
        dqn_action_bins=int(dqn_action_bins),
        action_regularization=action_regularization,
    )
    obs, _ = env.reset()
    steps: list[int] = []
    soc_hist: list[float] = []
    soh_hist: list[float] = []
    cost_hist: list[float] = []
    import_cost_hist: list[float] = []
    peak_penalty_hist: list[float] = []
    demand_charge_hist: list[float] = []
    throughput_penalty_hist: list[float] = []
    loss_penalty_hist: list[float] = []
    stress_penalty_hist: list[float] = []
    nse_cost_hist: list[float] = []
    curtailment_cost_hist: list[float] = []
    p_cmd_abs_hist: list[float] = []
    p_actual_abs_hist: list[float] = []
    battery_action_raw_hist: list[float] = []
    battery_action_applied_hist: list[float] = []
    battery_action_delta_hist: list[float] = []
    action_rate_penalty_hist: list[float] = []
    p_grid_kw_hist: list[float] = []
    import_kw_hist: list[float] = []
    price_hist: list[float] = []
    battery_loss_kw_hist: list[float] = []
    battery_eff_hist: list[float] = []
    battery_current_a_hist: list[float] = []
    battery_temperature_c_hist: list[float] = []
    battery_ocv_hysteresis_v_hist: list[float] = []
    battery_polarization_v_hist: list[float] = []
    battery_power_limit_ratio_hist: list[float] = []
    battery_resistance_state_hist: list[float] = []
    monthly_peak_billed_kw_hist: list[float] = []
    remaining_monthly_peak_headroom_kw_hist: list[float] = []
    nse_kw_hist: list[float] = []
    curtailment_kw_hist: list[float] = []
    p_cmd_kw_hist: list[float] = []
    p_actual_kw_hist: list[float] = []
    generator_power_kw_hist: list[float] = []
    generator_cost_hist: list[float] = []
    generator_action_hist: list[float] = []
    generator_low_load_penalty_sum = 0.0
    generator_low_load_hours = 0.0
    clip_flags = []
    throughput_kwh = 0.0
    battery_loss_kwh_sum = 0.0
    battery_current_sq_sum = 0.0
    nse_kwh_sum = 0.0
    curtailment_kwh_sum = 0.0
    dt_hours = float(config.dt_seconds) / 3600.0
    for step in range(eval_days * 24):
        action, _ = agent.predict(obs, deterministic=True)
        raw_battery_action = 0.0
        if agent_name != "dqn":
            raw_action_values = np.asarray(action, dtype=float).reshape(-1)
            raw_battery_action = float(np.clip(raw_action_values[0], -1.0, 1.0)) if raw_action_values.size else 0.0
        obs, reward, terminated, truncated, info = env.step(action)
        del reward
        p_cmd_w = float(info.get("p_cmd", 0.0))
        p_actual_w = float(info.get("p_actual", 0.0))
        p_grid_w = float(info.get("p_grid", 0.0))
        price = float(info.get("price", 0.0))
        battery_info = dict(info.get("battery_info", {}) or {})
        power_loss_w = float(battery_info.get("power_loss", 0.0))
        efficiency = float(battery_info.get("efficiency", 1.0))
        current_a = float(battery_info.get("current", 0.0))
        battery_temperature_c = float(info.get("battery_temperature_c", config.battery_params.temperature_init_c))
        battery_power_limit_ratio = float(info.get("battery_power_limit_ratio", 1.0))
        battery_resistance_state = float(info.get("battery_resistance_state", 0.0))
        current_monthly_peak_billed_kw = float(info.get("current_monthly_peak_billed_kw", 0.0))
        remaining_monthly_peak_headroom_kw = float(info.get("remaining_monthly_peak_headroom_kw", 0.0))
        ocv_hysteresis_v = float(battery_info.get("ocv_hysteresis_offset", 0.0))
        polarization_v = float(battery_info.get("polarization_voltage", 0.0))
        steps.append(step)
        soc_hist.append(float(info.get("soc", 0.0)))
        soh_hist.append(float(info.get("soh", 1.0)))
        cost_hist.append(float(info.get("cumulative_cost", 0.0)))
        import_cost_hist.append(float(info.get("import_cost", 0.0)))
        peak_penalty_hist.append(float(info.get("peak_import_penalty", 0.0)))
        demand_charge_hist.append(float(info.get("monthly_demand_charge", 0.0)))
        throughput_penalty_hist.append(float(info.get("battery_throughput_penalty", 0.0)))
        loss_penalty_hist.append(float(info.get("battery_loss_penalty", 0.0)))
        stress_penalty_hist.append(float(info.get("battery_stress_penalty", 0.0)))
        nse_cost_hist.append(float(info.get("nse_cost", 0.0)))
        curtailment_cost_hist.append(float(info.get("curtailment_cost", 0.0)))
        battery_action_raw_hist.append(float(info.get("battery_action_raw", raw_battery_action)))
        battery_action_applied_hist.append(float(info.get("battery_action_applied", raw_battery_action)))
        battery_action_delta_hist.append(float(info.get("battery_action_delta", 0.0)))
        action_rate_penalty_hist.append(float(info.get("action_rate_penalty", 0.0)))
        p_grid_kw_hist.append(p_grid_w / 1000.0)
        import_kw_hist.append(max(p_grid_w, 0.0) / 1000.0)
        price_hist.append(price)
        battery_loss_kw_hist.append(max(power_loss_w, 0.0) / 1000.0)
        battery_eff_hist.append(efficiency)
        battery_current_a_hist.append(current_a)
        battery_temperature_c_hist.append(battery_temperature_c)
        battery_power_limit_ratio_hist.append(battery_power_limit_ratio)
        battery_resistance_state_hist.append(battery_resistance_state)
        monthly_peak_billed_kw_hist.append(current_monthly_peak_billed_kw)
        remaining_monthly_peak_headroom_kw_hist.append(remaining_monthly_peak_headroom_kw)
        battery_ocv_hysteresis_v_hist.append(ocv_hysteresis_v)
        battery_polarization_v_hist.append(polarization_v)
        nse_kw_hist.append(max(float(info.get("nse_power", 0.0)), 0.0) / 1000.0)
        curtailment_kw_hist.append(max(float(info.get("curtailment_power", 0.0)), 0.0) / 1000.0)
        p_cmd_abs_hist.append(abs(p_cmd_w) / 1000.0)
        p_actual_abs_hist.append(abs(p_actual_w) / 1000.0)
        p_cmd_kw_hist.append(p_cmd_w / 1000.0)
        p_actual_kw_hist.append(p_actual_w / 1000.0)
        generator_power_kw_hist.append(float(info.get("generator_power", 0.0)) / 1000.0)
        generator_cost_hist.append(float(info.get("generator_cost", 0.0)))
        generator_action_hist.append(float(info.get("generator_action", 0.0)))
        generator_low_load_penalty_sum += float(info.get("generator_low_load_penalty", 0.0))
        generator_low_load_hours += dt_hours if bool(info.get("generator_low_load_region", False)) else 0.0
        clip_flags.append(abs(p_actual_w - p_cmd_w) > 1e-6)
        throughput_kwh += abs(p_actual_w) * dt_hours / 1000.0
        battery_loss_kwh_sum += max(power_loss_w, 0.0) * dt_hours / 1000.0
        battery_current_sq_sum += current_a * current_a
        nse_kwh_sum += max(float(info.get("nse_power", 0.0)), 0.0) * dt_hours / 1000.0
        curtailment_kwh_sum += max(float(info.get("curtailment_power", 0.0)), 0.0) * dt_hours / 1000.0
        if terminated or truncated:
            break
    env.close()
    steps_recorded = int(len(steps))
    current_rms_a = float(np.sqrt(battery_current_sq_sum / max(steps_recorded, 1))) if steps_recorded else 0.0
    return {
        "label": label,
        "eval_year": int(data_year),
        "total_cost": float(cost_hist[-1] if cost_hist else 0.0),
        "final_soh": float(soh_hist[-1] if soh_hist else 1.0),
        "soc_min": float(min(soc_hist) if soc_hist else 0.0),
        "soc_max": float(max(soc_hist) if soc_hist else 0.0),
        "import_cost_sum": float(sum(import_cost_hist)),
        "peak_import_penalty_sum": float(sum(peak_penalty_hist)),
        "monthly_demand_charge_sum": float(sum(demand_charge_hist)),
        "nse_cost_sum": float(sum(nse_cost_hist)),
        "curtailment_cost_sum": float(sum(curtailment_cost_hist)),
        "battery_throughput_penalty_sum": float(sum(throughput_penalty_hist)),
        "battery_loss_penalty_sum": float(sum(loss_penalty_hist)),
        "battery_stress_penalty_sum": float(sum(stress_penalty_hist)),
        "generator_cost_sum": float(sum(generator_cost_hist)),
        "generator_low_load_penalty_sum": float(generator_low_load_penalty_sum),
        "generator_low_load_hours": float(generator_low_load_hours),
        "generator_energy_kwh": float(sum(generator_power_kw_hist) * dt_hours),
        "generator_power_p95_kw": float(np.percentile(generator_power_kw_hist, 95)) if generator_power_kw_hist else 0.0,
        "nse_kwh_sum": float(nse_kwh_sum),
        "curtailment_kwh_sum": float(curtailment_kwh_sum),
        "max_grid_import_kw": float(max(import_kw_hist, default=0.0)),
        "p_cmd_abs_p95_kw": float(np.percentile(p_cmd_abs_hist, 95)) if p_cmd_abs_hist else 0.0,
        "p_actual_abs_p95_kw": float(np.percentile(p_actual_abs_hist, 95)) if p_actual_abs_hist else 0.0,
        "battery_action_raw_abs_mean": float(np.mean(np.abs(battery_action_raw_hist))) if battery_action_raw_hist else 0.0,
        "battery_action_applied_abs_mean": float(np.mean(np.abs(battery_action_applied_hist))) if battery_action_applied_hist else 0.0,
        "battery_action_delta_abs_mean": float(np.mean(np.abs(battery_action_delta_hist))) if battery_action_delta_hist else 0.0,
        "action_rate_penalty_sum": float(sum(action_rate_penalty_hist)),
        "clip_ratio": float(np.mean(clip_flags)) if clip_flags else 0.0,
        "throughput_kwh": float(throughput_kwh),
        "battery_loss_kwh_sum": float(battery_loss_kwh_sum),
        "battery_current_rms_a": float(current_rms_a),
        "battery_efficiency_mean": float(np.mean(battery_eff_hist)) if battery_eff_hist else 0.0,
        "battery_temperature_mean_c": float(np.mean(battery_temperature_c_hist)) if battery_temperature_c_hist else 0.0,
        "battery_temperature_max_c": float(np.max(battery_temperature_c_hist)) if battery_temperature_c_hist else 0.0,
        "battery_power_limit_ratio_mean": float(np.mean(battery_power_limit_ratio_hist)) if battery_power_limit_ratio_hist else 0.0,
        "battery_power_limit_ratio_min": float(np.min(battery_power_limit_ratio_hist)) if battery_power_limit_ratio_hist else 0.0,
        "battery_resistance_state_mean": float(np.mean(battery_resistance_state_hist)) if battery_resistance_state_hist else 0.0,
        "battery_resistance_state_max": float(np.max(battery_resistance_state_hist)) if battery_resistance_state_hist else 0.0,
        "monthly_peak_billed_kw_max": float(np.max(monthly_peak_billed_kw_hist)) if monthly_peak_billed_kw_hist else 0.0,
        "remaining_monthly_peak_headroom_kw_min": float(np.min(remaining_monthly_peak_headroom_kw_hist)) if remaining_monthly_peak_headroom_kw_hist else 0.0,
        "battery_ocv_hysteresis_abs_p95_v": float(np.percentile(np.abs(battery_ocv_hysteresis_v_hist), 95)) if battery_ocv_hysteresis_v_hist else 0.0,
        "battery_polarization_abs_p95_v": float(np.percentile(np.abs(battery_polarization_v_hist), 95)) if battery_polarization_v_hist else 0.0,
        **(
            {
                "steps": steps,
                "soc": soc_hist,
                "soh": soh_hist,
                "cost": cost_hist,
                "p_cmd_kw": p_cmd_kw_hist,
                "p_actual_kw": p_actual_kw_hist,
                "p_grid_kw": p_grid_kw_hist,
                "import_kw": import_kw_hist,
                "battery_action_raw": battery_action_raw_hist,
                "battery_action_applied": battery_action_applied_hist,
                "battery_action_delta": battery_action_delta_hist,
                "price": price_hist,
                "battery_loss_kw": battery_loss_kw_hist,
                "battery_efficiency": battery_eff_hist,
                "battery_current_a": battery_current_a_hist,
                "battery_temperature_c": battery_temperature_c_hist,
                "battery_power_limit_ratio": battery_power_limit_ratio_hist,
                "battery_resistance_state": battery_resistance_state_hist,
                "monthly_peak_billed_kw": monthly_peak_billed_kw_hist,
                "remaining_monthly_peak_headroom_kw": remaining_monthly_peak_headroom_kw_hist,
                "battery_ocv_hysteresis_v": battery_ocv_hysteresis_v_hist,
                "battery_polarization_v": battery_polarization_v_hist,
                "nse_kw": nse_kw_hist,
                "curtailment_kw": curtailment_kw_hist,
                "generator_power_kw": generator_power_kw_hist,
                "generator_action": generator_action_hist,
            }
            if record_timeseries
            else {}
        ),
    }


def evaluate_idle_baseline(eval_days: int, seed: int, data_dir: str | None, data_year: int, episode_start_hour: int, reward_mode: str, peak_import_penalty_per_kw: float, peak_import_threshold_kw: float, monthly_demand_charge_per_kw: float, monthly_demand_charge_threshold_kw: float, battery_throughput_penalty_per_kwh: float, battery_loss_penalty_per_kwh: float, battery_stress_penalty_per_kwh: float, midday_pv_boost_multiplier: float, evening_load_boost_multiplier: float, battery_power_scale: float, battery_energy_scale: float, pbm_low_soc_r_int_threshold: float, pbm_low_soc_r_int_factor: float, pbm_power_stress_r_int_factor: float, pbm_r_int_scale: float, tou_price_spread_multiplier: float = 1.0, grid_slack_enabled: bool = True, nse_penalty_per_kwh: float = 0.0, curtailment_penalty_per_kwh: float = 0.0, grid_import_max_kw: float = float("inf"), grid_export_max_kw: float = float("inf"), generator_enabled: bool | None = None) -> dict:
    config = build_config(
        battery_model="thevenin",
        simulation_days=eval_days,
        seed=seed,
        data_dir=data_dir,
        data_year=data_year,
        random_episode_start=False,
        episode_start_hour=int(episode_start_hour),
        reward_mode=reward_mode,
        random_initial_soc=False,
        initial_soc_min=0.0,
        initial_soc_max=1.0,
        full_year_random_start_stride_hours=RESIDENTIAL_FULL_YEAR_RANDOM_START_STRIDE_HOURS,
        peak_import_penalty_per_kw=peak_import_penalty_per_kw,
        peak_import_threshold_kw=peak_import_threshold_kw,
        monthly_demand_charge_per_kw=monthly_demand_charge_per_kw,
        monthly_demand_charge_threshold_kw=monthly_demand_charge_threshold_kw,
        monthly_peak_increment_penalty_per_kw=0.0,
        battery_throughput_penalty_per_kwh=battery_throughput_penalty_per_kwh,
        battery_loss_penalty_per_kwh=battery_loss_penalty_per_kwh,
        battery_stress_penalty_per_kwh=battery_stress_penalty_per_kwh,
        midday_pv_boost_multiplier=midday_pv_boost_multiplier,
        evening_load_boost_multiplier=evening_load_boost_multiplier,
        stress_episode_sampling=False,
        stress_sampling_strength=0.0,
        battery_power_scale=battery_power_scale,
        battery_energy_scale=battery_energy_scale,
        pbm_low_soc_r_int_threshold=pbm_low_soc_r_int_threshold,
        pbm_low_soc_r_int_factor=pbm_low_soc_r_int_factor,
        pbm_power_stress_r_int_factor=pbm_power_stress_r_int_factor,
        pbm_r_int_scale=pbm_r_int_scale,
        optimistic_ebm_training=False,
        optimistic_ebm_soc_min=0.0,
        optimistic_ebm_soc_max=1.0,
        optimistic_ebm_power_scale=1.0,
        optimistic_ebm_efficiency=1.0,
        optimistic_ebm_soc_penalty_scale=1.0,
        tou_price_spread_multiplier=tou_price_spread_multiplier,
        grid_slack_enabled=grid_slack_enabled,
        nse_penalty_per_kwh=nse_penalty_per_kwh,
        curtailment_penalty_per_kwh=curtailment_penalty_per_kwh,
        grid_import_max_kw=grid_import_max_kw,
        grid_export_max_kw=grid_export_max_kw,
        generator_enabled=generator_enabled,
    )
    config.battery_model = "thevenin"
    env = MicrogridEnv(config=config)
    obs, _ = env.reset()
    del obs
    total_cost = 0.0
    total_peak_penalty = 0.0
    total_monthly_demand_charge = 0.0
    zero_action = np.zeros(env.action_space.shape, dtype=np.float32)
    for _ in range(eval_days * 24):
        _, _, terminated, truncated, info = env.step(zero_action)
        total_cost = float(info.get("cumulative_cost", total_cost))
        total_peak_penalty += float(info.get("peak_import_penalty", 0.0))
        total_monthly_demand_charge += float(info.get("monthly_demand_charge", 0.0))
        if terminated or truncated:
            break
    env.close()
    return {"label": "idle_baseline", "total_cost": total_cost, "peak_import_penalty_sum": total_peak_penalty, "monthly_demand_charge_sum": total_monthly_demand_charge}


def export_timeseries(result: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"Hour": result["steps"], "SOC": result["soc"], "SOH": result["soh"], "Cumulative_Cost": result["cost"]}
    if "p_actual_kw" in result:
        payload["Battery_Power_kW"] = result["p_actual_kw"]
    if "generator_power_kw" in result:
        payload["Generator_Power_kW"] = result["generator_power_kw"]
    if "p_grid_kw" in result:
        payload["Grid_Power_kW"] = result["p_grid_kw"]
    if "import_kw" in result:
        payload["Grid_Import_kW"] = result["import_kw"]
    if "battery_action_raw" in result:
        payload["Battery_Action_Raw"] = result["battery_action_raw"]
    if "battery_action_applied" in result:
        payload["Battery_Action_Applied"] = result["battery_action_applied"]
    if "battery_action_delta" in result:
        payload["Battery_Action_Delta"] = result["battery_action_delta"]
    if "price" in result:
        payload["Price_per_kWh"] = result["price"]
    if "battery_loss_kw" in result:
        payload["Battery_Loss_kW"] = result["battery_loss_kw"]
    if "battery_efficiency" in result:
        payload["Battery_Efficiency"] = result["battery_efficiency"]
    if "battery_current_a" in result:
        payload["Battery_Current_A"] = result["battery_current_a"]
    if "battery_temperature_c" in result:
        payload["Battery_Temperature_C"] = result["battery_temperature_c"]
    if "battery_power_limit_ratio" in result:
        payload["Battery_Power_Limit_Ratio"] = result["battery_power_limit_ratio"]
    if "battery_resistance_state" in result:
        payload["Battery_Resistance_State"] = result["battery_resistance_state"]
    if "monthly_peak_billed_kw" in result:
        payload["Monthly_Peak_Billed_kW"] = result["monthly_peak_billed_kw"]
    if "remaining_monthly_peak_headroom_kw" in result:
        payload["Remaining_Monthly_Peak_Headroom_kW"] = result["remaining_monthly_peak_headroom_kw"]
    if "battery_ocv_hysteresis_v" in result:
        payload["Battery_OCV_Hysteresis_V"] = result["battery_ocv_hysteresis_v"]
    if "battery_polarization_v" in result:
        payload["Battery_Polarization_V"] = result["battery_polarization_v"]
    if "nse_kw" in result:
        payload["NSE_kW"] = result["nse_kw"]
    if "curtailment_kw" in result:
        payload["Curtailment_kW"] = result["curtailment_kw"]
    pd.DataFrame(payload).to_csv(out_path, index=False)

SUMMARY_DROP_COLUMNS = [
    "steps",
    "soc",
    "soh",
    "cost",
    "p_cmd_kw",
    "p_actual_kw",
    "p_grid_kw",
    "import_kw",
    "battery_action_raw",
    "battery_action_applied",
    "battery_action_delta",
    "price",
    "battery_loss_kw",
    "battery_efficiency",
    "battery_current_a",
    "battery_temperature_c",
    "battery_power_limit_ratio",
    "battery_resistance_state",
    "monthly_peak_billed_kw",
    "remaining_monthly_peak_headroom_kw",
    "battery_ocv_hysteresis_v",
    "battery_polarization_v",
    "nse_kw",
    "curtailment_kw",
    "generator_power_kw",
    "generator_action",
]


def _strip_timeseries(result: dict) -> dict:
    return {key: value for key, value in result.items() if key not in SUMMARY_DROP_COLUMNS}


def _dedupe_ints(values: Sequence[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for val in values:
        x = int(val)
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def run_seed_experiment(
    args: argparse.Namespace,
    seed: int,
    output_dir: Path,
    models_dir: Path,
    eval_start_hours: list[int],
    battery_throughput_penalty_per_kwh: float,
    degradation_meta: dict,
    storage_design: dict,
    generator_design: dict,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    agent_name = canonicalize_agent_name(getattr(args, "agent", "sac"))
    dqn_action_bins = int(getattr(args, "dqn_action_bins", RESIDENTIAL_DQN_ACTION_BINS))
    fair_train = bool(getattr(args, "fair_train", False))
    random_train_soc = not args.disable_random_train_soc
    stress_episode_sampling = bool(getattr(args, "stress_episode_sampling", False))
    if fair_train:
        ebm_stress_episode_sampling = bool(stress_episode_sampling)
    else:
        ebm_stress_episode_sampling = bool(args.enable_ebm_stress_episode_sampling) and stress_episode_sampling
    optimistic_ebm_training = bool(args.enable_optimistic_ebm) and not bool(args.disable_optimistic_ebm)
    curriculum_days = parse_curriculum_days(args.curriculum_days, args.train_days)
    ebm_curriculum_days = curriculum_days if fair_train else [int(args.train_days)]
    agent_hyperparams = build_agent_hyperparams(args)
    action_regularization = build_action_regularization(args)

    grid_slack_enabled = not bool(getattr(args, "islanded", False))
    nse_penalty_per_kwh = float(getattr(args, "nse_penalty_per_kwh", 0.0))
    curtailment_penalty_per_kwh = float(getattr(args, "curtailment_penalty_per_kwh", 0.0))
    tou_price_spread_multiplier = float(getattr(args, "tou_price_spread_multiplier", 1.0))
    grid_import_max_kw = float(getattr(args, "grid_import_max_kw", float("inf")))
    grid_export_max_kw = float(getattr(args, "grid_export_max_kw", float("inf")))
    if not grid_slack_enabled:
        grid_import_max_kw = 0.0
        grid_export_max_kw = 0.0
    generator_enabled = bool(getattr(args, "enable_generator", False)) and not bool(getattr(args, "disable_generator", False))

    train_monthly_peak_increment_penalty_per_kw = max(float(args.monthly_peak_increment_penalty_per_kw), 0.0)
    train_battery_throughput_penalty_per_kwh = max(
        resolve_train_reward_override(getattr(args, "train_battery_throughput_penalty_per_kwh", None), battery_throughput_penalty_per_kwh),
        0.0,
    )
    train_battery_loss_penalty_per_kwh = max(
        resolve_train_reward_override(getattr(args, "train_battery_loss_penalty_per_kwh", None), args.battery_loss_penalty_per_kwh),
        0.0,
    )
    train_battery_stress_penalty_per_kwh = max(
        resolve_train_reward_override(getattr(args, "train_battery_stress_penalty_per_kwh", None), args.battery_stress_penalty_per_kwh),
        0.0,
    )

    pbm_debug_config = build_config(
        battery_model="thevenin",
        simulation_days=args.train_days,
        seed=seed,
        data_dir=args.data_dir,
        data_year=args.train_year,
        random_episode_start=args.random_train_window,
        episode_start_hour=args.train_start_hour,
        reward_mode=args.reward_mode,
        random_initial_soc=random_train_soc,
        initial_soc_min=args.train_soc_min,
        initial_soc_max=args.train_soc_max,
        full_year_random_start_stride_hours=RESIDENTIAL_FULL_YEAR_RANDOM_START_STRIDE_HOURS,
        peak_import_penalty_per_kw=args.peak_import_penalty_per_kw,
        peak_import_threshold_kw=args.peak_import_threshold_kw,
        monthly_demand_charge_per_kw=args.monthly_demand_charge_per_kw,
        monthly_demand_charge_threshold_kw=args.monthly_demand_charge_threshold_kw,
        monthly_peak_increment_penalty_per_kw=train_monthly_peak_increment_penalty_per_kw,
        battery_throughput_penalty_per_kwh=train_battery_throughput_penalty_per_kwh,
        battery_loss_penalty_per_kwh=train_battery_loss_penalty_per_kwh,
        battery_stress_penalty_per_kwh=train_battery_stress_penalty_per_kwh,
        midday_pv_boost_multiplier=args.midday_pv_boost_multiplier,
        evening_load_boost_multiplier=args.evening_load_boost_multiplier,
        stress_episode_sampling=stress_episode_sampling,
        stress_sampling_strength=args.stress_sampling_strength,
        battery_power_scale=args.battery_power_scale,
        battery_energy_scale=args.battery_energy_scale,
        pbm_low_soc_r_int_threshold=args.pbm_low_soc_r_int_threshold,
        pbm_low_soc_r_int_factor=args.pbm_low_soc_r_int_factor,
        pbm_power_stress_r_int_factor=args.pbm_power_stress_r_int_factor,
        pbm_r_int_scale=args.pbm_r_int_scale,
        optimistic_ebm_training=optimistic_ebm_training,
        optimistic_ebm_soc_min=args.optimistic_ebm_soc_min,
        optimistic_ebm_soc_max=args.optimistic_ebm_soc_max,
        optimistic_ebm_power_scale=args.optimistic_ebm_power_scale,
        optimistic_ebm_efficiency=args.optimistic_ebm_efficiency,
        optimistic_ebm_soc_penalty_scale=args.optimistic_ebm_soc_penalty_scale,
        tou_price_spread_multiplier=tou_price_spread_multiplier,
        grid_slack_enabled=grid_slack_enabled,
        nse_penalty_per_kwh=nse_penalty_per_kwh,
        curtailment_penalty_per_kwh=curtailment_penalty_per_kwh,
        grid_import_max_kw=grid_import_max_kw,
        grid_export_max_kw=grid_export_max_kw,
        generator_enabled=generator_enabled,
    )
    if args.random_train_window:
        pbm_debug_config.full_year_random_start_hours = tuple(int(hour) for hour in RESIDENTIAL_FULL_YEAR_RANDOM_START_HOURS)

    ebm_debug_config = build_config(
        battery_model="simple",
        simulation_days=args.train_days,
        seed=seed,
        data_dir=args.data_dir,
        data_year=args.train_year,
        random_episode_start=args.random_train_window,
        episode_start_hour=args.train_start_hour,
        reward_mode=args.reward_mode,
        random_initial_soc=random_train_soc,
        initial_soc_min=args.train_soc_min,
        initial_soc_max=args.train_soc_max,
        full_year_random_start_stride_hours=RESIDENTIAL_FULL_YEAR_RANDOM_START_STRIDE_HOURS,
        peak_import_penalty_per_kw=args.peak_import_penalty_per_kw,
        peak_import_threshold_kw=args.peak_import_threshold_kw,
        monthly_demand_charge_per_kw=args.monthly_demand_charge_per_kw,
        monthly_demand_charge_threshold_kw=args.monthly_demand_charge_threshold_kw,
        monthly_peak_increment_penalty_per_kw=train_monthly_peak_increment_penalty_per_kw,
        battery_throughput_penalty_per_kwh=train_battery_throughput_penalty_per_kwh,
        battery_loss_penalty_per_kwh=train_battery_loss_penalty_per_kwh,
        battery_stress_penalty_per_kwh=train_battery_stress_penalty_per_kwh,
        midday_pv_boost_multiplier=args.midday_pv_boost_multiplier,
        evening_load_boost_multiplier=args.evening_load_boost_multiplier,
        stress_episode_sampling=ebm_stress_episode_sampling,
        stress_sampling_strength=args.stress_sampling_strength if ebm_stress_episode_sampling else 0.0,
        battery_power_scale=args.battery_power_scale,
        battery_energy_scale=args.battery_energy_scale,
        pbm_low_soc_r_int_threshold=args.pbm_low_soc_r_int_threshold,
        pbm_low_soc_r_int_factor=args.pbm_low_soc_r_int_factor,
        pbm_power_stress_r_int_factor=args.pbm_power_stress_r_int_factor,
        pbm_r_int_scale=args.pbm_r_int_scale,
        optimistic_ebm_training=optimistic_ebm_training,
        optimistic_ebm_soc_min=args.optimistic_ebm_soc_min,
        optimistic_ebm_soc_max=args.optimistic_ebm_soc_max,
        optimistic_ebm_power_scale=args.optimistic_ebm_power_scale,
        optimistic_ebm_efficiency=args.optimistic_ebm_efficiency,
        optimistic_ebm_soc_penalty_scale=args.optimistic_ebm_soc_penalty_scale,
        tou_price_spread_multiplier=tou_price_spread_multiplier,
        grid_slack_enabled=grid_slack_enabled,
        nse_penalty_per_kwh=nse_penalty_per_kwh,
        curtailment_penalty_per_kwh=curtailment_penalty_per_kwh,
        grid_import_max_kw=grid_import_max_kw,
        grid_export_max_kw=grid_export_max_kw,
        generator_enabled=generator_enabled,
    )
    if args.random_train_window:
        ebm_debug_config.full_year_random_start_hours = tuple(int(hour) for hour in RESIDENTIAL_FULL_YEAR_RANDOM_START_HOURS)

    pbm_steps = int(args.steps) if fair_train else max(int(args.steps), int(round(float(args.steps) * max(float(args.pbm_step_multiplier), 1.0))))
    pbm_replay_buffer_size = int(replay_buffer_size_for(agent_name, pbm_steps))
    ebm_replay_buffer_size = int(replay_buffer_size_for(agent_name, args.steps))

    suffix = f"seed{seed}_{args.steps}_train{args.train_year}_eval{args.eval_year}"
    pbm_model_path = models_dir / f"pbm_{agent_name}_{suffix}.zip"
    ebm_model_path = models_dir / f"ebm_{agent_name}_{suffix}.zip"
    validation_selection_enabled = bool(getattr(args, "validation_selection", False))
    validation_days = max(int(getattr(args, "validation_days", RESIDENTIAL_VALIDATION_SELECTION_DAYS)), 1)
    validation_interval_steps = max(int(getattr(args, "validation_interval_steps", RESIDENTIAL_VALIDATION_INTERVAL_STEPS)), 1)
    validation_eval_battery_model = str(getattr(args, "validation_eval_battery_model", "thevenin"))
    validation_start_hours = parse_int_list(getattr(args, "validation_start_hours", None)) or [
        int(hour) for hour in RESIDENTIAL_FULL_YEAR_RANDOM_START_HOURS
    ]
    validation_start_hours = _dedupe_ints([max(int(hour), 0) for hour in validation_start_hours])

    pbm_agent, pbm_train_meta = train_agent(
        agent_name=agent_name,
        battery_model="thevenin",
        steps=pbm_steps,
        train_days=args.train_days,
        seed=seed,
        data_dir=args.data_dir,
        data_year=args.train_year,
        random_episode_start=args.random_train_window,
        episode_start_hour=args.train_start_hour,
        reward_mode=args.reward_mode,
        random_initial_soc=random_train_soc,
        initial_soc_min=args.train_soc_min,
        initial_soc_max=args.train_soc_max,
        full_year_random_start_stride_hours=RESIDENTIAL_FULL_YEAR_RANDOM_START_STRIDE_HOURS,
        peak_import_penalty_per_kw=args.peak_import_penalty_per_kw,
        peak_import_threshold_kw=args.peak_import_threshold_kw,
        monthly_demand_charge_per_kw=args.monthly_demand_charge_per_kw,
        monthly_demand_charge_threshold_kw=args.monthly_demand_charge_threshold_kw,
        monthly_peak_increment_penalty_per_kw=train_monthly_peak_increment_penalty_per_kw,
        battery_throughput_penalty_per_kwh=train_battery_throughput_penalty_per_kwh,
        battery_loss_penalty_per_kwh=train_battery_loss_penalty_per_kwh,
        battery_stress_penalty_per_kwh=train_battery_stress_penalty_per_kwh,
        midday_pv_boost_multiplier=args.midday_pv_boost_multiplier,
        evening_load_boost_multiplier=args.evening_load_boost_multiplier,
        stress_episode_sampling=stress_episode_sampling,
        stress_sampling_strength=args.stress_sampling_strength,
        curriculum_days=curriculum_days,
        battery_power_scale=args.battery_power_scale,
        battery_energy_scale=args.battery_energy_scale,
        pbm_low_soc_r_int_threshold=args.pbm_low_soc_r_int_threshold,
        pbm_low_soc_r_int_factor=args.pbm_low_soc_r_int_factor,
        pbm_power_stress_r_int_factor=args.pbm_power_stress_r_int_factor,
        pbm_r_int_scale=args.pbm_r_int_scale,
        optimistic_ebm_training=False,
        optimistic_ebm_soc_min=args.optimistic_ebm_soc_min,
        optimistic_ebm_soc_max=args.optimistic_ebm_soc_max,
        optimistic_ebm_power_scale=args.optimistic_ebm_power_scale,
        optimistic_ebm_efficiency=args.optimistic_ebm_efficiency,
        optimistic_ebm_soc_penalty_scale=args.optimistic_ebm_soc_penalty_scale,
        model_path=pbm_model_path,
        tou_price_spread_multiplier=tou_price_spread_multiplier,
        grid_slack_enabled=grid_slack_enabled,
        nse_penalty_per_kwh=nse_penalty_per_kwh,
        curtailment_penalty_per_kwh=curtailment_penalty_per_kwh,
        grid_import_max_kw=grid_import_max_kw,
        grid_export_max_kw=grid_export_max_kw,
        generator_enabled=generator_enabled,
        force_cpu=bool(getattr(args, "cpu", False)),
        dqn_action_bins=dqn_action_bins,
        validation_selection_enabled=validation_selection_enabled,
        validation_days=validation_days,
        validation_start_hours=validation_start_hours,
        validation_interval_steps=validation_interval_steps,
        validation_eval_battery_model=validation_eval_battery_model,
        agent_hyperparams=agent_hyperparams,
        action_regularization=action_regularization,
    )
    ebm_agent, ebm_train_meta = train_agent(
        agent_name=agent_name,
        battery_model="simple",
        steps=args.steps,
        train_days=args.train_days,
        seed=seed,
        data_dir=args.data_dir,
        data_year=args.train_year,
        random_episode_start=args.random_train_window,
        episode_start_hour=args.train_start_hour,
        reward_mode=args.reward_mode,
        random_initial_soc=random_train_soc,
        initial_soc_min=args.train_soc_min,
        initial_soc_max=args.train_soc_max,
        full_year_random_start_stride_hours=RESIDENTIAL_FULL_YEAR_RANDOM_START_STRIDE_HOURS,
        peak_import_penalty_per_kw=args.peak_import_penalty_per_kw,
        peak_import_threshold_kw=args.peak_import_threshold_kw,
        monthly_demand_charge_per_kw=args.monthly_demand_charge_per_kw,
        monthly_demand_charge_threshold_kw=args.monthly_demand_charge_threshold_kw,
        monthly_peak_increment_penalty_per_kw=train_monthly_peak_increment_penalty_per_kw,
        battery_throughput_penalty_per_kwh=train_battery_throughput_penalty_per_kwh,
        battery_loss_penalty_per_kwh=train_battery_loss_penalty_per_kwh,
        battery_stress_penalty_per_kwh=train_battery_stress_penalty_per_kwh,
        midday_pv_boost_multiplier=args.midday_pv_boost_multiplier,
        evening_load_boost_multiplier=args.evening_load_boost_multiplier,
        stress_episode_sampling=ebm_stress_episode_sampling,
        stress_sampling_strength=args.stress_sampling_strength if ebm_stress_episode_sampling else 0.0,
        curriculum_days=ebm_curriculum_days,
        battery_power_scale=args.battery_power_scale,
        battery_energy_scale=args.battery_energy_scale,
        pbm_low_soc_r_int_threshold=args.pbm_low_soc_r_int_threshold,
        pbm_low_soc_r_int_factor=args.pbm_low_soc_r_int_factor,
        pbm_power_stress_r_int_factor=args.pbm_power_stress_r_int_factor,
        pbm_r_int_scale=args.pbm_r_int_scale,
        optimistic_ebm_training=optimistic_ebm_training,
        optimistic_ebm_soc_min=args.optimistic_ebm_soc_min,
        optimistic_ebm_soc_max=args.optimistic_ebm_soc_max,
        optimistic_ebm_power_scale=args.optimistic_ebm_power_scale,
        optimistic_ebm_efficiency=args.optimistic_ebm_efficiency,
        optimistic_ebm_soc_penalty_scale=args.optimistic_ebm_soc_penalty_scale,
        model_path=ebm_model_path,
        tou_price_spread_multiplier=tou_price_spread_multiplier,
        grid_slack_enabled=grid_slack_enabled,
        nse_penalty_per_kwh=nse_penalty_per_kwh,
        curtailment_penalty_per_kwh=curtailment_penalty_per_kwh,
        grid_import_max_kw=grid_import_max_kw,
        grid_export_max_kw=grid_export_max_kw,
        generator_enabled=generator_enabled,
        force_cpu=bool(getattr(args, "cpu", False)),
        dqn_action_bins=dqn_action_bins,
        validation_selection_enabled=validation_selection_enabled,
        validation_days=validation_days,
        validation_start_hours=validation_start_hours,
        validation_interval_steps=validation_interval_steps,
        validation_eval_battery_model=validation_eval_battery_model,
        agent_hyperparams=agent_hyperparams,
        action_regularization=action_regularization,
    )

    canonical_eval_start_hour = int(eval_start_hours[0]) if eval_start_hours else 0
    window_metrics: list[dict] = []
    window_summary_rows: list[dict] = []

    pbm_res = None
    ebm_res = None
    idle_res = None
    for start_hour in eval_start_hours or [canonical_eval_start_hour]:
        start_hour = max(int(start_hour), 0)
        suffix_hour = "" if start_hour == canonical_eval_start_hour else f"_sh{start_hour}"
        pbm_label = f"pbm_{suffix}{suffix_hour}"
        ebm_label = f"ebm_{suffix}{suffix_hour}"

        pbm_window = evaluate_rollout(
            pbm_agent,
            agent_name,
            pbm_label,
            args.eval_days,
            seed,
            args.data_dir,
            args.eval_year,
            start_hour,
            args.reward_mode,
            "thevenin",
            RESIDENTIAL_FULL_YEAR_RANDOM_START_STRIDE_HOURS,
            args.peak_import_penalty_per_kw,
            args.peak_import_threshold_kw,
            args.monthly_demand_charge_per_kw,
            args.monthly_demand_charge_threshold_kw,
            battery_throughput_penalty_per_kwh,
            args.battery_loss_penalty_per_kwh,
            args.battery_stress_penalty_per_kwh,
            args.midday_pv_boost_multiplier,
            args.evening_load_boost_multiplier,
            args.battery_power_scale,
            args.battery_energy_scale,
            args.pbm_low_soc_r_int_threshold,
            args.pbm_low_soc_r_int_factor,
            args.pbm_power_stress_r_int_factor,
            args.pbm_r_int_scale,
            tou_price_spread_multiplier=tou_price_spread_multiplier,
            grid_slack_enabled=grid_slack_enabled,
            nse_penalty_per_kwh=nse_penalty_per_kwh,
            curtailment_penalty_per_kwh=curtailment_penalty_per_kwh,
            grid_import_max_kw=grid_import_max_kw,
            grid_export_max_kw=grid_export_max_kw,
            generator_enabled=generator_enabled,
            dqn_action_bins=dqn_action_bins,
            action_regularization=action_regularization,
        )
        ebm_window = evaluate_rollout(
            ebm_agent,
            agent_name,
            ebm_label,
            args.eval_days,
            seed,
            args.data_dir,
            args.eval_year,
            start_hour,
            args.reward_mode,
            "thevenin",
            RESIDENTIAL_FULL_YEAR_RANDOM_START_STRIDE_HOURS,
            args.peak_import_penalty_per_kw,
            args.peak_import_threshold_kw,
            args.monthly_demand_charge_per_kw,
            args.monthly_demand_charge_threshold_kw,
            battery_throughput_penalty_per_kwh,
            args.battery_loss_penalty_per_kwh,
            args.battery_stress_penalty_per_kwh,
            args.midday_pv_boost_multiplier,
            args.evening_load_boost_multiplier,
            args.battery_power_scale,
            args.battery_energy_scale,
            args.pbm_low_soc_r_int_threshold,
            args.pbm_low_soc_r_int_factor,
            args.pbm_power_stress_r_int_factor,
            args.pbm_r_int_scale,
            optimistic_ebm_training=optimistic_ebm_training,
            optimistic_ebm_soc_min=args.optimistic_ebm_soc_min,
            optimistic_ebm_soc_max=args.optimistic_ebm_soc_max,
            optimistic_ebm_power_scale=args.optimistic_ebm_power_scale,
            optimistic_ebm_efficiency=args.optimistic_ebm_efficiency,
            optimistic_ebm_soc_penalty_scale=args.optimistic_ebm_soc_penalty_scale,
            tou_price_spread_multiplier=tou_price_spread_multiplier,
            grid_slack_enabled=grid_slack_enabled,
            nse_penalty_per_kwh=nse_penalty_per_kwh,
            curtailment_penalty_per_kwh=curtailment_penalty_per_kwh,
            grid_import_max_kw=grid_import_max_kw,
            grid_export_max_kw=grid_export_max_kw,
            generator_enabled=generator_enabled,
            dqn_action_bins=dqn_action_bins,
            action_regularization=action_regularization,
        )
        idle_window = evaluate_idle_baseline(
            args.eval_days,
            seed,
            args.data_dir,
            args.eval_year,
            start_hour,
            args.reward_mode,
            args.peak_import_penalty_per_kw,
            args.peak_import_threshold_kw,
            args.monthly_demand_charge_per_kw,
            args.monthly_demand_charge_threshold_kw,
            battery_throughput_penalty_per_kwh,
            args.battery_loss_penalty_per_kwh,
            args.battery_stress_penalty_per_kwh,
            args.midday_pv_boost_multiplier,
            args.evening_load_boost_multiplier,
            args.battery_power_scale,
            args.battery_energy_scale,
            args.pbm_low_soc_r_int_threshold,
            args.pbm_low_soc_r_int_factor,
            args.pbm_power_stress_r_int_factor,
            args.pbm_r_int_scale,
            tou_price_spread_multiplier=tou_price_spread_multiplier,
            grid_slack_enabled=grid_slack_enabled,
            nse_penalty_per_kwh=nse_penalty_per_kwh,
            curtailment_penalty_per_kwh=curtailment_penalty_per_kwh,
            grid_import_max_kw=grid_import_max_kw,
            grid_export_max_kw=grid_export_max_kw,
            generator_enabled=generator_enabled,
        )

        pbm_cost = float(pbm_window["total_cost"])
        ebm_cost = float(ebm_window["total_cost"])
        gap_pct = 100.0 * (ebm_cost - pbm_cost) / max(abs(pbm_cost), 1e-9)
        pbm_win = pbm_cost <= ebm_cost + 1e-9
        window_metrics.append(
            {
                "start_hour": int(start_hour),
                "pbm_cost": pbm_cost,
                "ebm_cost": ebm_cost,
                "idle_cost": float(idle_window["total_cost"]),
                "gap_pct": float(gap_pct),
                "pbm_win": bool(pbm_win),
            }
        )

        window_summary_rows.append({"agent": agent_name, "start_hour": int(start_hour), "method": "pbm", **_strip_timeseries(pbm_window)})
        window_summary_rows.append({"agent": agent_name, "start_hour": int(start_hour), "method": "ebm", **_strip_timeseries(ebm_window)})

        if start_hour == canonical_eval_start_hour:
            pbm_res = pbm_window
            ebm_res = ebm_window
            idle_res = idle_window
            export_timeseries(pbm_window, output_dir / f"pbm_{suffix}_timeseries.csv")
            export_timeseries(ebm_window, output_dir / f"ebm_{suffix}_timeseries.csv")
        else:
            export_timeseries(pbm_window, output_dir / f"pbm_{suffix}_sh{start_hour}_timeseries.csv")
            export_timeseries(ebm_window, output_dir / f"ebm_{suffix}_sh{start_hour}_timeseries.csv")

    if pbm_res is None or ebm_res is None or idle_res is None:
        raise RuntimeError("Evaluation did not produce canonical PBM/EBM results")

    gaps = [float(item["gap_pct"]) for item in window_metrics]
    win_rate = float(np.mean([bool(item["pbm_win"]) for item in window_metrics])) if window_metrics else 0.0
    window_gap_pct_mean = float(np.mean(gaps)) if gaps else 0.0
    window_gap_pct_worst = float(np.min(gaps)) if gaps else 0.0

    canonical_gap_pct = 100.0 * (float(ebm_res["total_cost"]) - float(pbm_res["total_cost"])) / max(abs(float(pbm_res["total_cost"])), 1e-9)

    simple_eval_pbm = evaluate_rollout(
        pbm_agent,
        agent_name,
        f"pbm_{suffix}_eval_simple",
        args.eval_days,
        seed,
        args.data_dir,
        args.eval_year,
        canonical_eval_start_hour,
        args.reward_mode,
        "simple",
        RESIDENTIAL_FULL_YEAR_RANDOM_START_STRIDE_HOURS,
        args.peak_import_penalty_per_kw,
        args.peak_import_threshold_kw,
        args.monthly_demand_charge_per_kw,
        args.monthly_demand_charge_threshold_kw,
        battery_throughput_penalty_per_kwh,
        args.battery_loss_penalty_per_kwh,
        args.battery_stress_penalty_per_kwh,
        args.midday_pv_boost_multiplier,
        args.evening_load_boost_multiplier,
        args.battery_power_scale,
        args.battery_energy_scale,
        args.pbm_low_soc_r_int_threshold,
        args.pbm_low_soc_r_int_factor,
        args.pbm_power_stress_r_int_factor,
        args.pbm_r_int_scale,
        optimistic_ebm_training=optimistic_ebm_training,
        optimistic_ebm_soc_min=args.optimistic_ebm_soc_min,
        optimistic_ebm_soc_max=args.optimistic_ebm_soc_max,
        optimistic_ebm_power_scale=args.optimistic_ebm_power_scale,
        optimistic_ebm_efficiency=args.optimistic_ebm_efficiency,
        optimistic_ebm_soc_penalty_scale=args.optimistic_ebm_soc_penalty_scale,
        tou_price_spread_multiplier=tou_price_spread_multiplier,
        grid_slack_enabled=grid_slack_enabled,
        nse_penalty_per_kwh=nse_penalty_per_kwh,
        curtailment_penalty_per_kwh=curtailment_penalty_per_kwh,
        grid_import_max_kw=grid_import_max_kw,
        grid_export_max_kw=grid_export_max_kw,
        generator_enabled=generator_enabled,
        record_timeseries=False,
        dqn_action_bins=dqn_action_bins,
        action_regularization=action_regularization,
    )
    simple_eval_ebm = evaluate_rollout(
        ebm_agent,
        agent_name,
        f"ebm_{suffix}_eval_simple",
        args.eval_days,
        seed,
        args.data_dir,
        args.eval_year,
        canonical_eval_start_hour,
        args.reward_mode,
        "simple",
        RESIDENTIAL_FULL_YEAR_RANDOM_START_STRIDE_HOURS,
        args.peak_import_penalty_per_kw,
        args.peak_import_threshold_kw,
        args.monthly_demand_charge_per_kw,
        args.monthly_demand_charge_threshold_kw,
        battery_throughput_penalty_per_kwh,
        args.battery_loss_penalty_per_kwh,
        args.battery_stress_penalty_per_kwh,
        args.midday_pv_boost_multiplier,
        args.evening_load_boost_multiplier,
        args.battery_power_scale,
        args.battery_energy_scale,
        args.pbm_low_soc_r_int_threshold,
        args.pbm_low_soc_r_int_factor,
        args.pbm_power_stress_r_int_factor,
        args.pbm_r_int_scale,
        optimistic_ebm_training=optimistic_ebm_training,
        optimistic_ebm_soc_min=args.optimistic_ebm_soc_min,
        optimistic_ebm_soc_max=args.optimistic_ebm_soc_max,
        optimistic_ebm_power_scale=args.optimistic_ebm_power_scale,
        optimistic_ebm_efficiency=args.optimistic_ebm_efficiency,
        optimistic_ebm_soc_penalty_scale=args.optimistic_ebm_soc_penalty_scale,
        tou_price_spread_multiplier=tou_price_spread_multiplier,
        grid_slack_enabled=grid_slack_enabled,
        nse_penalty_per_kwh=nse_penalty_per_kwh,
        curtailment_penalty_per_kwh=curtailment_penalty_per_kwh,
        grid_import_max_kw=grid_import_max_kw,
        grid_export_max_kw=grid_export_max_kw,
        generator_enabled=generator_enabled,
        record_timeseries=False,
        dqn_action_bins=dqn_action_bins,
        action_regularization=action_regularization,
    )

    eval_matrix_rows = [
        {"train_model": "pbm", "eval_model": "thevenin", **_strip_timeseries(pbm_res)},
        {"train_model": "ebm", "eval_model": "thevenin", **_strip_timeseries(ebm_res)},
        {"train_model": "pbm", "eval_model": "simple", **_strip_timeseries(simple_eval_pbm)},
        {"train_model": "ebm", "eval_model": "simple", **_strip_timeseries(simple_eval_ebm)},
    ]
    pd.DataFrame(eval_matrix_rows).to_csv(output_dir / "eval_matrix.csv", index=False)

    reverse_mfm_gap_pct = 100.0 * (float(simple_eval_pbm["total_cost"]) - float(simple_eval_ebm["total_cost"])) / max(
        abs(float(simple_eval_ebm["total_cost"])), 1e-9
    )
    report = {
        "agent": agent_name,
        "dqn_action_bins": int(dqn_action_bins),
        "seed": int(seed),
        "steps": int(args.steps),
        "pbm_steps": int(pbm_steps),
        "pbm_step_multiplier": float(args.pbm_step_multiplier),
        "train_year": int(args.train_year),
        "eval_year": int(args.eval_year),
        "eval_start_hour": int(canonical_eval_start_hour),
        "eval_start_hours": [int(x) for x in eval_start_hours],
        "pbm_cost": float(pbm_res["total_cost"]),
        "ebm_cost": float(ebm_res["total_cost"]),
        "idle_cost": float(idle_res["total_cost"]),
        "idle_monthly_demand_charge_sum": float(idle_res.get("monthly_demand_charge_sum", 0.0)),
        "pbm_value_vs_idle": float(idle_res["total_cost"]) - float(pbm_res["total_cost"]),
        "ebm_value_vs_idle": float(idle_res["total_cost"]) - float(ebm_res["total_cost"]),
        "gap_pct": float(canonical_gap_pct),
        "window_gap_pct_mean": float(window_gap_pct_mean),
        "window_gap_pct_worst": float(window_gap_pct_worst),
        "window_win_rate": float(win_rate),
        "window_metrics": window_metrics,
        "validation_selection_enabled": bool(validation_selection_enabled),
        "validation_days": int(validation_days),
        "validation_interval_steps": int(validation_interval_steps),
        "validation_start_hours": [int(hour) for hour in validation_start_hours],
        "validation_eval_battery_model": str(validation_eval_battery_model),
        "agent_hyperparams": agent_hyperparams,
        "action_regularization": action_regularization,
        "pbm_training_meta": pbm_train_meta,
        "ebm_training_meta": ebm_train_meta,
        "eval_matrix": {
            "pbm_eval_pbm_cost": float(pbm_res["total_cost"]),
            "ebm_eval_pbm_cost": float(ebm_res["total_cost"]),
            "pbm_eval_ebm_cost": float(simple_eval_pbm["total_cost"]),
            "ebm_eval_ebm_cost": float(simple_eval_ebm["total_cost"]),
            "mfm_gap_pct_ebm_to_pbm": float(canonical_gap_pct),
            "reverse_mfm_gap_pct_pbm_to_ebm": float(reverse_mfm_gap_pct),
        },
        "paper_primary_metric_name": "common_nl_env_gap_pct",
        "paper_primary_eval_env_battery_model": "thevenin",
        "paper_primary_pbm_cost": float(pbm_res["total_cost"]),
        "paper_primary_ebm_cost": float(ebm_res["total_cost"]),
        "paper_primary_gap_pct": float(canonical_gap_pct),
        "pbm_clip_ratio": pbm_res["clip_ratio"],
        "ebm_clip_ratio": ebm_res["clip_ratio"],
        "pbm_p_actual_abs_p95_kw": pbm_res["p_actual_abs_p95_kw"],
        "ebm_p_actual_abs_p95_kw": ebm_res["p_actual_abs_p95_kw"],
        "pbm_max_grid_import_kw": pbm_res.get("max_grid_import_kw", 0.0),
        "ebm_max_grid_import_kw": ebm_res.get("max_grid_import_kw", 0.0),
        "pbm_throughput_kwh": pbm_res["throughput_kwh"],
        "ebm_throughput_kwh": ebm_res["throughput_kwh"],
        "pbm_battery_loss_kwh_sum": pbm_res.get("battery_loss_kwh_sum", 0.0),
        "ebm_battery_loss_kwh_sum": ebm_res.get("battery_loss_kwh_sum", 0.0),
        "pbm_battery_current_rms_a": pbm_res.get("battery_current_rms_a", 0.0),
        "ebm_battery_current_rms_a": ebm_res.get("battery_current_rms_a", 0.0),
        "pbm_battery_temperature_max_c": pbm_res.get("battery_temperature_max_c", 0.0),
        "ebm_battery_temperature_max_c": ebm_res.get("battery_temperature_max_c", 0.0),
        "pbm_battery_ocv_hysteresis_abs_p95_v": pbm_res.get("battery_ocv_hysteresis_abs_p95_v", 0.0),
        "ebm_battery_ocv_hysteresis_abs_p95_v": ebm_res.get("battery_ocv_hysteresis_abs_p95_v", 0.0),
        "pbm_battery_polarization_abs_p95_v": pbm_res.get("battery_polarization_abs_p95_v", 0.0),
        "ebm_battery_polarization_abs_p95_v": ebm_res.get("battery_polarization_abs_p95_v", 0.0),
        "pbm_monthly_demand_charge_sum": pbm_res["monthly_demand_charge_sum"],
        "ebm_monthly_demand_charge_sum": ebm_res["monthly_demand_charge_sum"],
        "pbm_battery_throughput_penalty_sum": pbm_res["battery_throughput_penalty_sum"],
        "ebm_battery_throughput_penalty_sum": ebm_res["battery_throughput_penalty_sum"],
        "pbm_battery_loss_penalty_sum": pbm_res["battery_loss_penalty_sum"],
        "ebm_battery_loss_penalty_sum": ebm_res["battery_loss_penalty_sum"],
        "pbm_battery_stress_penalty_sum": pbm_res["battery_stress_penalty_sum"],
        "ebm_battery_stress_penalty_sum": ebm_res["battery_stress_penalty_sum"],
        "pbm_generator_cost_sum": pbm_res["generator_cost_sum"],
        "ebm_generator_cost_sum": ebm_res["generator_cost_sum"],
        "pbm_generator_low_load_penalty_sum": pbm_res["generator_low_load_penalty_sum"],
        "ebm_generator_low_load_penalty_sum": ebm_res["generator_low_load_penalty_sum"],
        "pbm_generator_low_load_hours": pbm_res["generator_low_load_hours"],
        "ebm_generator_low_load_hours": ebm_res["generator_low_load_hours"],
        "pbm_generator_energy_kwh": pbm_res["generator_energy_kwh"],
        "ebm_generator_energy_kwh": ebm_res["generator_energy_kwh"],
        "pbm_generator_power_p95_kw": pbm_res["generator_power_p95_kw"],
        "ebm_generator_power_p95_kw": ebm_res["generator_power_p95_kw"],
        "random_train_soc": bool(random_train_soc),
        "full_year_random_start_stride_hours": int(RESIDENTIAL_FULL_YEAR_RANDOM_START_STRIDE_HOURS),
        "full_year_random_start_hours": [int(hour) for hour in getattr(pbm_debug_config, "full_year_random_start_hours", ())],
        "auto_final_phase_random_window": bool(RESIDENTIAL_AUTO_FINAL_PHASE_RANDOM_WINDOW and len(curriculum_days) > 1),
        "stress_episode_sampling": bool(stress_episode_sampling),
        "ebm_stress_episode_sampling": bool(ebm_stress_episode_sampling),
        "curriculum_days": curriculum_days,
        "peak_import_penalty_per_kw": float(args.peak_import_penalty_per_kw),
        "peak_import_threshold_kw": float(args.peak_import_threshold_kw),
        "monthly_demand_charge_per_kw": float(args.monthly_demand_charge_per_kw),
        "monthly_demand_charge_threshold_kw": float(args.monthly_demand_charge_threshold_kw),
        "monthly_peak_increment_penalty_per_kw": float(args.monthly_peak_increment_penalty_per_kw),
        "train_monthly_peak_increment_penalty_per_kw": float(train_monthly_peak_increment_penalty_per_kw),
        "battery_throughput_penalty_per_kwh": float(battery_throughput_penalty_per_kwh),
        "battery_loss_penalty_per_kwh": float(args.battery_loss_penalty_per_kwh),
        "battery_stress_penalty_per_kwh": float(args.battery_stress_penalty_per_kwh),
        "train_battery_throughput_penalty_per_kwh": float(train_battery_throughput_penalty_per_kwh),
        "train_battery_loss_penalty_per_kwh": float(train_battery_loss_penalty_per_kwh),
        "train_battery_stress_penalty_per_kwh": float(train_battery_stress_penalty_per_kwh),
        "battery_degradation_model": degradation_meta,
        "midday_pv_boost_multiplier": float(args.midday_pv_boost_multiplier),
        "evening_load_boost_multiplier": float(args.evening_load_boost_multiplier),
        "cpu": bool(getattr(args, "cpu", False)),
        "islanded": bool(getattr(args, "islanded", False)),
        "grid_slack_enabled": bool(grid_slack_enabled),
        "nse_penalty_per_kwh": float(nse_penalty_per_kwh),
        "curtailment_penalty_per_kwh": float(curtailment_penalty_per_kwh),
        "tou_price_spread_multiplier": float(tou_price_spread_multiplier),
        "grid_import_max_kw": float(grid_import_max_kw),
        "grid_export_max_kw": float(grid_export_max_kw),
        "generator_enabled": bool(getattr(pbm_debug_config, "generator_enabled", False) and pbm_debug_config.generator_params is not None),
        "observation_stack_steps": int(getattr(pbm_debug_config, "observation_stack_steps", 1)),
        "battery_power_scale": float(args.battery_power_scale),
        "battery_energy_scale": float(args.battery_energy_scale),
        "pbm_low_soc_r_int_threshold": float(args.pbm_low_soc_r_int_threshold),
        "pbm_low_soc_r_int_factor": float(args.pbm_low_soc_r_int_factor),
        "pbm_power_stress_r_int_factor": float(args.pbm_power_stress_r_int_factor),
        "pbm_r_int_scale": float(args.pbm_r_int_scale),
        "optimistic_ebm_training": bool(optimistic_ebm_training),
        "optimistic_ebm_power_scale": float(args.optimistic_ebm_power_scale),
        "optimistic_ebm_soc_penalty_scale": float(args.optimistic_ebm_soc_penalty_scale),
        **storage_design,
        **{f"generator_{key}": value for key, value in generator_design.items()},
        "debug_config": {
            "pbm_train_battery": battery_debug_summary(pbm_debug_config.battery_params),
            "ebm_train_battery": battery_debug_summary(ebm_debug_config.battery_params),
            "generator": generator_debug_summary(pbm_debug_config.generator_params),
            "training_regime": {
                "agent": agent_name,
                "dqn_action_bins": int(dqn_action_bins),
                "fair_train": bool(fair_train),
                "pbm_steps": int(pbm_steps),
                "ebm_steps": int(args.steps),
                "pbm_replay_buffer_size": int(pbm_replay_buffer_size),
                "ebm_replay_buffer_size": int(ebm_replay_buffer_size),
                "pbm_curriculum_days": [int(day) for day in curriculum_days],
                "ebm_curriculum_days": [int(day) for day in ebm_curriculum_days],
                "pbm_phase_steps": [int(step) for step in allocate_phase_steps(int(pbm_steps), curriculum_days)],
                "ebm_phase_steps": [int(step) for step in allocate_phase_steps(int(args.steps), ebm_curriculum_days)],
                "pbm_stress_episode_sampling": bool(stress_episode_sampling),
                "ebm_stress_episode_sampling": bool(ebm_stress_episode_sampling),
                "random_train_soc": bool(random_train_soc),
                "curriculum_wide_soc_min": float(RESIDENTIAL_CURRICULUM_WIDE_SOC_MIN),
                "curriculum_final_soc_min": float(RESIDENTIAL_CURRICULUM_FINAL_SOC_MIN),
                "curriculum_wide_soc_max": float(RESIDENTIAL_CURRICULUM_WIDE_SOC_MAX),
                "auto_final_phase_random_window": bool(RESIDENTIAL_AUTO_FINAL_PHASE_RANDOM_WINDOW and len(curriculum_days) > 1),
                "full_year_random_start_mode": "daily_stride" if RESIDENTIAL_AUTO_FINAL_PHASE_RANDOM_WINDOW and len(curriculum_days) > 1 else "fixed_or_cli",
                "full_year_random_start_stride_hours": int(RESIDENTIAL_FULL_YEAR_RANDOM_START_STRIDE_HOURS),
                "full_year_random_start_hours": [int(hour) for hour in getattr(pbm_debug_config, "full_year_random_start_hours", ())],
                "validation_selection_enabled": bool(validation_selection_enabled),
                "validation_days": int(validation_days),
                "validation_interval_steps": int(validation_interval_steps),
                "validation_start_hours": [int(hour) for hour in validation_start_hours],
                "validation_eval_battery_model": str(validation_eval_battery_model),
                "agent_hyperparams": agent_hyperparams,
                "action_regularization": action_regularization,
            },
            "pbm_training_meta": pbm_train_meta,
            "ebm_training_meta": ebm_train_meta,
            "pbm_eval_trace": action_trace_summary(pbm_res),
            "ebm_eval_trace": action_trace_summary(ebm_res),
        },
    }

    summary_df = pd.DataFrame([pbm_res, ebm_res]).drop(columns=SUMMARY_DROP_COLUMNS)
    summary_df.insert(0, "agent", agent_name)
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    pd.DataFrame(window_summary_rows).to_csv(output_dir / "summary_windows.csv", index=False)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    return {
        "agent": agent_name,
        "seed": int(seed),
        "pbm_cost": float(pbm_res["total_cost"]),
        "ebm_cost": float(ebm_res["total_cost"]),
        "gap_pct": float(canonical_gap_pct),
        "window_gap_pct_mean": float(window_gap_pct_mean),
        "window_gap_pct_worst": float(window_gap_pct_worst),
        "window_win_rate": float(win_rate),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate PBM vs EBM on MG-RES with year-split data")
    parser.add_argument("--agent", type=str, default="sac", choices=list(SUPPORTED_AGENT_NAMES))
    parser.add_argument("--dqn-action-bins", type=int, default=RESIDENTIAL_DQN_ACTION_BINS)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--train-days", type=int, default=365)
    parser.add_argument("--eval-days", type=int, default=365)
    parser.add_argument("--train-year", type=int, default=2023)
    parser.add_argument("--eval-year", type=int, default=2024)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds (overrides --seed)")
    parser.add_argument("--data-dir", type=str, default='data')
    parser.add_argument("--train-start-hour", type=int, default=0)
    parser.add_argument("--eval-start-hour", type=int, default=0, help="Evaluation window start hour (single)")
    parser.add_argument("--eval-start-hours", type=str, default=None, help="Comma-separated evaluation start hours (overrides --eval-start-hour)")
    parser.add_argument("--reward-mode", type=str, default="cost", choices=["cost", "legacy"])
    parser.add_argument("--cpu", action="store_true", help="Force CPU training (helps avoid CUDA OOM).")
    parser.add_argument("--fair-train", action="store_true", help="Align PBM/EBM training protocol (curriculum + stress sampling) for fairness.")
    parser.add_argument("--random-train-window", action="store_true")
    parser.add_argument("--disable-random-train-soc", action="store_true")
    parser.add_argument("--train-soc-min", type=float, default=0.35)
    parser.add_argument("--train-soc-max", type=float, default=0.80)
    parser.add_argument("--islanded", action="store_true", help="Disable grid slack; unmet demand is NSE and surplus is curtailment.")
    parser.add_argument("--nse-penalty-per-kwh", type=float, default=100.0, help="Penalty ($/kWh) for non-served energy when islanded.")
    parser.add_argument("--curtailment-penalty-per-kwh", type=float, default=0.0, help="Penalty ($/kWh) for curtailment when islanded.")
    parser.add_argument(
        "--tou-price-spread-multiplier",
        type=float,
        default=RESIDENTIAL_GRID_TOU_PRICE_SPREAD_MULTIPLIER,
        help="Stretch TOU prices around their midpoint (grid-connected only).",
    )
    parser.add_argument("--grid-import-max-kw", type=float, default=float("inf"), help="Grid import limit (kW) when grid slack is enabled.")
    parser.add_argument("--grid-export-max-kw", type=float, default=float("inf"), help="Grid export limit (kW) when grid slack is enabled.")
    parser.add_argument("--enable-generator", action="store_true", help="Enable the optional residential backup generator.")
    parser.add_argument("--disable-generator", action="store_true", help="Force-disable the residential backup generator.")
    parser.add_argument("--peak-import-penalty-per-kw", type=float, default=0.0)
    parser.add_argument("--peak-import-threshold-kw", type=float, default=20.0)
    parser.add_argument("--monthly-demand-charge-per-kw", type=float, default=RESIDENTIAL_GRID_MONTHLY_DEMAND_CHARGE_PER_KW)
    parser.add_argument(
        "--monthly-demand-charge-threshold-kw",
        type=float,
        default=RESIDENTIAL_GRID_MONTHLY_DEMAND_CHARGE_THRESHOLD_W / 1000.0,
    )
    parser.add_argument("--monthly-peak-increment-penalty-per-kw", type=float, default=0.0)
    parser.add_argument(
        "--battery-throughput-penalty-per-kwh",
        type=float,
        default=0.0,
        help="Use a non-negative direct $/kWh throughput penalty, or a negative value to derive it from replacement-cost metadata.",
    )
    parser.add_argument("--battery-replacement-cost-per-kwh", type=float, default=RESIDENTIAL_GRID_BATTERY_REPLACEMENT_COST_PER_KWH)
    parser.add_argument("--battery-equivalent-full-cycles", type=float, default=RESIDENTIAL_GRID_BATTERY_EQUIVALENT_FULL_CYCLES)
    parser.add_argument("--battery-end-of-life-fraction", type=float, default=RESIDENTIAL_GRID_BATTERY_END_OF_LIFE_FRACTION)
    parser.add_argument(
        "--battery-degradation-cost-multiplier",
        type=float,
        default=RESIDENTIAL_GRID_BATTERY_DEGRADATION_COST_MULTIPLIER,
    )
    parser.add_argument("--battery-loss-penalty-per-kwh", type=float, default=0.0)
    parser.add_argument("--battery-stress-penalty-per-kwh", type=float, default=0.0)
    parser.add_argument(
        "--train-battery-throughput-penalty-per-kwh",
        type=float,
        default=None,
        help="Optional training-only throughput penalty override; evaluation still uses --battery-throughput-penalty-per-kwh.",
    )
    parser.add_argument(
        "--train-battery-loss-penalty-per-kwh",
        type=float,
        default=None,
        help="Optional training-only loss penalty override; evaluation still uses --battery-loss-penalty-per-kwh.",
    )
    parser.add_argument(
        "--train-battery-stress-penalty-per-kwh",
        type=float,
        default=None,
        help="Optional training-only stress penalty override; evaluation still uses --battery-stress-penalty-per-kwh.",
    )
    parser.add_argument("--midday-pv-boost-multiplier", type=float, default=1.0)
    parser.add_argument("--evening-load-boost-multiplier", type=float, default=1.0)
    parser.add_argument("--enable-stress-episode-sampling", dest="stress_episode_sampling", action="store_true")
    parser.add_argument("--disable-stress-episode-sampling", dest="stress_episode_sampling", action="store_false")
    parser.add_argument("--enable-ebm-stress-episode-sampling", action="store_true")
    parser.add_argument("--stress-sampling-strength", type=float, default=8.0)
    parser.add_argument("--curriculum-days", type=str, default="120,365")
    parser.add_argument("--pbm-step-multiplier", type=float, default=1.0)
    parser.add_argument("--policy-net-arch", type=str, default="256,128,64", help="Comma-separated hidden sizes for actor/critic MLPs.")
    parser.add_argument("--rl-learning-rate", type=float, default=3e-4)
    parser.add_argument("--rl-learning-starts", type=int, default=1000)
    parser.add_argument("--rl-batch-size", type=int, default=384)
    parser.add_argument("--rl-gamma", type=float, default=0.985)
    parser.add_argument("--rl-tau", type=float, default=0.003)
    parser.add_argument("--sac-ent-coef", type=str, default=RESIDENTIAL_SAC_ENT_COEF)
    parser.add_argument("--sac-target-entropy-scale", type=float, default=RESIDENTIAL_SAC_TARGET_ENTROPY_SCALE)
    parser.add_argument("--td3-action-noise-sigma", type=float, default=0.10)
    parser.add_argument("--td3-policy-delay", type=int, default=2)
    parser.add_argument("--td3-target-policy-noise", type=float, default=0.2)
    parser.add_argument("--td3-target-noise-clip", type=float, default=0.5)
    parser.add_argument(
        "--action-smoothing-coef",
        type=float,
        default=0.0,
        help="Fraction of the previously applied continuous action retained at the next step.",
    )
    parser.add_argument(
        "--action-max-delta",
        type=float,
        default=0.0,
        help="Maximum allowed per-step change in each continuous action dimension (normalized units).",
    )
    parser.add_argument(
        "--action-rate-penalty",
        type=float,
        default=0.0,
        help="Training reward penalty per unit of continuous-action change.",
    )
    parser.add_argument(
        "--enable-symmetric-battery-action",
        action="store_true",
        help="Cap positive battery commands so charge/discharge share the same max power magnitude.",
    )
    parser.add_argument("--battery-power-scale", type=float, default=1.0)
    parser.add_argument("--battery-energy-scale", type=float, default=1.0)
    parser.add_argument("--pbm-low-soc-r-int-threshold", type=float, default=RESIDENTIAL_PBM_LOW_SOC_R_INT_THRESHOLD)
    parser.add_argument("--pbm-low-soc-r-int-factor", type=float, default=RESIDENTIAL_PBM_LOW_SOC_R_INT_FACTOR)
    parser.add_argument("--pbm-power-stress-r-int-factor", type=float, default=RESIDENTIAL_PBM_POWER_STRESS_R_INT_FACTOR)
    parser.add_argument("--pbm-r-int-scale", type=float, default=1.0)
    parser.add_argument("--enable-optimistic-ebm", action="store_true")
    parser.add_argument("--disable-optimistic-ebm", action="store_true")
    parser.add_argument("--optimistic-ebm-soc-min", type=float, default=0.0)
    parser.add_argument("--optimistic-ebm-soc-max", type=float, default=1.00)
    parser.add_argument("--optimistic-ebm-power-scale", type=float, default=1.0)
    parser.add_argument("--optimistic-ebm-efficiency", type=float, default=1.00)
    parser.add_argument("--optimistic-ebm-soc-penalty-scale", type=float, default=1.0)
    parser.add_argument("--enable-validation-selection", dest="validation_selection", action="store_true")
    parser.add_argument("--disable-validation-selection", dest="validation_selection", action="store_false")
    parser.add_argument("--validation-days", type=int, default=RESIDENTIAL_VALIDATION_SELECTION_DAYS)
    parser.add_argument("--validation-interval-steps", type=int, default=RESIDENTIAL_VALIDATION_INTERVAL_STEPS)
    parser.add_argument(
        "--validation-start-hours",
        type=str,
        default=None,
        help="Comma-separated start hours for checkpoint validation windows.",
    )
    parser.add_argument(
        "--validation-eval-battery-model",
        type=str,
        default="thevenin",
        choices=["thevenin", "simple"],
        help="Battery model used when scoring checkpoints during training.",
    )
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "results" / "residential_d4_smoke"))
    parser.add_argument("--models-dir", type=str, default=str(PROJECT_ROOT / "models" / "residential_d4_smoke"))
    parser.set_defaults(stress_episode_sampling=False)
    parser.set_defaults(validation_selection=False)
    args = parser.parse_args(argv)

    battery_throughput_penalty_per_kwh, degradation_meta = resolve_battery_throughput_penalty(
        direct_penalty_per_kwh=args.battery_throughput_penalty_per_kwh,
        replacement_cost_per_kwh=args.battery_replacement_cost_per_kwh,
        equivalent_full_cycles=args.battery_equivalent_full_cycles,
        end_of_life_fraction=args.battery_end_of_life_fraction,
        degradation_cost_multiplier=args.battery_degradation_cost_multiplier,
    )
    train_monthly_peak_increment_penalty_per_kw = max(float(args.monthly_peak_increment_penalty_per_kw), 0.0)
    train_battery_throughput_penalty_per_kwh = float(battery_throughput_penalty_per_kwh)
    train_battery_loss_penalty_per_kwh = max(float(args.battery_loss_penalty_per_kwh), 0.0)
    train_battery_stress_penalty_per_kwh = max(float(args.battery_stress_penalty_per_kwh), 0.0)
    storage_design = summarize_storage_design(
        MicrogridConfig().battery_params,
        power_scale=args.battery_power_scale,
        energy_scale=args.battery_energy_scale,
    )
    generator_requested = bool(getattr(args, "enable_generator", False)) and not bool(getattr(args, "disable_generator", False))
    generator_design = generator_debug_summary(MicrogridConfig().generator_params if generator_requested else None)

    seeds = parse_int_list(args.seeds) or [int(args.seed)]
    seeds = _dedupe_ints(seeds)

    eval_start_hours = parse_int_list(args.eval_start_hours)
    if not eval_start_hours:
        eval_start_hours = [int(args.eval_start_hour)]
    eval_start_hours = _dedupe_ints([max(int(hour), 0) for hour in eval_start_hours])

    output_root = Path(args.output_dir)
    models_root = Path(args.models_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    models_root.mkdir(parents=True, exist_ok=True)

    aggregate_rows: list[dict] = []
    multi_seed = len(seeds) > 1
    for seed in seeds:
        seed_output_dir = output_root / f"seed{seed}" if multi_seed else output_root
        seed_models_dir = models_root / f"seed{seed}" if multi_seed else models_root
        result = run_seed_experiment(
            args=args,
            seed=int(seed),
            output_dir=seed_output_dir,
            models_dir=seed_models_dir,
            eval_start_hours=eval_start_hours,
            battery_throughput_penalty_per_kwh=battery_throughput_penalty_per_kwh,
            degradation_meta=degradation_meta,
            storage_design=storage_design,
            generator_design=generator_design,
        )
        aggregate_rows.append(result)

    if multi_seed:
        summary_path = output_root / "summary_seeds.csv"
        summary_df = pd.DataFrame(aggregate_rows)
        if summary_path.exists():
            try:
                existing = pd.read_csv(summary_path)
            except Exception:
                existing = None
            if existing is not None and not existing.empty:
                combined = pd.concat([existing, summary_df], ignore_index=True)
                if "agent" in combined.columns and "seed" in combined.columns:
                    combined["seed"] = combined["seed"].astype(int)
                    combined = combined.drop_duplicates(subset=["agent", "seed"], keep="last").sort_values(["agent", "seed"])
                elif "seed" in combined.columns:
                    combined["seed"] = combined["seed"].astype(int)
                    combined = combined.drop_duplicates(subset=["seed"], keep="last").sort_values("seed")
                summary_df = combined
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
