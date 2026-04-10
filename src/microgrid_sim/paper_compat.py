"""Lightweight compatibility helpers for paper plotting and legacy model loading."""

from __future__ import annotations

from typing import Any

from .cases import CIGREEuropeanLVConfig
from .rl_utils import load_agent as _load_agent


def get_device(force_cpu: bool = False) -> str:
    import torch

    if not force_cpu and torch.cuda.is_available() and int(torch.cuda.device_count()) > 0:
        return "cuda"
    return "cpu"


def load_agent(agent_name: str, model_path: str, device: str = "cpu"):
    return _load_agent(agent_name, model_path, device=device)


def build_cigre_compat_config(
    battery_model: str,
    simulation_days: int,
    seed: int,
    data_dir: str | None,
    data_year: int,
    random_episode_start: bool,
    episode_start_hour: int,
    reward_mode: str,
    component_commitment_enabled: bool,
    include_component_cost_in_objective: bool,
    random_initial_soc: bool,
    initial_soc_min: float,
    initial_soc_max: float,
    price_spread_multiplier: float,
    peak_import_penalty_per_kw: float,
    peak_import_threshold_kw: float,
    midday_pv_boost_multiplier: float,
    evening_load_boost_multiplier: float,
    stress_episode_sampling: bool,
    stress_sampling_strength: float,
    battery_power_scale: float,
    battery_energy_scale: float,
    optimistic_ebm_training: bool,
    optimistic_ebm_soc_min: float,
    optimistic_ebm_soc_max: float,
    optimistic_ebm_power_scale: float,
    optimistic_ebm_efficiency: float,
    optimistic_ebm_soc_penalty_scale: float,
):
    cfg = CIGREEuropeanLVConfig(
        simulation_days=int(simulation_days),
        seed=int(seed),
        data_dir=data_dir,
        data_year=int(data_year),
        random_episode_start=bool(random_episode_start),
        episode_start_hour=int(episode_start_hour),
        random_initial_soc=bool(random_initial_soc),
        initial_soc_min=float(initial_soc_min),
        initial_soc_max=float(initial_soc_max),
        tou_price_spread_multiplier=float(price_spread_multiplier),
        battery_model=str(battery_model),
    )

    # Preserve paper-era fields as loose attributes so old plotting helpers can
    # keep reading metadata without reviving the legacy experiments package.
    legacy_attrs: dict[str, Any] = {
        "reward_mode": str(reward_mode),
        "component_commitment_enabled": bool(component_commitment_enabled),
        "include_component_cost_in_objective": bool(include_component_cost_in_objective),
        "peak_import_penalty_per_kw": float(peak_import_penalty_per_kw),
        "peak_import_threshold_kw": float(peak_import_threshold_kw),
        "midday_pv_boost_multiplier": float(midday_pv_boost_multiplier),
        "evening_load_boost_multiplier": float(evening_load_boost_multiplier),
        "stress_episode_sampling": bool(stress_episode_sampling),
        "stress_sampling_strength": float(stress_sampling_strength),
        "battery_power_scale": float(battery_power_scale),
        "battery_energy_scale": float(battery_energy_scale),
        "optimistic_ebm_training": bool(optimistic_ebm_training),
        "optimistic_ebm_soc_min": float(optimistic_ebm_soc_min),
        "optimistic_ebm_soc_max": float(optimistic_ebm_soc_max),
        "optimistic_ebm_power_scale": float(optimistic_ebm_power_scale),
        "optimistic_ebm_efficiency": float(optimistic_ebm_efficiency),
        "optimistic_ebm_soc_penalty_scale": float(optimistic_ebm_soc_penalty_scale),
    }
    for name, value in legacy_attrs.items():
        setattr(cfg, name, value)
    return cfg
