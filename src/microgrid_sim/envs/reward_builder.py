"""Reward construction for the network microgrid environment."""

from __future__ import annotations

from ..network.constraints import compute_loading_violation, compute_voltage_violations


def compute_soc_shaping_penalties(*, soc: float, reward_cfg) -> tuple[float, float]:
    soc_sigma = max(float(reward_cfg.soc_sigma), 1e-6)
    soc_center_distance = abs(float(soc) - float(reward_cfg.soc_center)) / soc_sigma
    soc_center_penalty = float(reward_cfg.w_band) * (soc_center_distance**2)
    soc_edge_penalty = float(reward_cfg.w_edge) * ((soc_center_distance**2) / (1.0 + soc_center_distance))
    return float(soc_center_penalty), float(soc_edge_penalty)


def build_network_reward(
    config,
    battery_info: dict,
    metrics: dict[str, float],
    import_cost: float,
    price: float | None = None,
    power_flow_result: dict | None = None,
    is_terminal: bool = False,
) -> tuple[float, dict[str, float]]:
    reward_cfg = config.reward
    power_flow_result = dict(power_flow_result or {})
    battery_dispatch_enabled = str(getattr(config, "battery_model", "")).lower() != "none"
    undervoltage, overvoltage = compute_voltage_violations(
        metrics,
        v_min=float(config.network_voltage_min_pu),
        v_max=float(config.network_voltage_max_pu),
    )
    line_overload = compute_loading_violation(metrics.get("max_line_loading_pct", 0.0), config.network_line_loading_limit_pct)
    trafo_overload = compute_loading_violation(
        metrics.get("max_transformer_loading_pct", 0.0),
        config.network_transformer_loading_limit_pct,
    )
    soc_violation = float(battery_info.get("soc_violation", 0.0)) if battery_dispatch_enabled else 0.0
    soc = float(battery_info.get("soc", getattr(getattr(config, "battery_params", None), "soc_init", 0.5)))
    current_price = float(price if price is not None else battery_info.get("price", 0.0))
    dt_hours = float(config.dt_seconds) / 3600.0
    if battery_dispatch_enabled:
        battery_throughput_kwh = abs(float(battery_info.get("effective_power", 0.0))) * dt_hours / 1000.0
        battery_loss_kwh = max(float(battery_info.get("power_loss", 0.0)), 0.0) * dt_hours / 1000.0
        battery_stress_proxy_kwh = battery_throughput_kwh * max(float(battery_info.get("r_int_power_factor", 1.0)) - 1.0, 0.0)
        soc_center_penalty, soc_edge_penalty = compute_soc_shaping_penalties(soc=soc, reward_cfg=reward_cfg)
        discharge_power_limit_w = max(
            float(
                battery_info.get(
                    "battery_discharge_power_limit",
                    battery_info.get("p_max", 0.0),
                )
            ),
            0.0,
        )
        p_discharge_max = max(float(getattr(getattr(config, "battery_params", None), "p_discharge_max", 0.0)), 1e-9)
        discharge_limit_ratio = min(discharge_power_limit_w / p_discharge_max, 1.0)
        peak_reserve_shortfall = 0.0
        peak_reserve_penalty = 0.0
        if current_price >= float(reward_cfg.peak_price):
            peak_reserve_shortfall = max(float(reward_cfg.peak_reserve_power_floor) - discharge_limit_ratio, 0.0)
            peak_reserve_penalty = float(reward_cfg.w_peak_reserve) * peak_reserve_shortfall
        terminal_soc_target = getattr(config, "terminal_soc_target", None)
        if terminal_soc_target is None:
            terminal_soc_target = getattr(getattr(config, "battery_params", None), "soc_init", soc)
        terminal_soc_tolerance = max(float(getattr(config, "terminal_soc_tolerance", 0.0)), 0.0)
        nominal_energy_kwh = max(float(getattr(getattr(config, "battery_params", None), "nominal_energy_wh", 0.0)) / 1000.0, 0.0)
        terminal_soc_penalty_per_kwh = max(float(getattr(config, "terminal_soc_penalty_per_kwh", 0.0)), 0.0)
        terminal_soc_deviation = abs(soc - float(terminal_soc_target))
        terminal_soc_excess = max(terminal_soc_deviation - terminal_soc_tolerance, 0.0)
        terminal_soc_excess_kwh = terminal_soc_excess * nominal_energy_kwh
        terminal_soc_penalty = terminal_soc_penalty_per_kwh * terminal_soc_excess_kwh if is_terminal else 0.0
    else:
        battery_throughput_kwh = 0.0
        battery_loss_kwh = 0.0
        battery_stress_proxy_kwh = 0.0
        soc_center_penalty = 0.0
        soc_edge_penalty = 0.0
        discharge_limit_ratio = 0.0
        peak_reserve_shortfall = 0.0
        peak_reserve_penalty = 0.0
        terminal_soc_target = getattr(getattr(config, "battery_params", None), "soc_init", 0.5)
        terminal_soc_tolerance = max(float(getattr(config, "terminal_soc_tolerance", 0.0)), 0.0)
        terminal_soc_penalty = 0.0
        terminal_soc_deviation = 0.0
        terminal_soc_excess = 0.0
        terminal_soc_excess_kwh = 0.0
    pf_failure_penalty = 0.0
    if bool(power_flow_result.get("failed", False)) or not bool(power_flow_result.get("converged", True)):
        pf_failure_penalty = abs(float(reward_cfg.reward_min))

    step_reward = (
        -reward_cfg.w_cost * float(import_cost)
        -reward_cfg.w_soc_violation * soc_violation
        -reward_cfg.w_voltage_violation * (undervoltage + overvoltage)
        -reward_cfg.w_line_overload * (line_overload / 100.0)
        -reward_cfg.w_transformer_overload * (trafo_overload / 100.0)
        -float(getattr(config, "battery_throughput_penalty_per_kwh", 0.0)) * battery_throughput_kwh
        -float(getattr(config, "battery_loss_penalty_per_kwh", 0.0)) * battery_loss_kwh
        -float(getattr(config, "battery_stress_penalty_per_kwh", 0.0)) * battery_stress_proxy_kwh
        -pf_failure_penalty
    )
    battery_shaping_penalty = float(soc_center_penalty + soc_edge_penalty + peak_reserve_penalty)
    clipped_step_reward = max(min(float(step_reward), reward_cfg.reward_max), reward_cfg.reward_min)
    reward_after_battery_shaping = clipped_step_reward - battery_shaping_penalty
    reward = reward_after_battery_shaping - float(terminal_soc_penalty)
    penalties = {
        "undervoltage": float(undervoltage),
        "overvoltage": float(overvoltage),
        "line_overload_pct": float(line_overload),
        "transformer_overload_pct": float(trafo_overload),
        "battery_throughput_kwh": float(battery_throughput_kwh),
        "battery_loss_kwh": float(battery_loss_kwh),
        "battery_stress_kwh": float(battery_stress_proxy_kwh),
        "soc_center_penalty": float(soc_center_penalty),
        "soc_edge_penalty": float(soc_edge_penalty),
        "peak_reserve_shortfall": float(peak_reserve_shortfall),
        "peak_reserve_penalty": float(peak_reserve_penalty),
        "discharge_limit_ratio": float(discharge_limit_ratio),
        "terminal_soc_target": float(terminal_soc_target),
        "terminal_soc_tolerance": float(terminal_soc_tolerance),
        "terminal_soc_deviation": float(terminal_soc_deviation),
        "terminal_soc_excess": float(terminal_soc_excess),
        "terminal_soc_excess_kwh": float(terminal_soc_excess_kwh),
        "terminal_soc_penalty": float(terminal_soc_penalty),
        "power_flow_failure_penalty": float(pf_failure_penalty),
        "step_reward_before_clip": float(step_reward),
        "step_reward_after_clip": float(clipped_step_reward),
        "battery_shaping_penalty": float(battery_shaping_penalty),
        "reward_after_battery_shaping": float(reward_after_battery_shaping),
        "reward_after_peak_reserve_penalty": float(clipped_step_reward - float(peak_reserve_penalty)),
        "reward_after_terminal_penalty": float(reward),
    }
    return reward, penalties
