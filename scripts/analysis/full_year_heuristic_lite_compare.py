#!/usr/bin/env python3
"""Fast full-year heuristic and GA-lite style comparison against Oracle summaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from genetic_dispatch_baseline import _heuristic_seed_schedules, _initialize_env_state, _rollout_actions, build_config, build_env_from_config  # noqa: E402
from microgrid_sim.data.network_profiles import load_network_profiles  # noqa: E402
from microgrid_sim.models.battery import SimpleBattery  # noqa: E402
from microgrid_sim.time_utils import hours_to_steps, simulation_steps, steps_per_day  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Full-year heuristic and GA-lite comparison on IEEE33 and CIGRE.")
    parser.add_argument("--cases", type=str, default="ieee33,cigre", help="Comma-separated case keys")
    parser.add_argument("--regimes", type=str, default="network_stress", help="Comma-separated regimes")
    parser.add_argument("--reward-profile", type=str, default="paper_balanced", help="Reward profile")
    parser.add_argument("--battery-model", type=str, default="simple", help="Battery model")
    parser.add_argument("--days", type=int, default=365, help="Simulation days")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--baselines",
        type=str,
        default="none,heuristic_blended,heuristic_selector_lite",
        help="Comma-separated baselines: none, heuristic_peak_shave, heuristic_tou, heuristic_blended, heuristic_selector_lite",
    )
    parser.add_argument("--selector-window-days", type=int, default=7, help="Planning window for heuristic_selector_lite")
    parser.add_argument("--selector-stride-days", type=int, default=1, help="Execution stride for heuristic_selector_lite")
    parser.add_argument(
        "--evaluation-mode",
        type=str,
        default="surrogate",
        choices=("surrogate", "env"),
        help="Use fast energy-layer surrogate evaluation or full pandapower environment rollout.",
    )
    parser.add_argument("--voltage-penalty", type=float, default=2000.0, help="Analysis-only voltage penalty scale")
    parser.add_argument("--loading-penalty", type=float, default=250.0, help="Analysis-only loading penalty scale")
    parser.add_argument("--soc-penalty", type=float, default=1000.0, help="Analysis-only SOC violation penalty scale")
    parser.add_argument(
        "--oracle-summary-csv",
        type=Path,
        default=Path("results/full_year_oracle_compare_seed42/protocol_summary.csv"),
        help="Optional Oracle protocol summary CSV for gap calculation.",
    )
    parser.add_argument("--save-trajectories", action="store_true", help="Save actual rollout trajectories and selection logs")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/full_year_heuristic_lite_compare"),
        help="Output directory",
    )
    return parser


def _parse_csv_arg(raw: str) -> list[str]:
    return [item.strip().lower() for item in str(raw).split(",") if item.strip()]


def _heuristic_names_for_count(count: int) -> list[str]:
    base = ["zero", "peak_shave", "tou", "blended"]
    return base[:count]


def _power_from_action(action: float, charge_limit_w: float, discharge_limit_w: float) -> float:
    scalar = float(np.clip(action, -1.0, 1.0))
    if scalar >= 0.0:
        return scalar * float(discharge_limit_w)
    return scalar * float(charge_limit_w)


def _soc_penalty_terms(soc: float, config) -> tuple[float, float]:
    reward_cfg = config.reward
    soc_sigma = max(float(reward_cfg.soc_sigma), 1e-6)
    soc_center_penalty = float(reward_cfg.w_band) * (((float(soc) - float(reward_cfg.soc_center)) / soc_sigma) ** 2)
    soc_band_span = max(float(reward_cfg.soc_band_max) - float(reward_cfg.soc_band_min), 1e-6)
    if soc < float(reward_cfg.soc_band_min):
        soc_edge_distance = float(reward_cfg.soc_band_min) - float(soc)
    elif soc > float(reward_cfg.soc_band_max):
        soc_edge_distance = float(soc) - float(reward_cfg.soc_band_max)
    else:
        soc_edge_distance = 0.0
    soc_edge_penalty = float(reward_cfg.w_edge) * (soc_edge_distance / soc_band_span)
    return float(soc_center_penalty), float(soc_edge_penalty)


def _score_schedule_on_forecast(
    *,
    config,
    actions: np.ndarray,
    episode_start_hour: int,
    initial_soc: float,
    initial_temperature_c: float,
    include_terminal_penalty: bool,
) -> tuple[float, float, float]:
    profiles = load_network_profiles(config)
    start_step = max(hours_to_steps(int(episode_start_hour), config.dt_seconds), 0)
    horizon = len(np.asarray(actions).reshape(-1))
    end_step = start_step + horizon
    load_w = np.asarray(profiles.load_w[start_step:end_step], dtype=float)
    pv_w = np.asarray(profiles.pv_w[start_step:end_step], dtype=float)
    price = np.asarray(profiles.price[start_step:end_step], dtype=float)
    battery = SimpleBattery(config.battery_params)
    battery.reset(soc=float(initial_soc))
    battery.temperature_c = float(initial_temperature_c)
    dt_hours = float(config.dt_seconds) / 3600.0
    import_limit_w = float(config.grid_import_max) * 1_000_000.0 if np.isfinite(float(config.grid_import_max)) else float("inf")
    export_limit_w = float(config.grid_export_max) * 1_000_000.0 if np.isfinite(float(config.grid_export_max)) else float("inf")
    objective = 0.0
    terminal_soc_target = getattr(config, "terminal_soc_target", None)
    if terminal_soc_target is None:
        terminal_soc_target = getattr(config.battery_params, "soc_init", initial_soc)
    terminal_soc_tolerance = max(float(getattr(config, "terminal_soc_tolerance", 0.0)), 0.0)
    nominal_energy_kwh = max(float(getattr(config.battery_params, "nominal_energy_wh", 0.0)) / 1000.0, 0.0)
    terminal_soc_penalty_per_kwh = max(float(getattr(config, "terminal_soc_penalty_per_kwh", 0.0)), 0.0)

    for idx, raw_action in enumerate(np.asarray(actions, dtype=float).reshape(-1)):
        desired_power_w = _power_from_action(
            float(raw_action),
            charge_limit_w=float(config.battery_params.p_charge_max),
            discharge_limit_w=float(config.battery_params.p_discharge_max),
        )
        actual_power_w, _, battery_info = battery.step(desired_power_w, config.dt_seconds)
        net_grid_w = float(load_w[idx] - pv_w[idx] - actual_power_w)
        import_w = max(net_grid_w, 0.0)
        export_w = max(-net_grid_w, 0.0)
        import_cost = import_w * float(price[idx]) * dt_hours / 1000.0
        export_revenue = export_w * float(config.feed_in_tariff) * dt_hours / 1000.0
        import_violation_w = max(import_w - import_limit_w, 0.0) if np.isfinite(import_limit_w) else 0.0
        export_violation_w = max(export_w - export_limit_w, 0.0) if np.isfinite(export_limit_w) else 0.0
        grid_penalty_cost = (import_violation_w + export_violation_w) * float(config.grid_limit_violation_penalty_per_kwh) * dt_hours / 1000.0
        throughput_kwh = abs(float(battery_info.get("effective_power", 0.0))) * dt_hours / 1000.0
        loss_kwh = max(float(battery_info.get("power_loss", 0.0)), 0.0) * dt_hours / 1000.0
        stress_kwh = throughput_kwh * max(float(battery_info.get("r_int_power_factor", 1.0)) - 1.0, 0.0)
        objective += (
            import_cost
            - export_revenue
            + grid_penalty_cost
            + float(getattr(config, "battery_throughput_penalty_per_kwh", 0.0)) * throughput_kwh
            + float(getattr(config, "battery_loss_penalty_per_kwh", 0.0)) * loss_kwh
            + float(getattr(config, "battery_stress_penalty_per_kwh", 0.0)) * stress_kwh
        )

    if bool(include_terminal_penalty):
        terminal_soc_deviation = abs(float(battery.soc) - float(terminal_soc_target))
        terminal_soc_excess = max(terminal_soc_deviation - terminal_soc_tolerance, 0.0)
        terminal_soc_excess_kwh = terminal_soc_excess * nominal_energy_kwh
        objective += terminal_soc_penalty_per_kwh * terminal_soc_excess_kwh
    return float(objective), float(battery.soc), float(getattr(battery, "temperature_c", initial_temperature_c))


def evaluate_actions_on_surrogate(
    *,
    case_key: str,
    battery_model: str,
    regime: str,
    days: int,
    seed: int,
    reward_profile: str,
    actions: np.ndarray,
) -> dict[str, float | int | str]:
    config = build_config(case_key, battery_model, days, seed, regime, reward_profile)
    profiles = load_network_profiles(config)
    horizon = len(np.asarray(actions, dtype=float).reshape(-1))
    load_w = np.asarray(profiles.load_w[:horizon], dtype=float)
    pv_w = np.asarray(profiles.pv_w[:horizon], dtype=float)
    price = np.asarray(profiles.price[:horizon], dtype=float)
    timestamps = pd.DatetimeIndex(profiles.timestamps[:horizon])
    battery = SimpleBattery(config.battery_params)
    battery.reset(soc=float(config.battery_params.soc_init))
    dt_hours = float(config.dt_seconds) / 3600.0
    import_limit_w = float(config.grid_import_max) * 1_000_000.0 if np.isfinite(float(config.grid_import_max)) else float("inf")
    export_limit_w = float(config.grid_export_max) * 1_000_000.0 if np.isfinite(float(config.grid_export_max)) else float("inf")
    terminal_soc_target = getattr(config, "terminal_soc_target", None)
    if terminal_soc_target is None:
        terminal_soc_target = getattr(config.battery_params, "soc_init", battery.soc)
    terminal_soc_tolerance = max(float(getattr(config, "terminal_soc_tolerance", 0.0)), 0.0)
    nominal_energy_kwh = max(float(getattr(config.battery_params, "nominal_energy_wh", 0.0)) / 1000.0, 0.0)
    terminal_soc_penalty_per_kwh = max(float(getattr(config, "terminal_soc_penalty_per_kwh", 0.0)), 0.0)

    energy_cost = 0.0
    grid_penalty = 0.0
    throughput_penalty = 0.0
    loss_penalty = 0.0
    stress_penalty = 0.0
    soc_history: list[float] = []
    import_history_mw: list[float] = []
    line_loading_dummy: list[float] = []

    for idx, raw_action in enumerate(np.asarray(actions, dtype=float).reshape(-1)):
        desired_power_w = _power_from_action(
            float(raw_action),
            charge_limit_w=float(config.battery_params.p_charge_max),
            discharge_limit_w=float(config.battery_params.p_discharge_max),
        )
        actual_power_w, _, battery_info = battery.step(desired_power_w, config.dt_seconds)
        net_grid_w = float(load_w[idx] - pv_w[idx] - actual_power_w)
        import_w = max(net_grid_w, 0.0)
        export_w = max(-net_grid_w, 0.0)
        import_cost = import_w * float(price[idx]) * dt_hours / 1000.0
        export_revenue = export_w * float(config.feed_in_tariff) * dt_hours / 1000.0
        energy_cost += import_cost - export_revenue
        import_violation_w = max(import_w - import_limit_w, 0.0) if np.isfinite(import_limit_w) else 0.0
        export_violation_w = max(export_w - export_limit_w, 0.0) if np.isfinite(export_limit_w) else 0.0
        grid_penalty += (import_violation_w + export_violation_w) * float(config.grid_limit_violation_penalty_per_kwh) * dt_hours / 1000.0
        throughput_kwh = abs(float(battery_info.get("effective_power", 0.0))) * dt_hours / 1000.0
        loss_kwh = max(float(battery_info.get("power_loss", 0.0)), 0.0) * dt_hours / 1000.0
        stress_kwh = throughput_kwh * max(float(battery_info.get("r_int_power_factor", 1.0)) - 1.0, 0.0)
        throughput_penalty += float(getattr(config, "battery_throughput_penalty_per_kwh", 0.0)) * throughput_kwh
        loss_penalty += float(getattr(config, "battery_loss_penalty_per_kwh", 0.0)) * loss_kwh
        stress_penalty += float(getattr(config, "battery_stress_penalty_per_kwh", 0.0)) * stress_kwh
        soc_history.append(float(battery.soc))
        import_history_mw.append(float(import_w / 1_000_000.0))
        line_loading_dummy.append(0.0)

    final_soc = float(soc_history[-1]) if soc_history else float(config.battery_params.soc_init)
    terminal_soc_deviation = abs(final_soc - float(terminal_soc_target))
    terminal_soc_excess = max(terminal_soc_deviation - terminal_soc_tolerance, 0.0)
    terminal_soc_penalty = terminal_soc_penalty_per_kwh * terminal_soc_excess * nominal_energy_kwh
    final_cumulative_cost = float(energy_cost + grid_penalty)
    final_cumulative_objective_cost = float(final_cumulative_cost + throughput_penalty + loss_penalty + stress_penalty + terminal_soc_penalty)
    soc_min = float(config.battery_params.soc_min)
    soc_max = float(config.battery_params.soc_max)
    return {
        "steps": int(horizon),
        "profile_start_timestamp": str(timestamps[0]) if len(timestamps) > 0 else "",
        "profile_end_timestamp": str(timestamps[-1]) if len(timestamps) > 0 else "",
        "total_reward": float(-final_cumulative_objective_cost),
        "final_cumulative_cost": float(final_cumulative_cost),
        "final_cumulative_objective_cost": float(final_cumulative_objective_cost),
        "final_soc": float(final_soc),
        "total_terminal_soc_penalty": float(terminal_soc_penalty),
        "min_voltage_worst": float("nan"),
        "max_line_loading_peak": float("nan"),
        "mean_grid_import_mw": float(np.mean(import_history_mw)) if import_history_mw else 0.0,
        "soc_lower_dwell_fraction": _dwell_fraction(np.asarray(soc_history, dtype=float) <= soc_min + 1e-6) if soc_history else 0.0,
        "soc_upper_dwell_fraction": _dwell_fraction(np.asarray(soc_history, dtype=float) >= soc_max - 1e-6) if soc_history else 0.0,
        "selection_days": 0,
    }


def build_selector_schedule(
    *,
    case_key: str,
    battery_model: str,
    regime: str,
    days: int,
    seed: int,
    reward_profile: str,
    selector_window_days: int,
    selector_stride_days: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    full_config = build_config(case_key, battery_model, days, seed, regime, reward_profile)
    steps_day = steps_per_day(full_config.dt_seconds)
    current_soc = float(full_config.battery_params.soc_init)
    current_temperature_c = float(full_config.battery_params.temperature_init_c)
    actions_all: list[float] = []
    selection_rows: list[dict[str, float | int | str]] = []
    total_days = int(days)
    current_day = 0
    while current_day < total_days:
        remaining_days = total_days - current_day
        plan_days = min(max(int(selector_window_days), 1), remaining_days)
        execute_days = min(max(int(selector_stride_days), 1), remaining_days)
        execute_steps = int(execute_days * steps_day)
        episode_start_hour = int(current_day * 24)
        start_soc = float(current_soc)
        candidate_actions = _heuristic_seed_schedules(
            case_key=case_key,
            battery_model=battery_model,
            regime=regime,
            days=int(plan_days),
            seed=seed,
            reward_profile=reward_profile,
            episode_start_hour=episode_start_hour,
            initial_soc=current_soc,
        )
        candidate_names = _heuristic_names_for_count(len(candidate_actions))
        best_idx = 0
        best_score = float("inf")
        best_end_soc = current_soc
        best_end_temp = current_temperature_c
        day_scores: dict[str, float] = {}
        for idx, candidate in enumerate(candidate_actions):
            day_config = build_config(
                case_key=case_key,
                battery_model=battery_model,
                days=int(plan_days),
                seed=seed,
                regime=regime,
                reward_profile=reward_profile,
                episode_start_hour=episode_start_hour,
            )
            score, end_soc, end_temp = _score_schedule_on_forecast(
                config=day_config,
                actions=np.asarray(candidate, dtype=float),
                episode_start_hour=episode_start_hour,
                initial_soc=current_soc,
                initial_temperature_c=current_temperature_c,
                include_terminal_penalty=bool(execute_days >= remaining_days),
            )
            candidate_name = candidate_names[idx] if idx < len(candidate_names) else f"candidate_{idx}"
            day_scores[candidate_name] = float(score)
            if score < best_score:
                best_score = float(score)
                best_idx = int(idx)
                best_end_soc = float(end_soc)
                best_end_temp = float(end_temp)
        selected_name = candidate_names[best_idx] if best_idx < len(candidate_names) else f"candidate_{best_idx}"
        selected_actions = np.asarray(candidate_actions[best_idx], dtype=np.float32)
        executed_actions = np.asarray(selected_actions[:execute_steps], dtype=np.float32)
        actions_all.extend(float(value) for value in executed_actions)
        _, current_soc, current_temperature_c = _score_schedule_on_forecast(
            config=build_config(
                case_key=case_key,
                battery_model=battery_model,
                days=int(execute_days),
                seed=seed,
                regime=regime,
                reward_profile=reward_profile,
                episode_start_hour=episode_start_hour,
            ),
            actions=np.asarray(executed_actions, dtype=float),
            episode_start_hour=episode_start_hour,
            initial_soc=current_soc,
            initial_temperature_c=current_temperature_c,
            include_terminal_penalty=False,
        )
        selection_rows.append(
            {
                "day_index": int(current_day),
                "episode_start_hour": int(episode_start_hour),
                "plan_days": int(plan_days),
                "execute_days": int(execute_days),
                "selected_candidate": str(selected_name),
                "selected_score": float(best_score),
                "start_soc": float(start_soc),
                "end_soc": float(current_soc),
                **{f"score_{name}": float(score) for name, score in day_scores.items()},
            }
        )
        current_day += int(execute_days)
    return np.asarray(actions_all, dtype=np.float32), pd.DataFrame(selection_rows)


def build_named_schedule(
    *,
    case_key: str,
    battery_model: str,
    regime: str,
    days: int,
    seed: int,
    reward_profile: str,
    baseline_name: str,
    selector_window_days: int,
    selector_stride_days: int,
) -> tuple[np.ndarray, pd.DataFrame | None]:
    if baseline_name == "none":
        horizon = simulation_steps(days, build_config(case_key, battery_model, days, seed, regime, reward_profile).dt_seconds)
        return np.zeros(horizon, dtype=np.float32), None
    if baseline_name == "heuristic_selector_lite":
        return build_selector_schedule(
            case_key=case_key,
            battery_model=battery_model,
            regime=regime,
            days=days,
            seed=seed,
            reward_profile=reward_profile,
            selector_window_days=int(selector_window_days),
            selector_stride_days=int(selector_stride_days),
        )
    schedules = _heuristic_seed_schedules(
        case_key=case_key,
        battery_model=battery_model,
        regime=regime,
        days=days,
        seed=seed,
        reward_profile=reward_profile,
        initial_soc=float(build_config(case_key, battery_model, days, seed, regime, reward_profile).battery_params.soc_init),
    )
    mapping = {
        "heuristic_zero": 0,
        "heuristic_peak_shave": 1,
        "heuristic_tou": 2,
        "heuristic_blended": 3,
    }
    if baseline_name not in mapping:
        raise ValueError(f"Unsupported baseline '{baseline_name}'")
    return np.asarray(schedules[mapping[baseline_name]], dtype=np.float32), None


def _dwell_fraction(mask: pd.Series | list[bool] | np.ndarray) -> float:
    series = pd.Series(mask, dtype=float)
    if series.empty:
        return 0.0
    return float(series.mean())


def _summarize_trajectory(
    *,
    trajectory: pd.DataFrame,
    case_key: str,
    regime: str,
    baseline: str,
    battery_model: str,
    days: int,
    seed: int,
    reward_profile: str,
    profile_start_timestamp: str,
    profile_end_timestamp: str,
    selection_df: pd.DataFrame | None = None,
) -> dict[str, float | int | str]:
    config = build_config(case_key, battery_model, days, seed, regime, reward_profile)
    soc_min = float(config.battery_params.soc_min)
    soc_max = float(config.battery_params.soc_max)
    final_cost = float(trajectory["cumulative_cost"].iloc[-1]) if not trajectory.empty else 0.0
    final_objective_cost = float(trajectory["cumulative_objective_cost"].iloc[-1]) if not trajectory.empty else final_cost
    final_soc = float(trajectory["soc"].iloc[-1]) if not trajectory.empty else float(config.battery_params.soc_init)
    lower_dwell = _dwell_fraction(trajectory["soc"].astype(float) <= soc_min + 1e-6) if not trajectory.empty else 0.0
    upper_dwell = _dwell_fraction(trajectory["soc"].astype(float) >= soc_max - 1e-6) if not trajectory.empty else 0.0
    return {
        "case": str(case_key),
        "regime": str(regime),
        "baseline": str(baseline),
        "battery_model": str(battery_model),
        "days": int(days),
        "seed": int(seed),
        "reward_profile": str(reward_profile),
        "profile_start_timestamp": str(profile_start_timestamp),
        "profile_end_timestamp": str(profile_end_timestamp),
        "steps": int(len(trajectory)),
        "total_reward": float(trajectory["reward"].sum()) if not trajectory.empty else 0.0,
        "final_cumulative_cost": final_cost,
        "final_cumulative_objective_cost": final_objective_cost,
        "final_soc": final_soc,
        "total_terminal_soc_penalty": float(trajectory["terminal_soc_penalty"].sum()) if not trajectory.empty else 0.0,
        "min_voltage_worst": float(trajectory["min_bus_voltage_pu"].min()) if not trajectory.empty else 1.0,
        "max_line_loading_peak": float(trajectory["max_line_loading_pct"].max()) if not trajectory.empty else 0.0,
        "mean_grid_import_mw": float(trajectory["grid_import_mw"].mean()) if not trajectory.empty else 0.0,
        "soc_lower_dwell_fraction": float(lower_dwell),
        "soc_upper_dwell_fraction": float(upper_dwell),
        "selection_days": int(len(selection_df)) if selection_df is not None else 0,
    }


def build_gap_summary(detail_df: pd.DataFrame, oracle_summary_csv: Path | None) -> pd.DataFrame:
    if detail_df.empty:
        return pd.DataFrame()
    oracle_df = pd.DataFrame()
    if oracle_summary_csv is not None and Path(oracle_summary_csv).exists():
        oracle_df = pd.read_csv(oracle_summary_csv)
    rows: list[dict[str, float | int | str]] = []
    case_map = {"ieee33": "ieee33_network", "cigre": "cigre_eu_lv_network"}
    for (case_key, regime), group in detail_df.groupby(["case", "regime"], dropna=False):
        by_baseline = {str(row["baseline"]): row for _, row in group.iterrows()}
        none_row = by_baseline.get("none")
        if none_row is None:
            continue
        oracle_row = None
        if not oracle_df.empty:
            oracle_match = oracle_df[
                (oracle_df["case_key"].astype(str).str.lower() == str(case_map.get(str(case_key).lower(), case_key)).lower())
                & (oracle_df["regime"].astype(str).str.lower() == str(regime).lower())
            ]
            if not oracle_match.empty:
                oracle_row = oracle_match.iloc[0]
        none_cost = float(none_row["final_cumulative_objective_cost"])
        oracle_cost = float(oracle_row["oracle_objective"]) if oracle_row is not None else np.nan
        oracle_savings = float(none_cost - oracle_cost) if oracle_row is not None else np.nan
        for _, row in group.iterrows():
            baseline = str(row["baseline"])
            if baseline == "none":
                continue
            algorithm_cost = float(row["final_cumulative_objective_cost"])
            savings_vs_none = float(none_cost - algorithm_cost)
            recovery_fraction = float(savings_vs_none / oracle_savings) if oracle_row is not None and abs(oracle_savings) > 1e-9 else np.nan
            gap_to_oracle = float(algorithm_cost - oracle_cost) if oracle_row is not None else np.nan
            rows.append(
                {
                    "protocol": f"full_year_{int(row['days'])}d_heuristic_lite",
                    "case": str(case_key),
                    "regime": str(regime),
                    "baseline": baseline,
                    "none_objective": none_cost,
                    "oracle_objective": oracle_cost,
                    "baseline_objective": algorithm_cost,
                    "savings_vs_none": savings_vs_none,
                    "oracle_savings_vs_none": oracle_savings,
                    "recovery_fraction": recovery_fraction,
                    "gap_to_oracle": gap_to_oracle,
                    "final_soc": float(row["final_soc"]),
                    "soc_lower_dwell_fraction": float(row["soc_lower_dwell_fraction"]),
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    args = build_parser().parse_args()
    case_keys = _parse_csv_arg(args.cases)
    regimes = _parse_csv_arg(args.regimes)
    baselines = _parse_csv_arg(args.baselines)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectories_dir = output_dir / "trajectories"
    if bool(args.save_trajectories):
        trajectories_dir.mkdir(parents=True, exist_ok=True)

    detail_rows: list[dict[str, float | int | str]] = []
    for case_key in case_keys:
        for regime in regimes:
            for baseline_name in baselines:
                actual_battery_model = "none" if baseline_name == "none" else str(args.battery_model)
                actions, selection_df = build_named_schedule(
                    case_key=case_key,
                    battery_model=actual_battery_model,
                    regime=regime,
                    days=int(args.days),
                    seed=int(args.seed),
                    reward_profile=str(args.reward_profile),
                    baseline_name=baseline_name,
                    selector_window_days=int(args.selector_window_days),
                    selector_stride_days=int(args.selector_stride_days),
                )
                if str(args.evaluation_mode).lower() == "env":
                    config = build_config(case_key, actual_battery_model, args.days, args.seed, regime, args.reward_profile)
                    env = build_env_from_config(config)
                    try:
                        _initialize_env_state(env, seed=int(args.seed))
                        profile_start_timestamp = str(pd.DatetimeIndex(env._profiles.timestamps)[0])
                        profile_end_timestamp = str(pd.DatetimeIndex(env._profiles.timestamps)[-1])
                        _, _, trajectory = _rollout_actions(
                            env,
                            actions=np.asarray(actions, dtype=float),
                            voltage_penalty=float(args.voltage_penalty),
                            loading_penalty=float(args.loading_penalty),
                            soc_penalty=float(args.soc_penalty),
                        )
                    finally:
                        env.close()
                    row = _summarize_trajectory(
                        trajectory=trajectory,
                        case_key=case_key,
                        regime=regime,
                        baseline=baseline_name,
                        battery_model=actual_battery_model,
                        days=int(args.days),
                        seed=int(args.seed),
                        reward_profile=str(args.reward_profile),
                        profile_start_timestamp=profile_start_timestamp,
                        profile_end_timestamp=profile_end_timestamp,
                        selection_df=selection_df,
                    )
                else:
                    surrogate = evaluate_actions_on_surrogate(
                        case_key=case_key,
                        battery_model=actual_battery_model,
                        regime=regime,
                        days=int(args.days),
                        seed=int(args.seed),
                        reward_profile=str(args.reward_profile),
                        actions=np.asarray(actions, dtype=float),
                    )
                    row = {
                        "case": str(case_key),
                        "regime": str(regime),
                        "baseline": str(baseline_name),
                        "battery_model": str(actual_battery_model),
                        "days": int(args.days),
                        "seed": int(args.seed),
                        "reward_profile": str(args.reward_profile),
                        "profile_start_timestamp": str(surrogate["profile_start_timestamp"]),
                        "profile_end_timestamp": str(surrogate["profile_end_timestamp"]),
                        "steps": int(surrogate["steps"]),
                        "total_reward": float(surrogate["total_reward"]),
                        "final_cumulative_cost": float(surrogate["final_cumulative_cost"]),
                        "final_cumulative_objective_cost": float(surrogate["final_cumulative_objective_cost"]),
                        "final_soc": float(surrogate["final_soc"]),
                        "total_terminal_soc_penalty": float(surrogate["total_terminal_soc_penalty"]),
                        "min_voltage_worst": surrogate["min_voltage_worst"],
                        "max_line_loading_peak": surrogate["max_line_loading_peak"],
                        "mean_grid_import_mw": float(surrogate["mean_grid_import_mw"]),
                        "soc_lower_dwell_fraction": float(surrogate["soc_lower_dwell_fraction"]),
                        "soc_upper_dwell_fraction": float(surrogate["soc_upper_dwell_fraction"]),
                        "selection_days": int(len(selection_df)) if selection_df is not None else 0,
                    }
                detail_rows.append(row)
                if bool(args.save_trajectories) and str(args.evaluation_mode).lower() == "env":
                    trajectory.to_csv(trajectories_dir / f"{case_key}_{regime}_{baseline_name}_trajectory.csv", index=False)
                if bool(args.save_trajectories) and selection_df is not None:
                    selection_df.to_csv(trajectories_dir / f"{case_key}_{regime}_{baseline_name}_selection.csv", index=False)
                print(
                    f"[heuristic-lite] case={case_key} regime={regime} baseline={baseline_name} "
                    f"objective={row['final_cumulative_objective_cost']:.3f} final_soc={row['final_soc']:.3f}"
                )

    detail_df = pd.DataFrame(detail_rows)
    gap_df = build_gap_summary(detail_df, args.oracle_summary_csv)
    detail_csv = output_dir / "detail.csv"
    gap_csv = output_dir / "gap_summary.csv"
    detail_json = output_dir / "detail.json"
    detail_df.to_csv(detail_csv, index=False)
    gap_df.to_csv(gap_csv, index=False)
    detail_json.write_text(json.dumps(detail_rows, indent=2), encoding="utf-8")

    print("\n=== Heuristic / GA-lite Detail ===")
    print(
        detail_df[
            [
                "case",
                "regime",
                "baseline",
                "battery_model",
                "final_cumulative_cost",
                "final_cumulative_objective_cost",
                "final_soc",
                "soc_lower_dwell_fraction",
                "soc_upper_dwell_fraction",
            ]
        ].round(6).to_string(index=False)
    )
    print("\n=== Gap Summary ===")
    if not gap_df.empty:
        print(gap_df.round(6).to_string(index=False))
    print(f"\nSaved detail CSV: {detail_csv}")
    print(f"Saved gap summary CSV: {gap_csv}")
    print(f"Saved detail JSON: {detail_json}")
    if bool(args.save_trajectories):
        print(f"Saved trajectories: {trajectories_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
