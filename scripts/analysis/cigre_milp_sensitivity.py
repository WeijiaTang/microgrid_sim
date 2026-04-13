from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

import microgrid_sim.baselines.dispatch as dispatch_module
from microgrid_sim.baselines.dispatch import _battery_command_to_action, run_milp_baseline
from microgrid_sim.cases import CIGREEuropeanLVConfig
from microgrid_sim.envs.network_microgrid import NetworkMicrogridEnv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a CIGRE MILP Oracle siting and sizing sensitivity scan."
    )
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--regime", type=str, default="network_stress")
    parser.add_argument("--reward-profile", type=str, default="paper_balanced")
    parser.add_argument("--efficiency-model", type=str, default="realistic")
    parser.add_argument(
        "--battery-buses",
        type=str,
        default="Bus R11,Bus R18",
        help="Comma-separated list of CIGRE bus names to test.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/cigre_milp_sensitivity"),
    )
    return parser


def _variant_params(base_params) -> dict[str, object]:
    return {
        "base_200kw_358kwh": base_params,
        "power_150kw_179kwh": replace(
            base_params,
            num_cells_parallel=1,
            nominal_energy_kwh=179.2,
            p_charge_max=150_000.0,
            p_discharge_max=150_000.0,
        ),
        "legacy_100kw_179kwh": replace(
            base_params,
            num_cells_parallel=1,
            nominal_energy_kwh=179.2,
            p_charge_max=100_000.0,
            p_discharge_max=100_000.0,
        ),
    }


def _replay_metrics(config: CIGREEuropeanLVConfig, realized_battery_power_w: list[float]) -> dict[str, float]:
    env = NetworkMicrogridEnv(config)
    try:
        env.reset()
        energy_cost = 0.0
        grid_penalty = 0.0
        min_bus_voltage = float("inf")
        max_line_loading = 0.0
        max_transformer_loading = 0.0
        max_line_current = 0.0
        max_import_violation = 0.0
        import_violation_hours = 0.0
        throughput_kwh = 0.0

        for power_w in realized_battery_power_w:
            action = _battery_command_to_action(env, float(power_w))
            _, _, terminated, truncated, info = env.step(action)
            energy_cost += float(info.get("net_energy_cost", 0.0))
            grid_penalty += float(info.get("grid_limit_penalty_cost", 0.0))
            min_bus_voltage = min(min_bus_voltage, float(info.get("min_bus_voltage_pu", 1.0)))
            max_line_loading = max(max_line_loading, float(info.get("max_line_loading_pct", 0.0)))
            max_transformer_loading = max(
                max_transformer_loading,
                float(info.get("max_transformer_loading_pct", 0.0)),
            )
            max_line_current = max(max_line_current, float(info.get("max_line_current_ka", 0.0)))
            import_violation_mw = float(info.get("grid_import_limit_violation_mw", 0.0))
            max_import_violation = max(max_import_violation, import_violation_mw)
            if import_violation_mw > 1e-9:
                import_violation_hours += float(config.dt_seconds) / 3600.0
            throughput_kwh += float(info.get("battery_throughput_kwh", 0.0))
            if terminated or truncated:
                break

        return {
            "energy_cost": float(energy_cost),
            "grid_penalty": float(grid_penalty),
            "min_bus_voltage_worst": float(min_bus_voltage),
            "max_line_loading_peak": float(max_line_loading),
            "max_transformer_loading_peak": float(max_transformer_loading),
            "max_line_current_peak_ka": float(max_line_current),
            "max_import_violation_mw": float(max_import_violation),
            "import_violation_hours": float(import_violation_hours),
            "throughput_kwh": float(throughput_kwh),
        }
    finally:
        env.close()


def evaluate_config(config: CIGREEuropeanLVConfig, efficiency_model: str) -> dict[str, float | str]:
    env = NetworkMicrogridEnv(config)
    try:
        result = run_milp_baseline(
            env,
            simulation_days=int(config.simulation_days),
            chunk_days=0,
            efficiency_model=efficiency_model,
        )
    finally:
        env.close()

    replay = _replay_metrics(config, realized_battery_power_w=list(result["battery_power"]))
    return {
        "case": "cigre",
        "regime": str(config.regime),
        "battery_model": str(config.battery_model),
        "battery_bus_name": str(config.battery_bus_name),
        "storage_power_kw": float(config.battery_params.p_discharge_max) / 1000.0,
        "storage_energy_kwh": float(config.battery_params.nominal_energy_wh) / 1000.0,
        "total_cost": float(result["total_cost"]),
        "total_objective_cost": float(result["total_objective_cost"]),
        "final_soc": float(result["soc"][-1]) if result["soc"] else float(config.battery_params.soc_init),
        **replay,
    }


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dispatch_module.tqdm = lambda iterable, desc=None: iterable

    battery_buses = [bus.strip() for bus in str(args.battery_buses).split(",") if bus.strip()]
    base_config = CIGREEuropeanLVConfig(
        simulation_days=int(args.days),
        battery_model="simple",
        reward_profile=str(args.reward_profile),
        regime=str(args.regime),
        seed=int(args.seed),
    )

    rows: list[dict[str, float | str]] = []
    none_row = evaluate_config(replace(base_config, battery_model="none"), efficiency_model=str(args.efficiency_model))
    none_row["variant"] = "no_battery"
    rows.append(none_row)

    for bus_name in battery_buses:
        for variant_name, params in _variant_params(base_config.battery_params).items():
            config = replace(base_config, battery_bus_name=bus_name, battery_params=params)
            row = evaluate_config(config, efficiency_model=str(args.efficiency_model))
            row["variant"] = variant_name
            rows.append(row)

    summary_df = pd.DataFrame(rows)
    none_total_cost = float(summary_df.loc[summary_df["variant"] == "no_battery", "total_cost"].iloc[0])
    none_grid_penalty = float(summary_df.loc[summary_df["variant"] == "no_battery", "grid_penalty"].iloc[0])
    none_min_voltage = float(summary_df.loc[summary_df["variant"] == "no_battery", "min_bus_voltage_worst"].iloc[0])
    summary_df["total_cost_delta_vs_none"] = summary_df["total_cost"] - none_total_cost
    summary_df["grid_penalty_delta_vs_none"] = summary_df["grid_penalty"] - none_grid_penalty
    summary_df["min_bus_voltage_delta_vs_none"] = summary_df["min_bus_voltage_worst"] - none_min_voltage

    summary_df = summary_df.sort_values(["variant", "total_cost", "battery_bus_name"], kind="stable")
    summary_path = output_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)

    ranked = summary_df.loc[summary_df["variant"] != "no_battery"].sort_values(
        ["total_cost", "min_bus_voltage_worst"],
        ascending=[True, False],
        kind="stable",
    )
    best_path = output_dir / "best_candidates.csv"
    ranked.head(5).to_csv(best_path, index=False)

    print(f"Saved sensitivity summary to: {summary_path}")
    print(f"Saved ranked candidates to: {best_path}")
    if not ranked.empty:
        print("Top candidates:")
        print(
            ranked[
                [
                    "variant",
                    "battery_bus_name",
                    "storage_power_kw",
                    "storage_energy_kwh",
                    "total_cost",
                    "total_cost_delta_vs_none",
                    "min_bus_voltage_worst",
                    "max_line_loading_peak",
                    "grid_penalty",
                ]
            ].head(5).to_string(index=False)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
