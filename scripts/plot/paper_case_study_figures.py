#!/usr/bin/env python3
"""Generate paper-ready case-study figures for MG-RES and MG-CIGRE.

This script rebuilds the main case-study plots used in ``en/texfile/4CaseStudy.tex``:

- ``MGRES_eta100_cumulative_cost``
- ``CIGRE_eta100_cumulative_cost``
- ``MGRES_E9E_efficiency_hist_kde``
- ``CIGRE_E9E_efficiency_hist_kde``
- ``MGRES_E10_3D_efficiency_scatter_discharge``
- ``MGRES_E10_3D_efficiency_scatter_charge``
- ``CIGRE_E10_3D_efficiency_scatter_discharge``
- ``CIGRE_E10_3D_efficiency_scatter_charge``

Notes
-----
- Residential figures load the packaged paper timeseries directly for exact consistency.
- CIGRE cumulative-cost uses the packaged paper timeseries for exact consistency.
- CIGRE mechanism figures use a compatibility evaluation for the historical 910k models:
  the current environment produces a 19-D observation, while the old models were trained
  on the 12-D observation interface. For representative mechanism visualization, the
  agent receives the first 12 observation components matching that older interface.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.stats import gaussian_kde

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import _cmd_run_cigre_gap_experiment as cigre_exp
from experiments import _cmd_run_residential_d4_experiment as res_exp
from microgrid_sim.envs import CIGREMicrogridEnv

PBM_COLOR = "#0b6e8a"
EBM_COLOR = "#c05a2b"
PBM_ALT = "#2a9d8f"
EBM_ALT = "#d1495b"
GRID_COLOR = "#4b5563"
ANNOTATION_BG = "#f6f4ef"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate case-study figures for the English paper.")
    parser.add_argument("--res-seed", type=int, default=48, help="Representative residential seed.")
    parser.add_argument("--cigre-seed", type=int, default=44, help="Representative CIGRE seed.")
    parser.add_argument(
        "--res-days",
        type=int,
        default=365,
        help="Residential plot horizon in days; defaults to full packaged yearly rollout.",
    )
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU when loading/evaluating models.")
    parser.add_argument(
        "--skip-eval-cache",
        action="store_true",
        help="Ignore cached plot-data CSVs and recompute representative evaluations.",
    )
    return parser.parse_args()


def set_paper_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#222222",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": "#d8d8d8",
            "grid.linewidth": 0.55,
            "grid.alpha": 1.0,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
            "font.size": 10.5,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10.5,
            "legend.fontsize": 9.4,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "legend.framealpha": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def ensure_output_dirs() -> dict[str, Path]:
    dirs = {
        "pdf": REPO_ROOT / "en" / "figures" / "pdf",
        "eps": REPO_ROOT / "en" / "figures" / "eps",
        "png": REPO_ROOT / "en" / "figures" / "png",
        "tiff": REPO_ROOT / "en" / "figures" / "tiff",
        "plot_data": REPO_ROOT / "results" / "paper" / "plot_data",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def save_figure(fig: plt.Figure, name: str, dirs: dict[str, Path]) -> None:
    fig.savefig(dirs["pdf"] / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(dirs["eps"] / f"{name}.eps", bbox_inches="tight")
    fig.savefig(dirs["png"] / f"{name}.png", bbox_inches="tight", dpi=450)
    fig.savefig(dirs["tiff"] / f"{name}.tiff", bbox_inches="tight", dpi=450)
    plt.close(fig)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sanitize_series(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return arr
    arr = arr[np.isfinite(arr)]
    return arr


def load_residential_cost_timeseries(seed: int, days: int | None = None) -> dict[str, pd.DataFrame]:
    base = REPO_ROOT / "results" / "paper" / "res" / f"seed{seed}"
    report = read_json(base / "report.json")
    steps = int(report["steps"])
    train_year = int(report["train_year"])
    eval_year = int(report["eval_year"])
    pbm_df = pd.read_csv(base / f"pbm_seed{seed}_{steps}_train{train_year}_eval{eval_year}_timeseries.csv")
    ebm_df = pd.read_csv(base / f"ebm_seed{seed}_{steps}_train{train_year}_eval{eval_year}_timeseries.csv")
    if days is not None and days > 0:
        horizon = int(days) * 24
        pbm_df = pbm_df.iloc[:horizon].copy()
        ebm_df = ebm_df.iloc[:horizon].copy()
    return {"pbm": pbm_df, "ebm": ebm_df}


def build_cigre_compat_plot_timeseries(seed: int, force_cpu: bool, cache_dir: Path, use_cache: bool) -> dict[str, pd.DataFrame]:
    pbm_cache = cache_dir / f"cigre_seed{seed}_compat_pbm_plot_timeseries.csv"
    ebm_cache = cache_dir / f"cigre_seed{seed}_compat_ebm_plot_timeseries.csv"
    meta_cache = cache_dir / f"cigre_seed{seed}_compat_meta.json"

    if use_cache and pbm_cache.exists() and ebm_cache.exists() and meta_cache.exists():
        return {
            "pbm": pd.read_csv(pbm_cache),
            "ebm": pd.read_csv(ebm_cache),
        }

    report_path = REPO_ROOT / "results" / "paper" / "cigre" / f"seed{seed}" / "report.json"
    report = read_json(report_path)
    steps = int(report["steps"])
    train_year = int(report["train_year"])
    eval_year = int(report["eval_year"])
    pbm_model_path = REPO_ROOT / "models" / "cigre_d4_multiseed_910k" / f"seed{seed}" / f"pbm_seed{seed}_{steps}_train{train_year}_eval{eval_year}.zip"
    ebm_model_path = REPO_ROOT / "models" / "cigre_d4_multiseed_910k" / f"seed{seed}" / f"ebm_seed{seed}_{steps}_train{train_year}_eval{eval_year}.zip"
    device = cigre_exp.get_device()
    pbm_agent = cigre_exp.load_agent("sac", str(pbm_model_path), device=device)
    ebm_agent = cigre_exp.load_agent("sac", str(ebm_model_path), device=device)

    config = cigre_exp.build_config(
        battery_model="thevenin",
        simulation_days=int(report["eval_days"]),
        seed=int(seed),
        data_dir=report.get("data_dir"),
        data_year=eval_year,
        random_episode_start=False,
        episode_start_hour=0,
        reward_mode=str(report.get("reward_mode", "cost")),
        component_commitment_enabled=bool(report.get("component_commitment_enabled", False)),
        include_component_cost_in_objective=bool(report.get("include_component_cost_in_objective", False)),
        random_initial_soc=False,
        initial_soc_min=0.0,
        initial_soc_max=1.0,
        price_spread_multiplier=float(report.get("price_spread_multiplier", 8.0)),
        peak_import_penalty_per_kw=float(report.get("peak_import_penalty_per_kw", 1.5)),
        peak_import_threshold_kw=float(report.get("peak_import_threshold_kw", 10.0)),
        midday_pv_boost_multiplier=float(report.get("midday_pv_boost_multiplier", 1.25)),
        evening_load_boost_multiplier=float(report.get("evening_load_boost_multiplier", 1.35)),
        stress_episode_sampling=False,
        stress_sampling_strength=0.0,
        battery_power_scale=float(report.get("battery_power_scale", 1.0)),
        battery_energy_scale=float(report.get("battery_energy_scale", 1.0)),
        optimistic_ebm_training=False,
        optimistic_ebm_soc_min=float(report.get("optimistic_ebm_soc_min", 0.0)),
        optimistic_ebm_soc_max=float(report.get("optimistic_ebm_soc_max", 1.0)),
        optimistic_ebm_power_scale=float(report.get("optimistic_ebm_power_scale", 1.0)),
        optimistic_ebm_efficiency=float(report.get("optimistic_ebm_efficiency", 1.0)),
        optimistic_ebm_soc_penalty_scale=float(report.get("optimistic_ebm_soc_penalty_scale", 1.0)),
    )

    def run(agent, label: str) -> pd.DataFrame:
        env = CIGREMicrogridEnv(config=config, battery_model="thevenin")
        obs, _ = env.reset()
        rows: list[dict] = []
        for step in range(int(report["eval_days"]) * 24):
            compat_obs = np.asarray(obs[:12], dtype=np.float32)
            action, _ = agent.predict(compat_obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            battery_info = dict(info.get("battery_info", {}) or {})
            p_cmd_kw = float(info.get("p_cmd", 0.0)) / 1000.0
            p_actual_kw = float(info.get("p_actual", 0.0)) / 1000.0
            p_grid_kw = float(info.get("p_grid", 0.0)) / 1000.0
            power_loss_kw = float(battery_info.get("power_loss", 0.0)) / 1000.0
            rows.append(
                {
                    "Hour": step,
                    "SOC": float(info.get("soc", 0.0)),
                    "SOH": float(info.get("soh", 1.0)),
                    "Cumulative_Cost": float(info.get("cumulative_cost", 0.0)),
                    "Battery_Command_kW": p_cmd_kw,
                    "Battery_Actual_kW": p_actual_kw,
                    "Grid_Power_kW": p_grid_kw,
                    "Grid_Import_kW": max(p_grid_kw, 0.0),
                    "Import_Cost": float(info.get("import_cost", 0.0)),
                    "Peak_Import_Penalty": float(info.get("peak_import_penalty", 0.0)),
                    "Price_per_kWh": float(info.get("price", 0.0)),
                    "Battery_Efficiency": float(battery_info.get("efficiency", 1.0)),
                    "Battery_Current_A": float(battery_info.get("current", 0.0)),
                    "Battery_Power_Loss_kW": power_loss_kw,
                    "Battery_Temperature_C": float(battery_info.get("temperature_c", info.get("battery_temperature_c", 0.0))),
                    "Battery_Rint_Ohm": float(battery_info.get("r_int", 0.0)),
                    "Battery_OCV_V": float(battery_info.get("v_ocv", 0.0)),
                    "Battery_OCV_Hysteresis_V": float(battery_info.get("ocv_hysteresis_offset", 0.0)),
                    "Battery_Polarization_V": float(battery_info.get("polarization_voltage", 0.0)),
                    "Label": label,
                }
            )
            if terminated or truncated:
                break
        env.close()
        return pd.DataFrame(rows)

    pbm_df = run(pbm_agent, "PBM")
    ebm_df = run(ebm_agent, "EBM")
    pbm_df.to_csv(pbm_cache, index=False)
    ebm_df.to_csv(ebm_cache, index=False)
    meta_cache.write_text(
        json.dumps(
            {
                "seed": seed,
                "note": (
                    "CIGRE mechanism plots use compatibility evaluation: historical 910k models "
                    "receive the first 12 components of the current 19-D observation."
                ),
                "pbm_model_path": str(pbm_model_path),
                "ebm_model_path": str(ebm_model_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"pbm": pbm_df, "ebm": ebm_df}


def load_cigre_cost_timeseries(seed: int) -> dict[str, pd.DataFrame]:
    base = REPO_ROOT / "results" / "paper" / "cigre" / f"seed{seed}"
    pbm_df = pd.read_csv(base / f"pbm_seed{seed}_910000_train2023_eval2024_timeseries.csv")
    ebm_df = pd.read_csv(base / f"ebm_seed{seed}_910000_train2023_eval2024_timeseries.csv")
    return {"pbm": pbm_df, "ebm": ebm_df}


def add_terminal_gap_box(ax: plt.Axes, pbm_cost: float, ebm_cost: float, unit: str = "$") -> None:
    gap_pct = 100.0 * (ebm_cost - pbm_cost) / max(abs(pbm_cost), 1e-9)
    text = (
        f"Final PBM: {unit}{pbm_cost:,.1f}\n"
        f"Final EBM: {unit}{ebm_cost:,.1f}\n"
        f"MFM gap: {gap_pct:+.2f}%"
    )
    ax.text(
        0.985,
        0.05,
        text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.2,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": ANNOTATION_BG, "edgecolor": "#999999"},
    )


def plot_cumulative_cost(
    res_pbm: pd.DataFrame,
    res_ebm: pd.DataFrame,
    title: str | None,
    name: str,
    dirs: dict[str, Path],
) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.plot(res_pbm["Hour"], res_pbm["Cumulative_Cost"], color=PBM_COLOR, lw=2.3, label="SAC-PBM")
    ax.plot(res_ebm["Hour"], res_ebm["Cumulative_Cost"], color=EBM_COLOR, lw=2.1, ls="--", label="SAC-EBM")
    gap = res_ebm["Cumulative_Cost"].to_numpy(dtype=float) - res_pbm["Cumulative_Cost"].to_numpy(dtype=float)
    ax.plot(res_pbm["Hour"], gap, color=GRID_COLOR, lw=1.1, ls=":", label="EBM minus PBM")
    if title:
        ax.set_title(title)
    ax.set_xlabel("Hour")
    ax.set_ylabel("Cumulative Electricity Cost ($)")
    ax.legend(loc="upper left", frameon=True, framealpha=1.0)
    add_terminal_gap_box(
        ax,
        float(res_pbm["Cumulative_Cost"].iloc[-1]),
        float(res_ebm["Cumulative_Cost"].iloc[-1]),
    )
    ax.margins(x=0.01)
    save_figure(fig, name, dirs)


def _mode_masks(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    power_col = "Battery_Actual_kW" if "Battery_Actual_kW" in df.columns else "Battery_Power_kW"
    power = df[power_col].to_numpy(dtype=float)
    discharge = power > 1e-6
    charge = power < -1e-6
    return discharge, charge


def _kde_curve(values: np.ndarray, grid: np.ndarray) -> np.ndarray | None:
    values = values[np.isfinite(values)]
    if values.size < 10:
        return None
    if float(np.std(values)) < 1e-6:
        return None
    return gaussian_kde(values)(grid)


def plot_efficiency_histogram(
    pbm_df: pd.DataFrame,
    ebm_df: pd.DataFrame,
    title: str | None,
    name: str,
    dirs: dict[str, Path],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.3, 4.2), sharey=True)
    modes = [
        ("Discharging", _mode_masks(pbm_df)[0], _mode_masks(ebm_df)[0]),
        ("Charging", _mode_masks(pbm_df)[1], _mode_masks(ebm_df)[1]),
    ]
    for ax, (mode_name, pbm_mask, ebm_mask) in zip(axes, modes):
        pbm_eff = sanitize_series(100.0 * pbm_df.loc[pbm_mask, "Battery_Efficiency"])
        ebm_eff = sanitize_series(100.0 * ebm_df.loc[ebm_mask, "Battery_Efficiency"])
        lo = min(float(np.min(pbm_eff)) if pbm_eff.size else 80.0, float(np.min(ebm_eff)) if ebm_eff.size else 80.0)
        hi = max(float(np.max(pbm_eff)) if pbm_eff.size else 100.0, float(np.max(ebm_eff)) if ebm_eff.size else 100.0)
        lo = math.floor(lo * 2.0) / 2.0 - 0.5
        hi = math.ceil(hi * 2.0) / 2.0 + 0.5
        bins = np.linspace(lo, hi, 28)
        grid = np.linspace(lo, hi, 240)

        ax.hist(pbm_eff, bins=bins, density=True, histtype="step", color=PBM_COLOR, lw=1.8, label="SAC-PBM")
        ax.hist(ebm_eff, bins=bins, density=True, histtype="step", color=EBM_COLOR, lw=1.8, ls="--", label="SAC-EBM")
        pbm_kde = _kde_curve(pbm_eff, grid)
        ebm_kde = _kde_curve(ebm_eff, grid)
        if pbm_kde is not None:
            ax.plot(grid, pbm_kde, color=PBM_ALT, lw=1.4)
        if ebm_kde is not None:
            ax.plot(grid, ebm_kde, color=EBM_ALT, lw=1.4, ls="-.")

        med_text = (
            f"PBM median: {np.median(pbm_eff):.2f}%\n"
            f"EBM median: {np.median(ebm_eff):.2f}%"
            if pbm_eff.size and ebm_eff.size
            else "Sparse samples"
        )
        ax.text(
            0.97,
            0.93,
            med_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.8,
            bbox={"boxstyle": "round,pad=0.28", "facecolor": "#fbfaf7", "edgecolor": "#b7b7b7"},
        )
        ax.set_title(mode_name)
        ax.set_xlabel("Instantaneous Efficiency (%)")
        ax.set_xlim(lo, hi)
        decimals = 1 if (hi - lo) < 5.0 else 0
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=100.0, decimals=decimals))

    axes[0].set_ylabel("Density")
    axes[0].legend(loc="upper left", frameon=True, framealpha=1.0)
    if title:
        fig.suptitle(title, y=1.02, fontsize=12)
    fig.tight_layout(w_pad=1.8)
    save_figure(fig, name, dirs)


def sample_indices(length: int, max_points: int) -> np.ndarray:
    if length <= max_points:
        return np.arange(length, dtype=int)
    return np.linspace(0, length - 1, max_points, dtype=int)


def plot_3d_efficiency_scatter(
    pbm_df: pd.DataFrame,
    ebm_df: pd.DataFrame,
    mode: str,
    title: str | None,
    name: str,
    dirs: dict[str, Path],
) -> None:
    fig = plt.figure(figsize=(7.8, 5.8))
    ax = fig.add_subplot(111, projection="3d")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
        axis.pane.set_edgecolor((0.85, 0.85, 0.85, 1.0))
        axis._axinfo["grid"]["color"] = (0.85, 0.85, 0.85, 1.0)
    pbm_power_col = "Battery_Actual_kW" if "Battery_Actual_kW" in pbm_df.columns else "Battery_Power_kW"
    ebm_power_col = "Battery_Actual_kW" if "Battery_Actual_kW" in ebm_df.columns else "Battery_Power_kW"
    if mode == "discharge":
        pbm_mask = pbm_df[pbm_power_col].to_numpy(dtype=float) > 1e-6
        ebm_mask = ebm_df[ebm_power_col].to_numpy(dtype=float) > 1e-6
    else:
        pbm_mask = pbm_df[pbm_power_col].to_numpy(dtype=float) < -1e-6
        ebm_mask = ebm_df[ebm_power_col].to_numpy(dtype=float) < -1e-6

    def xyz(df: pd.DataFrame, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sub = df.loc[mask]
        power_col = "Battery_Actual_kW" if "Battery_Actual_kW" in sub.columns else "Battery_Power_kW"
        return (
            sub["SOC"].to_numpy(dtype=float) * 100.0,
            np.abs(sub[power_col].to_numpy(dtype=float)),
            sub["Battery_Efficiency"].to_numpy(dtype=float) * 100.0,
        )

    pbm_x, pbm_y, pbm_z = xyz(pbm_df, pbm_mask)
    ebm_x, ebm_y, ebm_z = xyz(ebm_df, ebm_mask)
    pbm_idx = sample_indices(len(pbm_x), 1600 if len(pbm_x) > 4000 else 1000)
    ebm_idx = sample_indices(len(ebm_x), 1600 if len(ebm_x) > 4000 else 1000)

    if pbm_x.size or ebm_x.size:
        y_max = max(float(np.max(pbm_y)) if pbm_y.size else 0.0, float(np.max(ebm_y)) if ebm_y.size else 0.0)
        z_min = min(float(np.min(pbm_z)) if pbm_z.size else 90.0, float(np.min(ebm_z)) if ebm_z.size else 90.0)
        z_max = max(float(np.max(pbm_z)) if pbm_z.size else 100.0, float(np.max(ebm_z)) if ebm_z.size else 100.0)
        low_soc_color = "#c7b98a"
        ax.plot([20.0, 20.0], [0.0, y_max], [z_min, z_min], color=low_soc_color, lw=2.0, ls="--")
        ax.plot([20.0, 20.0], [0.0, y_max], [z_max, z_max], color=low_soc_color, lw=1.2, ls=":")
        ax.plot([20.0, 20.0], [0.0, 0.0], [z_min, z_max], color=low_soc_color, lw=1.2, ls=":")
        ax.plot([20.0, 20.0], [y_max, y_max], [z_min, z_max], color=low_soc_color, lw=1.2, ls=":")

    ax.scatter(
        pbm_x[pbm_idx],
        pbm_y[pbm_idx],
        pbm_z[pbm_idx],
        s=15,
        c=PBM_COLOR,
        marker="o",
        linewidths=0.0,
        depthshade=False,
        rasterized=True,
    )
    ax.scatter(
        ebm_x[ebm_idx],
        ebm_y[ebm_idx],
        ebm_z[ebm_idx],
        s=17,
        c=EBM_COLOR,
        marker="^",
        linewidths=0.0,
        depthshade=False,
        rasterized=True,
    )

    ax.set_xlabel("SOC (%)", labelpad=8)
    ax.set_ylabel(r"$|P_{\mathrm{bess}}|$ (kW)", labelpad=8)
    ax.set_zlabel("Efficiency (%)", labelpad=8)
    if title:
        ax.set_title(title, pad=10)
    ax.view_init(elev=24, azim=-58)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100.0, decimals=0))
    ax.zaxis.set_major_formatter(PercentFormatter(xmax=100.0, decimals=0))

    handles = [
        mlines.Line2D([], [], color=PBM_COLOR, marker="o", linestyle="None", markersize=7, label="SAC-PBM"),
        mlines.Line2D([], [], color=EBM_COLOR, marker="^", linestyle="None", markersize=7, label="SAC-EBM"),
        mlines.Line2D([], [], color="#c7b98a", lw=2.0, ls="--", label="Low-SOC boundary (20%)"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=True, framealpha=1.0)

    note = (
        f"PBM low-SOC share: {100.0 * np.mean(pbm_x < 20.0):.1f}%\n"
        f"EBM low-SOC share: {100.0 * np.mean(ebm_x < 20.0):.1f}%"
        if pbm_x.size and ebm_x.size
        else "No valid points"
    )
    ax.text2D(
        0.985,
        0.03,
        note,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.8,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#fbfaf7", "edgecolor": "#b7b7b7"},
    )

    save_figure(fig, name, dirs)


def main() -> None:
    args = parse_args()
    set_paper_style()
    dirs = ensure_output_dirs()

    use_cache = not bool(args.skip_eval_cache)

    res_data = load_residential_cost_timeseries(seed=int(args.res_seed), days=int(args.res_days))
    cigre_compat = build_cigre_compat_plot_timeseries(
        seed=int(args.cigre_seed),
        force_cpu=bool(args.force_cpu),
        cache_dir=dirs["plot_data"],
        use_cache=use_cache,
    )
    cigre_cost = load_cigre_cost_timeseries(seed=int(args.cigre_seed))

    plot_cumulative_cost(
        res_data["pbm"],
        res_data["ebm"],
        title=None,
        name="MGRES_eta100_cumulative_cost",
        dirs=dirs,
    )
    plot_cumulative_cost(
        cigre_cost["pbm"],
        cigre_cost["ebm"],
        title=None,
        name="CIGRE_eta100_cumulative_cost",
        dirs=dirs,
    )

    plot_efficiency_histogram(
        res_data["pbm"],
        res_data["ebm"],
        title=None,
        name="MGRES_E9E_efficiency_hist_kde",
        dirs=dirs,
    )
    plot_efficiency_histogram(
        cigre_compat["pbm"],
        cigre_compat["ebm"],
        title=None,
        name="CIGRE_E9E_efficiency_hist_kde",
        dirs=dirs,
    )

    plot_3d_efficiency_scatter(
        res_data["pbm"],
        res_data["ebm"],
        mode="discharge",
        title=None,
        name="MGRES_E10_3D_efficiency_scatter_discharge",
        dirs=dirs,
    )
    plot_3d_efficiency_scatter(
        res_data["pbm"],
        res_data["ebm"],
        mode="charge",
        title=None,
        name="MGRES_E10_3D_efficiency_scatter_charge",
        dirs=dirs,
    )
    plot_3d_efficiency_scatter(
        cigre_compat["pbm"],
        cigre_compat["ebm"],
        mode="discharge",
        title=None,
        name="CIGRE_E10_3D_efficiency_scatter_discharge",
        dirs=dirs,
    )
    plot_3d_efficiency_scatter(
        cigre_compat["pbm"],
        cigre_compat["ebm"],
        mode="charge",
        title=None,
        name="CIGRE_E10_3D_efficiency_scatter_charge",
        dirs=dirs,
    )

    print("Saved case-study figures to:")
    for key in ("png", "pdf", "eps", "tiff"):
        print(f"  {key}: {dirs[key]}")
    print(f"Cached plot data: {dirs['plot_data']}")


if __name__ == "__main__":
    main()
