#!/usr/bin/env python3
"""Generate paper-ready case-study figures for MG-RES and MG-CIGRE.

This script rebuilds the main case-study plots used in ``microgrid-paper/sections/04-case-study.tex``:

- ``MGRES_eta100_cumulative_cost``
- ``CIGRE_eta100_cumulative_cost``
- ``CASE_seed_cost_dumbbell_pair``
- ``CASE_monthly_pbm_advantage_pair``
- ``MGRES_dispatch_zoom_window``

Notes
-----
- All figures are drawn directly from the released paper packages in ``results/paper``.
- The main text now emphasizes evidence that is closest to the released annual summaries:
  seedwise paired costs, monthly cost gains, cumulative gap growth, and one residential
  dispatch zoom chosen from the representative seed.
- Legacy figure basenames are retained only for the two annual cumulative-cost panels to
  avoid LaTeX path churn in the existing manuscript.
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, PercentFormatter
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
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTH_DAY_COUNTS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate case-study figures for the English paper.")
    parser.add_argument("--res-seed", type=int, default=48, help="Representative residential seed.")
    parser.add_argument("--cigre-seed", type=int, default=44, help="Representative CIGRE seed.")
    parser.add_argument("--dispatch-window-days", type=int, default=7, help="Length of the residential dispatch zoom window.")
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
        "pdf": REPO_ROOT / "microgrid-paper" / "figures" / "pdf",
        "eps": REPO_ROOT / "microgrid-paper" / "figures" / "eps",
        "png": REPO_ROOT / "microgrid-paper" / "figures" / "png",
        "tiff": REPO_ROOT / "microgrid-paper" / "figures" / "tiff",
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


def load_case_cost_timeseries(case_name: str, seed: int) -> dict[str, pd.DataFrame | dict]:
    base = REPO_ROOT / "results" / "paper" / case_name / f"seed{seed}"
    report = read_json(base / "report.json")
    steps = int(report["steps"])
    train_year = int(report["train_year"])
    eval_year = int(report["eval_year"])
    pbm_df = pd.read_csv(base / f"pbm_seed{seed}_{steps}_train{train_year}_eval{eval_year}_timeseries.csv")
    ebm_df = pd.read_csv(base / f"ebm_seed{seed}_{steps}_train{train_year}_eval{eval_year}_timeseries.csv")
    return {"report": report, "pbm": pbm_df, "ebm": ebm_df}


def load_case_summary(case_name: str) -> pd.DataFrame:
    return pd.read_csv(REPO_ROOT / "results" / "paper" / case_name / "summary_seeds.csv")


def load_residential_cost_timeseries(seed: int, days: int | None = None) -> dict[str, pd.DataFrame]:
    case_data = load_case_cost_timeseries("res", seed)
    pbm_df = case_data["pbm"]
    ebm_df = case_data["ebm"]
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
                    "CIGRE mechanism plots use compatibility evaluation only: historical 910k models "
                    "receive the first 12 components of the current 19-D observation, so the plots "
                    "are qualitative rather than primary quantitative evidence."
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
    case_data = load_case_cost_timeseries("cigre", seed)
    return {"pbm": case_data["pbm"], "ebm": case_data["ebm"]}


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


def _day_axis(hours: pd.Series | np.ndarray) -> np.ndarray:
    return np.asarray(hours, dtype=float) / 24.0


def _month_edges(total_hours: int) -> list[int]:
    if int(total_hours) == sum(MONTH_DAY_COUNTS) * 24:
        edges = [0]
        for days in MONTH_DAY_COUNTS:
            edges.append(edges[-1] + int(days) * 24)
        return edges
    return [int(x) for x in np.linspace(0, int(total_hours), 13)]


def _add_month_guides(ax: plt.Axes, total_hours: int) -> None:
    for edge in _month_edges(total_hours)[1:-1]:
        ax.axvline(edge / 24.0, color="#d8d3c8", lw=0.8, ls=":", zorder=0)


def _currency_formatter(decimals: int = 0) -> FuncFormatter:
    return FuncFormatter(lambda x, _pos: f"{x:,.{decimals}f}")


def plot_cumulative_cost(
    res_pbm: pd.DataFrame,
    res_ebm: pd.DataFrame,
    title: str | None,
    name: str,
    dirs: dict[str, Path],
    highlight_day_range: tuple[int, int] | None = None,
    highlight_label: str | None = None,
) -> None:
    fig, (ax_cost, ax_gap) = plt.subplots(
        2,
        1,
        figsize=(6.4, 5.0),
        sharex=True,
        gridspec_kw={"height_ratios": [2.3, 1.0], "hspace": 0.08},
    )
    days = _day_axis(res_pbm["Hour"])
    pbm_cost = res_pbm["Cumulative_Cost"].to_numpy(dtype=float)
    ebm_cost = res_ebm["Cumulative_Cost"].to_numpy(dtype=float)
    gap = ebm_cost - pbm_cost

    ax_cost.plot(days, pbm_cost, color=PBM_COLOR, lw=2.2, label="SAC-PBM")
    ax_cost.plot(days, ebm_cost, color=EBM_COLOR, lw=2.0, ls="--", label="SAC-EBM")
    ax_cost.set_ylabel("Cumulative Cost ($)")
    ax_cost.yaxis.set_major_formatter(_currency_formatter(0))
    ax_cost.legend(loc="upper left", frameon=True, framealpha=1.0)
    add_terminal_gap_box(ax_cost, float(pbm_cost[-1]), float(ebm_cost[-1]))

    ax_gap.axhline(0.0, color="#9a9a9a", lw=0.9)
    ax_gap.plot(days, gap, color=GRID_COLOR, lw=1.9)
    ax_gap.fill_between(days, 0.0, gap, color="#cfd6dc", alpha=0.85)
    ax_gap.set_ylabel("Cum. Gap ($)")
    ax_gap.yaxis.set_major_formatter(_currency_formatter(0))
    ax_gap.set_xlabel("Day of Year")
    ax_gap.text(
        0.985,
        0.12,
        f"Final gap: ${gap[-1]:,.1f}",
        transform=ax_gap.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.8,
        bbox={"boxstyle": "round,pad=0.26", "facecolor": "#fbfaf7", "edgecolor": "#b7b7b7"},
    )

    for axis in (ax_cost, ax_gap):
        _add_month_guides(axis, len(res_pbm))
        axis.margins(x=0.01)
        if highlight_day_range is not None:
            start_day, end_day = highlight_day_range
            axis.axvspan(
                float(start_day),
                float(end_day),
                facecolor="#eadfc2",
                edgecolor="none",
                alpha=0.38,
                zorder=0.2,
            )
    if title:
        ax_cost.set_title(title)
    if highlight_day_range is not None:
        start_day, end_day = highlight_day_range
        label = highlight_label or "Selected dispatch zoom"
        ax_cost.text(
            0.02,
            0.06,
            f"{label}: days {start_day}-{end_day}",
            transform=ax_cost.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.26", "facecolor": "#fbfaf7", "edgecolor": "#b7b7b7"},
        )
    save_figure(fig, name, dirs)


def plot_seed_cost_dumbbell_pair(
    res_summary: pd.DataFrame,
    cigre_summary: pd.DataFrame,
    name: str,
    dirs: dict[str, Path],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    legend_handles = None
    for ax, summary, title in zip(
        axes,
        (res_summary.sort_values("seed"), cigre_summary.sort_values("seed")),
        ("MG-RES", "MG-CIGRE"),
    ):
        seeds = summary["seed"].astype(int).tolist()
        y = np.arange(len(seeds), dtype=float)
        pbm_vals = summary["pbm_cost"].to_numpy(dtype=float)
        ebm_vals = summary["ebm_cost"].to_numpy(dtype=float)
        gap_vals = summary["gap_pct"].to_numpy(dtype=float)
        x_min = float(min(np.min(pbm_vals), np.min(ebm_vals)))
        x_max = float(max(np.max(pbm_vals), np.max(ebm_vals)))
        x_pad = max((x_max - x_min) * 0.12, 0.03 * x_max)
        for yi, pbm_val, ebm_val, gap_pct in zip(y, pbm_vals, ebm_vals, gap_vals):
            ax.plot([pbm_val, ebm_val], [yi, yi], color="#c4c4c4", lw=2.2, zorder=1)
            ax.scatter(pbm_val, yi, s=62, color=PBM_COLOR, edgecolor="white", linewidth=0.7, zorder=3)
            ax.scatter(ebm_val, yi, s=72, color=EBM_COLOR, marker="^", edgecolor="white", linewidth=0.7, zorder=3)
            ax.text(ebm_val + 0.018 * (x_max - x_min + x_pad), yi, f"{gap_pct:.2f}%", va="center", fontsize=8.5)
        ax.set_yticks(y, [str(seed) for seed in seeds])
        ax.set_xlabel("Annual Electricity Cost ($)")
        ax.xaxis.set_major_formatter(_currency_formatter(0))
        ax.set_title(title)
        ax.set_xlim(x_min - 0.02 * x_pad, x_max + x_pad)
        ax.grid(axis="x", alpha=0.85)
        stats_text = f"Mean gap: {summary['gap_pct'].mean():.2f}%\nStd: {summary['gap_pct'].std(ddof=1):.2f}%"
        ax.text(
            0.98,
            0.06,
            stats_text,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.28", "facecolor": "#fbfaf7", "edgecolor": "#b7b7b7"},
        )
    axes[0].set_ylabel("Seed")
    legend_handles = [
        Line2D([], [], marker="o", color=PBM_COLOR, linestyle="None", markersize=7, label="SAC-PBM"),
        Line2D([], [], marker="^", color=EBM_COLOR, linestyle="None", markersize=7, label="SAC-EBM"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=2, frameon=True, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.96), w_pad=2.0)
    save_figure(fig, name, dirs)


def compute_monthly_gain(pbm_df: pd.DataFrame, ebm_df: pd.DataFrame) -> pd.DataFrame:
    gap = ebm_df["Cumulative_Cost"].to_numpy(dtype=float) - pbm_df["Cumulative_Cost"].to_numpy(dtype=float)
    edges = _month_edges(len(gap))
    rows: list[dict[str, float | str]] = []
    prev_gap = 0.0
    for idx, month_name in enumerate(MONTH_NAMES):
        end_idx = min(edges[idx + 1] - 1, len(gap) - 1)
        current_gap = float(gap[end_idx])
        rows.append({"month": month_name, "gain": current_gap - prev_gap})
        prev_gap = current_gap
    return pd.DataFrame(rows)


def compute_monthly_gain_stats(case_name: str) -> pd.DataFrame:
    summary = load_case_summary(case_name)
    rows: list[pd.DataFrame] = []
    for seed in sorted(summary["seed"].astype(int).tolist()):
        pair = load_case_cost_timeseries(case_name, seed)
        monthly = compute_monthly_gain(pair["pbm"], pair["ebm"])
        monthly["seed"] = seed
        rows.append(monthly)
    merged = pd.concat(rows, ignore_index=True)
    merged["month"] = pd.Categorical(merged["month"], categories=MONTH_NAMES, ordered=True)
    stats = (
        merged.groupby("month", observed=True)["gain"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
        .rename(columns={"mean": "gain_mean", "std": "gain_std", "min": "gain_min", "max": "gain_max"})
    )
    return stats


def plot_monthly_gain_pair(
    res_stats: pd.DataFrame,
    cigre_stats: pd.DataFrame,
    name: str,
    dirs: dict[str, Path],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), sharex=True)
    for ax, stats, title in zip(axes, (res_stats, cigre_stats), ("MG-RES", "MG-CIGRE")):
        x = np.arange(len(stats), dtype=float)
        means = stats["gain_mean"].to_numpy(dtype=float)
        stds = np.nan_to_num(stats["gain_std"].to_numpy(dtype=float), nan=0.0)
        ax.bar(x, means, width=0.72, color=PBM_COLOR, alpha=0.88, edgecolor="#2b3a42", linewidth=0.4)
        ax.errorbar(x, means, yerr=stds, fmt="none", ecolor="#374151", elinewidth=1.0, capsize=2.5, zorder=3)
        ax.axhline(0.0, color="#9a9a9a", lw=0.9)
        ax.set_xticks(x, MONTH_NAMES)
        ax.set_title(title)
        ax.yaxis.set_major_formatter(_currency_formatter(0))
        note = (
            f"Mean over seeds > 0: {int(np.sum(means > 0))}/12 months\n"
            f"Mean annual gain: ${float(np.sum(means)):,.1f}"
        )
        ax.text(
            0.98,
            0.95,
            note,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.4,
            bbox={"boxstyle": "round,pad=0.28", "facecolor": "#fbfaf7", "edgecolor": "#b7b7b7"},
        )
    axes[0].set_ylabel("Monthly PBM Advantage ($)")
    fig.tight_layout(w_pad=2.0)
    save_figure(fig, name, dirs)


def select_high_gap_window(pbm_df: pd.DataFrame, ebm_df: pd.DataFrame, window_days: int) -> tuple[int, float]:
    gap = ebm_df["Cumulative_Cost"].to_numpy(dtype=float) - pbm_df["Cumulative_Cost"].to_numpy(dtype=float)
    daily_gap = pd.Series(gap).groupby(np.arange(len(gap)) // 24).last()
    daily_gain = daily_gap.diff().fillna(daily_gap)
    weights = np.ones(int(window_days), dtype=float)
    rolling = np.convolve(daily_gain.to_numpy(dtype=float), weights, mode="valid")
    start_day = int(np.argmax(rolling))
    return start_day, float(rolling[start_day])


def plot_residential_dispatch_zoom(
    pbm_df: pd.DataFrame,
    ebm_df: pd.DataFrame,
    start_day: int,
    window_days: int,
    name: str,
    dirs: dict[str, Path],
) -> None:
    start_hour = int(start_day) * 24
    end_hour = start_hour + int(window_days) * 24
    pbm = pbm_df.iloc[start_hour:end_hour].copy()
    ebm = ebm_df.iloc[start_hour:end_hour].copy()
    x_days = np.arange(len(pbm), dtype=float) / 24.0
    gap = ebm["Cumulative_Cost"].to_numpy(dtype=float) - pbm["Cumulative_Cost"].to_numpy(dtype=float)
    gap = gap - float(gap[0])

    fig, axes = plt.subplots(
        5,
        1,
        figsize=(8.3, 7.9),
        sharex=True,
        gridspec_kw={"height_ratios": [0.9, 1.1, 1.1, 1.1, 0.9], "hspace": 0.08},
    )
    day_marks = np.arange(0, int(window_days) + 1, 1, dtype=float)
    for ax in axes:
        for day_mark in day_marks:
            ax.axvline(day_mark, color="#d8d3c8", lw=0.8, ls=":", zorder=0)

    axes[0].step(x_days, pbm["Price_per_kWh"].to_numpy(dtype=float), where="post", color="#6b7280", lw=1.6)
    axes[0].set_ylabel("Price\n($/kWh)")

    axes[1].plot(x_days, pbm["SOC"].to_numpy(dtype=float) * 100.0, color=PBM_COLOR, lw=1.9, label="SAC-PBM")
    axes[1].plot(x_days, ebm["SOC"].to_numpy(dtype=float) * 100.0, color=EBM_COLOR, lw=1.7, ls="--", label="SAC-EBM")
    axes[1].axhline(20.0, color="#c7b98a", lw=1.2, ls="--")
    axes[1].set_ylabel("SOC (%)")
    axes[1].legend(loc="upper left", ncol=2, frameon=True, framealpha=1.0)

    axes[2].plot(x_days, pbm["Battery_Power_kW"].to_numpy(dtype=float), color=PBM_COLOR, lw=1.9)
    axes[2].plot(x_days, ebm["Battery_Power_kW"].to_numpy(dtype=float), color=EBM_COLOR, lw=1.7, ls="--")
    axes[2].axhline(0.0, color="#9a9a9a", lw=0.8)
    axes[2].set_ylabel("Battery\nPower (kW)")

    axes[3].plot(x_days, pbm["Grid_Import_kW"].to_numpy(dtype=float), color=PBM_COLOR, lw=1.9)
    axes[3].plot(x_days, ebm["Grid_Import_kW"].to_numpy(dtype=float), color=EBM_COLOR, lw=1.7, ls="--")
    axes[3].set_ylabel("Grid\nImport (kW)")

    axes[4].plot(x_days, gap, color=GRID_COLOR, lw=1.9)
    axes[4].fill_between(x_days, 0.0, gap, color="#cfd6dc", alpha=0.85)
    axes[4].axhline(0.0, color="#9a9a9a", lw=0.8)
    axes[4].set_ylabel("Gap\n($)")
    axes[4].set_xlabel("Day within Selected Window")
    axes[4].yaxis.set_major_formatter(_currency_formatter(1))

    for ax in axes[1:4]:
        ax.margins(x=0.01)
    window_gain = float(gap[-1])
    axes[4].text(
        0.985,
        0.08,
        f"Days {start_day}-{start_day + window_days - 1}\n"
        f"Hours {start_hour}-{end_hour - 1}\n"
        f"Selected by max {window_days}-day gap growth\n"
        f"Window gain: ${window_gain:,.2f}",
        transform=axes[4].transAxes,
        ha="right",
        va="bottom",
        fontsize=8.4,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "#fbfaf7", "edgecolor": "#b7b7b7"},
    )
    save_figure(fig, name, dirs)


def _mode_masks(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    power_col = "Battery_Actual_kW" if "Battery_Actual_kW" in df.columns else "Battery_Power_kW"
    power = df[power_col].to_numpy(dtype=float)
    discharge = power > 1e-6
    charge = power < -1e-6
    return discharge, charge


def _loss_column(df: pd.DataFrame) -> str:
    if "Battery_Power_Loss_kW" in df.columns:
        return "Battery_Power_Loss_kW"
    if "Battery_Loss_kW" in df.columns:
        return "Battery_Loss_kW"
    raise KeyError("Expected a battery loss column in plot input data")


def _kde_curve(values: np.ndarray, grid: np.ndarray) -> np.ndarray | None:
    values = values[np.isfinite(values)]
    if values.size < 10:
        return None
    if float(np.std(values)) < 1e-6:
        return None
    return gaussian_kde(values)(grid)


def _robust_upper_limit(*arrays: np.ndarray, quantile: float = 0.995, floor: float = 0.10) -> float:
    pooled = [np.asarray(arr, dtype=float).ravel() for arr in arrays if np.asarray(arr).size]
    if not pooled:
        return floor
    values = np.concatenate(pooled)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return floor
    raw = max(float(np.quantile(values, quantile)), floor)
    if raw <= 0.20:
        step = 0.025
    elif raw <= 0.50:
        step = 0.05
    else:
        step = 0.10
    return max(step * 4.0, math.ceil(raw / step) * step)


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
    pbm_loss_col = _loss_column(pbm_df)
    ebm_loss_col = _loss_column(ebm_df)
    for ax, (mode_name, pbm_mask, ebm_mask) in zip(axes, modes):
        pbm_eff = sanitize_series(pbm_df.loc[pbm_mask, pbm_loss_col])
        ebm_eff = sanitize_series(ebm_df.loc[ebm_mask, ebm_loss_col])
        lo = 0.0
        hi = _robust_upper_limit(pbm_eff, ebm_eff, quantile=0.995, floor=0.10)
        bins = np.linspace(lo, hi, 24)
        grid = np.linspace(lo, hi, 240)
        pbm_tail = 100.0 * float(np.mean(pbm_eff > hi)) if pbm_eff.size else 0.0
        ebm_tail = 100.0 * float(np.mean(ebm_eff > hi)) if ebm_eff.size else 0.0

        ax.hist(pbm_eff, bins=bins, density=True, histtype="step", color=PBM_COLOR, lw=1.8, label="SAC-PBM")
        ax.hist(ebm_eff, bins=bins, density=True, histtype="step", color=EBM_COLOR, lw=1.8, ls="--", label="SAC-EBM")
        pbm_kde = _kde_curve(pbm_eff, grid)
        ebm_kde = _kde_curve(ebm_eff, grid)
        if pbm_kde is not None:
            ax.plot(grid, pbm_kde, color=PBM_ALT, lw=1.4)
        if ebm_kde is not None:
            ax.plot(grid, ebm_kde, color=EBM_ALT, lw=1.4, ls="-.")

        med_text = (
            f"PBM median: {np.median(pbm_eff):.3f} kW\n"
            f"EBM median: {np.median(ebm_eff):.3f} kW\n"
            f"Shown <= {hi:.3f} kW\n"
            f"Tail: {pbm_tail:.1f}% / {ebm_tail:.1f}%"
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
        ax.set_xlabel("Battery Power Loss (kW)")
        ax.set_xlim(lo, hi)

    axes[0].set_ylabel("Density")
    axes[0].legend(loc="upper left", frameon=True, framealpha=1.0)
    if title:
        fig.suptitle(title, y=1.02, fontsize=12)
    fig.tight_layout(w_pad=1.8)
    save_figure(fig, name, dirs)


def _mode_soc_power(df: pd.DataFrame, mode: str) -> tuple[np.ndarray, np.ndarray]:
    power_col = "Battery_Actual_kW" if "Battery_Actual_kW" in df.columns else "Battery_Power_kW"
    power = df[power_col].to_numpy(dtype=float)
    if mode == "discharge":
        mask = power > 1e-6
    else:
        mask = power < -1e-6
    sub = df.loc[mask]
    return sub["SOC"].to_numpy(dtype=float) * 100.0, np.abs(sub[power_col].to_numpy(dtype=float))


def _power_edges(pbm_power: np.ndarray, ebm_power: np.ndarray) -> np.ndarray:
    upper = _robust_upper_limit(pbm_power, ebm_power, quantile=0.995, floor=1.0)
    if upper <= 2.0:
        step = 0.25
    elif upper <= 6.0:
        step = 0.5
    else:
        step = 1.0
    edge_max = max(step * 4.0, math.ceil(upper / step) * step)
    return np.arange(0.0, edge_max + step, step)


def plot_soc_power_occupancy_difference(
    pbm_df: pd.DataFrame,
    ebm_df: pd.DataFrame,
    mode: str,
    title: str | None,
    name: str,
    dirs: dict[str, Path],
    soc_reference_pct: float | None = None,
    reference_label: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(4.1, 3.8))
    pbm_soc, pbm_power = _mode_soc_power(pbm_df, mode)
    ebm_soc, ebm_power = _mode_soc_power(ebm_df, mode)
    soc_edges = np.arange(10.0, 100.0, 5.0)
    power_edges = _power_edges(pbm_power, ebm_power)

    pbm_hist, _, _ = np.histogram2d(pbm_soc, pbm_power, bins=[soc_edges, power_edges])
    ebm_hist, _, _ = np.histogram2d(ebm_soc, ebm_power, bins=[soc_edges, power_edges])
    pbm_hist = pbm_hist / max(float(np.sum(pbm_hist)), 1.0)
    ebm_hist = ebm_hist / max(float(np.sum(ebm_hist)), 1.0)
    diff_pp = (pbm_hist - ebm_hist).T * 100.0
    vmax = max(float(np.max(np.abs(diff_pp))), 1.0)

    mesh = ax.pcolormesh(
        soc_edges,
        power_edges,
        diff_pp,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
        shading="auto",
    )
    cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Occupancy Difference (PBM - EBM, pp)")

    if soc_reference_pct is not None:
        ax.axvline(float(soc_reference_pct), color="#c7b98a", lw=1.8, ls="--")

    ax.set_xlabel("SOC (%)")
    ax.set_ylabel(r"$|P_{\mathrm{bess}}|$ (kW)")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100.0, decimals=0))
    if title:
        ax.set_title(title)

    if pbm_soc.size and ebm_soc.size:
        note_lines = [
            f"PBM mode hours: {len(pbm_soc)}",
            f"EBM mode hours: {len(ebm_soc)}",
        ]
        if soc_reference_pct is not None:
            label = reference_label or "Reference SOC"
            note_lines.extend(
                [
                    f"{label}: {soc_reference_pct:.0f}%",
                    f"PBM below ref.: {100.0 * np.mean(pbm_soc < soc_reference_pct):.1f}%",
                    f"EBM below ref.: {100.0 * np.mean(ebm_soc < soc_reference_pct):.1f}%",
                ]
            )
        note = "\n".join(note_lines)
    else:
        note = "No valid points"

    ax.text(
        0.985,
        0.03,
        note,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.2,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#fbfaf7", "edgecolor": "#b7b7b7"},
    )
    save_figure(fig, name, dirs)


def main() -> None:
    args = parse_args()
    set_paper_style()
    dirs = ensure_output_dirs()
    res_data = load_residential_cost_timeseries(seed=int(args.res_seed))
    cigre_cost = load_cigre_cost_timeseries(seed=int(args.cigre_seed))
    res_summary = load_case_summary("res")
    cigre_summary = load_case_summary("cigre")
    res_monthly = compute_monthly_gain_stats("res")
    cigre_monthly = compute_monthly_gain_stats("cigre")
    window_start_day, _ = select_high_gap_window(
        res_data["pbm"],
        res_data["ebm"],
        window_days=int(args.dispatch_window_days),
    )

    plot_cumulative_cost(
        res_data["pbm"],
        res_data["ebm"],
        title=None,
        name="MGRES_eta100_cumulative_cost",
        dirs=dirs,
        highlight_day_range=(window_start_day, window_start_day + int(args.dispatch_window_days) - 1),
        highlight_label="Dispatch zoom",
    )
    plot_cumulative_cost(
        cigre_cost["pbm"],
        cigre_cost["ebm"],
        title=None,
        name="CIGRE_eta100_cumulative_cost",
        dirs=dirs,
    )
    plot_seed_cost_dumbbell_pair(
        res_summary=res_summary,
        cigre_summary=cigre_summary,
        name="CASE_seed_cost_dumbbell_pair",
        dirs=dirs,
    )
    plot_monthly_gain_pair(
        res_stats=res_monthly,
        cigre_stats=cigre_monthly,
        name="CASE_monthly_pbm_advantage_pair",
        dirs=dirs,
    )
    plot_residential_dispatch_zoom(
        pbm_df=res_data["pbm"],
        ebm_df=res_data["ebm"],
        start_day=window_start_day,
        window_days=int(args.dispatch_window_days),
        name="MGRES_dispatch_zoom_window",
        dirs=dirs,
    )

    print("Saved case-study figures to:")
    for key in ("png", "pdf", "eps", "tiff"):
        print(f"  {key}: {dirs[key]}")


if __name__ == "__main__":
    main()
