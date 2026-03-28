#!/usr/bin/env python3
"""Generate refreshed PBM physics figures and a composite graphical abstract."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from matplotlib.ticker import FuncFormatter

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from microgrid_sim.cases import residential_battery_params

PBM_COLOR = "#0b6e8a"
PBM_LIGHT = "#60a5b8"
EBM_COLOR = "#c05a2b"
GOLD = "#d1b26f"
INK = "#22313f"
MUTED = "#5f6b76"
GRID = "#d9dee2"
PANEL = "#f6f7f8"
ACCENT_FILL = "#eef5f7"
SOFT_FILL = "#f7efe6"


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#2f3437",
            "axes.linewidth": 0.85,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "grid.alpha": 1.0,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
            "font.size": 10.5,
            "axes.titlesize": 11.2,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.2,
            "ytick.labelsize": 9.2,
            "legend.fontsize": 8.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def ensure_dirs() -> dict[str, dict[str, Path]]:
    paper_dirs = {
        "pdf": REPO_ROOT / "microgrid-paper" / "figures" / "pdf",
        "png": REPO_ROOT / "microgrid-paper" / "figures" / "png",
        "tiff": REPO_ROOT / "microgrid-paper" / "figures" / "tiff",
    }
    ga_dirs = {
        "pdf": REPO_ROOT / "microgrid-paper" / "figures" / "graphical-abstract" / "pdf",
        "png": REPO_ROOT / "microgrid-paper" / "figures" / "graphical-abstract" / "png",
        "tiff": REPO_ROOT / "microgrid-paper" / "figures" / "graphical-abstract" / "tiff",
    }
    for group in (paper_dirs, ga_dirs):
        for path in group.values():
            path.mkdir(parents=True, exist_ok=True)
    return {"paper": paper_dirs, "ga": ga_dirs}


def save_figure(fig: plt.Figure, name: str, dirs: dict[str, Path]) -> None:
    fig.savefig(dirs["pdf"] / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(dirs["png"] / f"{name}.png", bbox_inches="tight", dpi=450)
    fig.savefig(dirs["tiff"] / f"{name}.tiff", bbox_inches="tight", dpi=450)
    plt.close(fig)


def _interp_curve(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    return np.interp(x, xp, fp)


def _primary_timeseries(case_name: str, seed: int, branch: str) -> pd.DataFrame:
    base = REPO_ROOT / "results" / "paper" / case_name / f"seed{seed}"
    matches = sorted(
        path
        for path in base.glob(f"{branch}_seed{seed}_*_timeseries.csv")
        if "_sh" not in path.name
    )
    if not matches:
        raise FileNotFoundError(f"No primary timeseries found for {case_name} seed {seed} {branch}.")
    return pd.read_csv(matches[0])


def _summary(case_name: str) -> pd.DataFrame:
    return pd.read_csv(REPO_ROOT / "results" / "paper" / case_name / "summary_seeds.csv")


def plot_ocv_axis(
    ax: plt.Axes,
    *,
    title: str = "OCV-SOC relation",
    show_legend: bool = True,
    show_note: bool = True,
    short_labels: bool = False,
    inline_labels: bool = False,
) -> None:
    params = residential_battery_params()
    soc = np.linspace(0.0, 1.0, 400)
    soc_pct = soc * 100.0
    base = _interp_curve(soc, params.soc_breakpoints, np.asarray(params.ocv_values, dtype=float))
    charge = _interp_curve(soc, params.soc_breakpoints, np.asarray(params.ocv_charge_values, dtype=float))
    discharge = _interp_curve(soc, params.soc_breakpoints, np.asarray(params.ocv_discharge_values, dtype=float))

    band_label = "Band" if short_labels else "Hysteresis band"
    charge_label = "Chg." if short_labels else "Charge branch"
    discharge_label = "Disch." if short_labels else "Discharge branch"
    base_label = "Base" if short_labels else "Base OCV"

    ax.fill_between(soc_pct, discharge, charge, color=GOLD, alpha=0.24, label=band_label)
    ax.plot(soc_pct, charge, color=EBM_COLOR, lw=1.6, ls="--", label=charge_label)
    ax.plot(soc_pct, discharge, color=PBM_LIGHT, lw=1.6, ls="--", label=discharge_label)
    ax.plot(soc_pct, base, color=PBM_COLOR, lw=2.6, label=base_label)
    ax.set_xlim(0, 100)
    ax.set_ylim(2.45, 3.72)
    ax.set_xlabel("SOC (%)")
    ax.set_ylabel("Cell OCV (V)")
    ax.set_title(title)
    if show_legend:
        ax.legend(loc="upper left", frameon=True)
    if inline_labels:
        idx_charge = int(0.80 * (len(soc_pct) - 1))
        idx_discharge = int(0.90 * (len(soc_pct) - 1))
        ax.text(float(soc_pct[idx_charge]) + 1.1, charge[idx_charge] + 0.012, charge_label, fontsize=8.0, color=EBM_COLOR, va="center")
        ax.text(
            float(soc_pct[idx_discharge]) + 1.2,
            discharge[idx_discharge] - 0.024,
            discharge_label,
            fontsize=8.0,
            color=PBM_LIGHT,
            va="center",
        )
    if show_note:
        ax.text(
            0.98,
            0.04,
            "Shared LiFePO$_4$ lookup",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.8,
            color=MUTED,
        )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_rint_axis(
    ax: plt.Axes,
    inset_ax: plt.Axes | None = None,
    *,
    title: str = "$R_{\\mathrm{int}}$ varies with SOC and temperature",
    show_legend: bool = True,
    temp_labels: list[str] | None = None,
    inline_labels: bool = False,
) -> None:
    params = residential_battery_params()
    soc = np.linspace(0.0, 1.0, 400)
    soc_pct = soc * 100.0
    base = _interp_curve(soc, params.soc_breakpoints, np.asarray(params.r_int_values, dtype=float)) * 1000.0

    temp_levels = [10.0, 25.0, 40.0]
    temp_colors = ["#bb6b37", PBM_COLOR, "#4f8fc0"]
    legend_labels = temp_labels or ["10 C", "25 C", "40 C"]
    for temp_c, color, label in zip(temp_levels, temp_colors, legend_labels):
        factor = np.clip(
            1.0 + params.r_int_temp_coeff_per_c * (params.reference_temperature_c - temp_c),
            params.min_r_int_temp_factor,
            params.max_r_int_temp_factor,
        )
        ax.plot(soc_pct, base * factor, color=color, lw=2.15, label=label)

    ax.set_xlim(0, 100)
    ax.set_xlabel("SOC (%)")
    ax.set_ylabel("Cell $R_{\\mathrm{int}}$ (m$\\Omega$)")
    ax.set_title(title)
    if show_legend:
        ax.legend(loc="upper center", ncol=3, frameon=True)
    if inline_labels:
        x_loc = 88.0
        idx = int(np.searchsorted(soc_pct, x_loc))
        idx = min(idx, len(soc_pct) - 1)
        for color, label, offset in zip(temp_colors, legend_labels, [0.22, 0.0, -0.22]):
            y_val = base[idx] * np.clip(
                1.0 + params.r_int_temp_coeff_per_c * (params.reference_temperature_c - float(label.split()[0])),
                params.min_r_int_temp_factor,
                params.max_r_int_temp_factor,
            )
            ax.text(x_loc + 1.8, y_val + offset, label, fontsize=8.1, color=color, va="center")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    if inset_ax is not None:
        temps = np.linspace(0.0, 55.0, 220)
        factors = np.clip(
            1.0 + params.r_int_temp_coeff_per_c * (params.reference_temperature_c - temps),
            params.min_r_int_temp_factor,
            params.max_r_int_temp_factor,
        )
        inset_ax.plot(temps, factors, color=INK, lw=1.7)
        inset_ax.axvline(params.reference_temperature_c, color=PBM_COLOR, lw=1.0, ls="--")
        inset_ax.set_title("Temp factor", fontsize=8.6)
        inset_ax.set_xlabel("T (C)", fontsize=8.0)
        inset_ax.set_ylabel("x", fontsize=8.0, rotation=0, labelpad=7)
        inset_ax.tick_params(labelsize=7.2)
        inset_ax.grid(True, alpha=0.8)


def build_ocv_figure(dirs: dict[str, Path]) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.05))
    plot_ocv_axis(ax)
    save_figure(fig, "Z1_OCV_SOC_curve", dirs)


def build_rint_figure(dirs: dict[str, Path]) -> None:
    fig = plt.figure(figsize=(7.2, 4.15))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[3.4, 1.25], wspace=0.28)
    ax_main = fig.add_subplot(gs[0, 0])
    ax_inset = fig.add_subplot(gs[0, 1])
    plot_rint_axis(ax_main, ax_inset)
    save_figure(fig, "Z2_Rint_SOC_Temp", dirs)


def _draw_house_icon(ax: plt.Axes, x: float, y: float, scale: float) -> None:
    roof = Polygon(
        [[x, y + 0.07 * scale], [x + 0.09 * scale, y + 0.15 * scale], [x + 0.18 * scale, y + 0.07 * scale]],
        closed=True,
        fc=GOLD,
        ec=INK,
        lw=1.0,
    )
    body = Rectangle((x + 0.025 * scale, y), 0.13 * scale, 0.08 * scale, fc="white", ec=INK, lw=1.0)
    pv = Rectangle((x + 0.12 * scale, y + 0.10 * scale), 0.07 * scale, 0.035 * scale, angle=-18, fc=PBM_LIGHT, ec=INK, lw=0.9)
    battery = Rectangle((x + 0.205 * scale, y + 0.015 * scale), 0.05 * scale, 0.09 * scale, fc=ACCENT_FILL, ec=PBM_COLOR, lw=1.0)
    ax.add_patch(roof)
    ax.add_patch(body)
    ax.add_patch(pv)
    ax.add_patch(battery)


def _draw_grid_icon(ax: plt.Axes, x: float, y: float, scale: float) -> None:
    ax.plot([x + 0.03 * scale, x + 0.11 * scale], [y + 0.14 * scale, y], color=INK, lw=1.0)
    ax.plot([x + 0.11 * scale, x + 0.19 * scale], [y, y + 0.14 * scale], color=INK, lw=1.0)
    ax.plot([x + 0.07 * scale, x + 0.15 * scale], [y + 0.07 * scale, y + 0.07 * scale], color=INK, lw=1.0)
    ax.plot([x + 0.09 * scale, x + 0.13 * scale], [y + 0.14 * scale, y + 0.14 * scale], color=INK, lw=1.0)
    ax.plot([x + 0.10 * scale, x + 0.12 * scale], [y + 0.17 * scale, y + 0.17 * scale], color=INK, lw=1.0)
    ax.plot([x + 0.055 * scale, x + 0.165 * scale], [y - 0.005 * scale, y - 0.005 * scale], color=INK, lw=1.0)
    battery = Rectangle((x + 0.215 * scale, y + 0.02 * scale), 0.05 * scale, 0.09 * scale, fc=ACCENT_FILL, ec=PBM_COLOR, lw=1.0)
    ax.add_patch(battery)


def _boxed_text(ax: plt.Axes, x: float, y: float, w: float, h: float, title: str, lines: list[str], edge: str, fill: str) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        lw=1.2,
        ec=edge,
        fc=fill,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.text(x + 0.04 * w, y + 0.72 * h, title, transform=ax.transAxes, ha="left", va="center", fontsize=10.8, fontweight="bold", color=INK)
    ax.text(x + 0.04 * w, y + 0.48 * h, "\n".join(lines), transform=ax.transAxes, ha="left", va="top", fontsize=8.9, color=MUTED)


def _mini_card(ax: plt.Axes, x: float, y: float, w: float, h: float, title: str, detail: str, edge: str, fill: str) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.028",
        lw=1.2,
        ec=edge,
        fc=fill,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.text(x + 0.05 * w, y + 0.68 * h, title, transform=ax.transAxes, ha="left", va="center", fontsize=10.9, fontweight="bold", color=INK)
    ax.text(x + 0.05 * w, y + 0.34 * h, detail, transform=ax.transAxes, ha="left", va="center", fontsize=8.8, color=MUTED)


def _percent_box(ax: plt.Axes, title: str, value: float, face: str) -> None:
    ax.set_axis_off()
    patch = FancyBboxPatch(
        (0.02, 0.08),
        0.96,
        0.84,
        boxstyle="round,pad=0.016,rounding_size=0.04",
        lw=1.1,
        ec=PBM_COLOR,
        fc=face,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.text(0.08, 0.68, title, transform=ax.transAxes, ha="left", va="center", fontsize=10.4, fontweight="bold", color=INK)
    ax.text(0.08, 0.34, f"{value:.2f}%", transform=ax.transAxes, ha="left", va="center", fontsize=22, fontweight="bold", color=PBM_COLOR)


def _draw_gap_axis(ax: plt.Axes, days: np.ndarray, gap: np.ndarray, label: str, color: str) -> None:
    normalized_gap = gap / max(float(gap[-1]), 1e-9)
    ax.plot(days, normalized_gap, color=color, lw=2.1, label=label)
    ax.fill_between(days, 0.0, normalized_gap, color=color, alpha=0.14)


def build_graphical_abstract(dirs: dict[str, Path]) -> None:
    res_summary = _summary("res")
    cigre_summary = _summary("cigre")
    res_gap_pct = float(res_summary["gap_pct"].mean())
    cigre_gap_pct = float(cigre_summary["gap_pct"].mean())

    res_pbm = _primary_timeseries("res", 48, "pbm")
    res_ebm = _primary_timeseries("res", 48, "ebm")
    cigre_pbm = _primary_timeseries("cigre", 44, "pbm")
    cigre_ebm = _primary_timeseries("cigre", 44, "ebm")
    res_days = np.arange(len(res_pbm), dtype=float) / 24.0
    cigre_days = np.arange(len(cigre_pbm), dtype=float) / 24.0
    res_gap = res_ebm["Cumulative_Cost"].to_numpy(dtype=float) - res_pbm["Cumulative_Cost"].to_numpy(dtype=float)
    cigre_gap = cigre_ebm["Cumulative_Cost"].to_numpy(dtype=float) - cigre_pbm["Cumulative_Cost"].to_numpy(dtype=float)

    fig = plt.figure(figsize=(13.3, 4.85))
    outer = fig.add_gridspec(1, 3, width_ratios=[0.92, 1.18, 1.05], wspace=0.16)

    ax_left = fig.add_subplot(outer[0, 0])
    ax_left.set_axis_off()
    ax_left.set_xlim(0, 1)
    ax_left.set_ylim(0, 1)
    ax_left.text(0.03, 0.95, "Cases and protocol", ha="left", va="top", fontsize=11.8, fontweight="bold", color=INK, transform=ax_left.transAxes)

    _mini_card(ax_left, 0.03, 0.64, 0.43, 0.16, "MG-RES", "Residential", PBM_COLOR, PANEL)
    _mini_card(ax_left, 0.52, 0.64, 0.43, 0.16, "MG-CIGRE", "CIGRE PCC", PBM_COLOR, PANEL)

    _mini_card(ax_left, 0.05, 0.27, 0.22, 0.12, "EBM", "Fixed eff.", EBM_COLOR, SOFT_FILL)
    _mini_card(ax_left, 0.37, 0.27, 0.22, 0.12, "PBM", "OCV-$R_{int}$", PBM_COLOR, ACCENT_FILL)
    _mini_card(ax_left, 0.69, 0.27, 0.22, 0.12, "PBM eval", "PBM", PBM_COLOR, PANEL)
    for x0, x1 in ((0.29, 0.37), (0.61, 0.69)):
        arrow = FancyArrowPatch((x0, 0.33), (x1, 0.33), arrowstyle="-|>", mutation_scale=13, lw=1.15, color=INK, transform=ax_left.transAxes)
        ax_left.add_patch(arrow)
    ax_left.text(0.05, 0.17, "Training model only.", ha="left", va="center", fontsize=8.5, color=MUTED, transform=ax_left.transAxes)

    mid = outer[0, 1].subgridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.35)
    ax_ocv = fig.add_subplot(mid[0, 0])
    ax_rint = fig.add_subplot(mid[1, 0])
    plot_ocv_axis(
        ax_ocv,
        title="OCV hysteresis",
        show_legend=False,
        show_note=False,
        short_labels=True,
        inline_labels=True,
    )
    plot_rint_axis(
        ax_rint,
        None,
        title="$R_{int}$ vs SOC and T",
        show_legend=False,
        temp_labels=["10", "25", "40"],
        inline_labels=True,
    )
    ax_ocv.text(0.00, 1.08, "PBM battery physics", transform=ax_ocv.transAxes, ha="left", va="bottom", fontsize=11.6, fontweight="bold", color=INK)

    right = outer[0, 2].subgridspec(3, 1, height_ratios=[0.72, 0.72, 1.10], hspace=0.18)
    ax_box_res = fig.add_subplot(right[0, 0])
    ax_box_cigre = fig.add_subplot(right[1, 0])
    _percent_box(ax_box_res, "MG-RES", res_gap_pct, ACCENT_FILL)
    _percent_box(ax_box_cigre, "MG-CIGRE", cigre_gap_pct, PANEL)
    ax_box_res.text(0.02, 1.10, "PBM reduces annual cost", transform=ax_box_res.transAxes, ha="left", va="bottom", fontsize=11.6, fontweight="bold", color=INK)

    ax_gap = fig.add_subplot(right[2, 0])
    _draw_gap_axis(ax_gap, res_days, res_gap, "RES", PBM_COLOR)
    _draw_gap_axis(ax_gap, cigre_days, cigre_gap, "CIGRE", EBM_COLOR)
    ax_gap.set_xlim(0, 365)
    ax_gap.set_ylim(0, 1.03)
    ax_gap.set_xlabel("Day of year")
    ax_gap.set_ylabel("Norm. gap")
    ax_gap.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:.0%}"))
    ax_gap.legend(loc="upper left", frameon=True)
    ax_gap.set_title("Annual gap", fontsize=10.6)
    for spine in ("top", "right"):
        ax_gap.spines[spine].set_visible(False)
    save_figure(fig, "Graphical_Abstract_MFM", dirs)


def main() -> None:
    set_style()
    dirs = ensure_dirs()
    build_ocv_figure(dirs["paper"])
    build_rint_figure(dirs["paper"])
    build_graphical_abstract(dirs["ga"])


if __name__ == "__main__":
    main()
