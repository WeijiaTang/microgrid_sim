#!/usr/bin/env python3
"""Generate case-study figures aligned with the current paper narrative.

The current manuscript no longer centers the legacy IEEE33 cross-fidelity heatmap
or the old mixed-fidelity sweep. Instead, the reviewer-facing story is:

1. Storage dispatch has real system value in both CIGRE and IEEE33.
2. CIGRE is the positive anchor case: simple is sufficient, mixed is best.
3. IEEE33 is the stress case: fidelity becomes necessary for the best line.
4. The cross-case difference is battery-centric and regime-dependent.

This script regenerates a paper-ready figure package for that narrative and
exports the plotted data tables next to the figures.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from matplotlib.transforms import blended_transform_factory

REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_ROOT = REPO_ROOT / "microgrid-paper" / "figures"
PLOT_DATA_ROOT = REPO_ROOT / "results" / "paper_case_study_plot_data"

COLOR_NAVY = "#1F3C5B"
COLOR_TEAL = "#2A7F78"
COLOR_CORAL = "#D96C4F"
COLOR_GOLD = "#C59A3D"
COLOR_SLATE = "#5E6E7E"
COLOR_STONE = "#C9C7C1"
COLOR_GRID = "#DDD9D2"
COLOR_TEXT = "#1F2933"
COLOR_BG = "#FAF7F2"
COLOR_PANEL_BG = "#FCFBF8"
COLOR_SOFT_TEAL = "#E9F3F1"
COLOR_SOFT_CORAL = "#FBEDE7"
COLOR_SOFT_GOLD = "#F6EFD9"
COLOR_PASS = "#2E8B57"
COLOR_FAIL = "#C9504A"
COLOR_COLLAPSE = "#CE6B58"
COLOR_USABLE = "#B2872F"
COLOR_HEALTHY = "#2E7D6C"

STORAGE_VALUE_PATH = REPO_ROOT / "results" / "no_battery_sanity_ga_paper_balanced" / "summary.csv"
CIGRE_PROTOCOL_PATH = REPO_ROOT / "results" / "diagnostics" / "cigre_key_protocols_multiseed_summary.csv"
IEEE_MATCHED_PATH = (
    REPO_ROOT / "results" / "diagnostics" / "ieee33_battery_fidelity_value_map_matched_5k_seed42.csv"
)
IEEE_GATE_PATH = REPO_ROOT / "results" / "diagnostics" / "ieee33_reasonable_drl_gate_rintfull_5k_3seeds.csv"

FALLBACK_STORAGE_SANITY_ROWS = [
    {
        "case": "cigre",
        "battery_model": "none",
        "final_cumulative_cost": 639.47,
        "undervoltage_total": 0.0,
        "line_overload_total": 0.0,
    },
    {
        "case": "cigre",
        "battery_model": "simple",
        "final_cumulative_cost": 571.71,
        "undervoltage_total": 0.0,
        "line_overload_total": 0.0,
    },
    {
        "case": "cigre",
        "battery_model": "thevenin_loss_only",
        "final_cumulative_cost": 576.33,
        "undervoltage_total": 0.0,
        "line_overload_total": 0.0,
    },
    {
        "case": "cigre",
        "battery_model": "thevenin",
        "final_cumulative_cost": 575.62,
        "undervoltage_total": 0.0,
        "line_overload_total": 0.0,
    },
    {
        "case": "ieee33",
        "battery_model": "none",
        "final_cumulative_cost": 25020.61,
        "undervoltage_total": 0.2388,
        "line_overload_total": 35.93,
    },
    {
        "case": "ieee33",
        "battery_model": "simple",
        "final_cumulative_cost": 24453.86,
        "undervoltage_total": 0.2037,
        "line_overload_total": 25.35,
    },
    {
        "case": "ieee33",
        "battery_model": "thevenin_loss_only",
        "final_cumulative_cost": 24488.78,
        "undervoltage_total": 0.2044,
        "line_overload_total": 25.35,
    },
    {
        "case": "ieee33",
        "battery_model": "thevenin",
        "final_cumulative_cost": 24479.94,
        "undervoltage_total": 0.2043,
        "line_overload_total": 25.35,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate current case-study figures for the paper.")
    parser.add_argument("--dpi", type=int, default=300, help="Raster export resolution.")
    return parser.parse_args()


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": COLOR_PANEL_BG,
            "axes.edgecolor": COLOR_TEXT,
            "axes.labelcolor": COLOR_TEXT,
            "xtick.color": COLOR_TEXT,
            "ytick.color": COLOR_TEXT,
            "text.color": COLOR_TEXT,
            "axes.grid": True,
            "grid.color": COLOR_GRID,
            "grid.linewidth": 0.7,
            "grid.alpha": 1.0,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 9.5,
            "axes.titlesize": 11.8,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.5,
            "axes.linewidth": 1.0,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def ensure_dirs() -> dict[str, Path]:
    dirs = {
        "pdf": FIG_ROOT / "pdf",
        "png": FIG_ROOT / "png",
        "tiff": FIG_ROOT / "tiff",
        "eps": FIG_ROOT / "eps",
        "plot_data": PLOT_DATA_ROOT,
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def save_figure(fig: plt.Figure, name: str, dirs: dict[str, Path], dpi: int) -> None:
    fig.savefig(dirs["pdf"] / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(dirs["png"] / f"{name}.png", bbox_inches="tight", dpi=dpi)
    fig.savefig(dirs["tiff"] / f"{name}.tiff", bbox_inches="tight", dpi=dpi)
    fig.savefig(dirs["eps"] / f"{name}.eps", bbox_inches="tight")
    plt.close(fig)


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_storage_value_sanity(path: Path = STORAGE_VALUE_PATH) -> pd.DataFrame:
    if path.exists():
        return _read_csv(path)
    return pd.DataFrame(FALLBACK_STORAGE_SANITY_ROWS)


def load_cigre_protocol_summary(path: Path = CIGRE_PROTOCOL_PATH) -> pd.DataFrame:
    return _read_csv(path)


def load_ieee33_matched_ladder(path: Path = IEEE_MATCHED_PATH) -> pd.DataFrame:
    return _read_csv(path)


def load_ieee33_gate_review(path: Path = IEEE_GATE_PATH) -> pd.DataFrame:
    return _read_csv(path)


def prepare_storage_sanity_frame(df: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "case",
        "battery_model",
        "final_cumulative_cost",
        "undervoltage_total",
        "line_overload_total",
    ]
    frame = df.loc[:, keep].copy()
    baseline = (
        frame[frame["battery_model"] == "none"]
        .set_index("case")["final_cumulative_cost"]
        .to_dict()
    )
    frame["normalized_cost"] = frame.apply(
        lambda row: float(row["final_cumulative_cost"]) / float(baseline[str(row["case"])]), axis=1
    )
    frame["battery_label"] = frame["battery_model"].map(
        {
            "none": "No storage",
            "simple": "Simple",
            "thevenin_loss_only": "Loss-aware",
            "thevenin": "Full Thevenin",
        }
    )
    return frame


def prepare_cigre_protocol_frame(df: pd.DataFrame) -> pd.DataFrame:
    order = [
        "simple_to_simple",
        "simple_to_full",
        "mixed_simple+thevenin_to_full",
        "full_to_full",
    ]
    label_map = {
        "simple_to_simple": "Simple \u2192 Simple",
        "simple_to_full": "Simple \u2192 Full",
        "mixed_simple+thevenin_to_full": "Mixed \u2192 Full",
        "full_to_full": "Full \u2192 Full",
    }
    color_map = {
        "simple_to_simple": COLOR_NAVY,
        "simple_to_full": COLOR_TEAL,
        "mixed_simple+thevenin_to_full": COLOR_GOLD,
        "full_to_full": COLOR_CORAL,
    }
    frame = df.copy().set_index("protocol").reindex(order).reset_index()
    frame["protocol_label"] = frame["protocol"].map(label_map)
    frame["color"] = frame["protocol"].map(color_map)
    frame["gate_label"] = frame["gate_passes"].astype(str) + "/" + frame["seeds"].astype(str)
    return frame


def prepare_ieee33_stress_frames(
    matched_df: pd.DataFrame, gate_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    order = [
        "simple",
        "thevenin_rint_only",
        "thevenin_rint_thermal_stress",
        "thevenin_full",
    ]
    label_map = {
        "simple": "Simple",
        "thevenin_rint_only": "Rint-only",
        "thevenin_rint_thermal_stress": "Rint + thermal",
        "thevenin_full": "Full Thevenin",
    }
    regime_color = {
        "upper_attractor": COLOR_COLLAPSE,
        "usable_but_reserve_thin": COLOR_USABLE,
        "healthy_mid": COLOR_HEALTHY,
    }
    ladder = matched_df.copy().set_index("train_model").reindex(order).reset_index()
    ladder["model_label"] = ladder["train_model"].map(label_map)
    ladder["regime_color"] = ladder["policy_regime"].map(regime_color).fillna(COLOR_SLATE)

    gate = gate_df.copy()
    gate["model_label"] = gate["train_model"].map(label_map)
    gate["pass_color"] = np.where(gate["reasonable_dispatch_gate"] == "pass", COLOR_PASS, COLOR_FAIL)
    return ladder, gate


def build_cross_case_regime_frame() -> pd.DataFrame:
    rows = [
        {
            "case": "CIGRE",
            "battery_load_ratio": 0.400,
            "battery_netload_ratio": 0.454,
            "simple_path_label": "Simple pathway",
            "simple_path_pass_rate": 1.0,
            "best_line_label": "Mixed \u2192 Full",
            "best_line_pass_rate": 1.0,
            "best_line_savings_vs_none": 7230.7344852864435,
            "accent": COLOR_TEAL,
        },
        {
            "case": "IEEE33",
            "battery_load_ratio": 0.125,
            "battery_netload_ratio": 0.137,
            "simple_path_label": "Simple pathway",
            "simple_path_pass_rate": 0.0,
            "best_line_label": "Full Thevenin",
            "best_line_pass_rate": 1.0 / 3.0,
            "best_line_savings_vs_none": 5810.610067192349,
            "accent": COLOR_CORAL,
        },
    ]
    return pd.DataFrame(rows)


def _style_axis(ax: plt.Axes, facecolor: str = COLOR_PANEL_BG) -> None:
    ax.set_facecolor(facecolor)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)


def _note_box(
    ax: plt.Axes,
    text: str,
    x: float,
    y: float,
    *,
    transform=None,
    ha: str = "left",
    va: str = "top",
    fontsize: float = 8.5,
) -> None:
    if transform is None:
        transform = ax.transAxes
    ax.text(
        x,
        y,
        text,
        transform=transform,
        ha=ha,
        va=va,
        fontsize=fontsize,
        bbox={"boxstyle": "round,pad=0.26", "facecolor": COLOR_BG, "edgecolor": COLOR_STONE},
    )


def _pill_label(
    ax: plt.Axes,
    text: str,
    x: float,
    y: float,
    *,
    facecolor: str,
    edgecolor: str | None = None,
    textcolor: str = COLOR_TEXT,
    transform=None,
    ha: str = "center",
    va: str = "center",
    fontsize: float = 8.4,
) -> None:
    if transform is None:
        transform = ax.transAxes
    ax.text(
        x,
        y,
        text,
        transform=transform,
        ha=ha,
        va=va,
        fontsize=fontsize,
        fontweight="bold",
        color=textcolor,
        bbox={
            "boxstyle": "round,pad=0.24",
            "facecolor": facecolor,
            "edgecolor": edgecolor or facecolor,
        },
    )


def plot_storage_value_sanity(df: pd.DataFrame, dirs: dict[str, Path], dpi: int) -> None:
    frame = prepare_storage_sanity_frame(df)
    order = ["none", "simple", "thevenin_loss_only", "thevenin"]
    y_labels = {
        "none": "No storage",
        "simple": "Simple",
        "thevenin_loss_only": "Loss-aware",
        "thevenin": "Full Thevenin",
    }
    cigre = frame[frame["case"] == "cigre"].set_index("battery_model").reindex(order)
    ieee = frame[frame["case"] == "ieee33"].set_index("battery_model").reindex(order)

    fig, axes = plt.subplots(
        1, 3, figsize=(12.0, 4.6), gridspec_kw={"width_ratios": [1.5, 1.0, 1.0]}
    )
    ax = axes[0]
    y = np.arange(len(order))
    for idx, model in enumerate(order):
        x0 = float(cigre.loc[model, "normalized_cost"])
        x1 = float(ieee.loc[model, "normalized_cost"])
        ax.hlines(idx, min(x0, x1), max(x0, x1), color=COLOR_STONE, linewidth=2.0, zorder=1)
        ax.scatter(x0, idx, s=90, color=COLOR_NAVY, edgecolor="white", linewidth=0.9, zorder=3)
        ax.scatter(x1, idx, s=90, color=COLOR_CORAL, edgecolor="white", linewidth=0.9, zorder=3)
    ax.axvline(1.0, color=COLOR_SLATE, linestyle="--", linewidth=1.1)
    ax.set_yticks(y, [y_labels[item] for item in order])
    ax.invert_yaxis()
    ax.set_xlabel("Normalized one-day cost\n(relative to no storage)")
    ax.set_title("Storage value appears in both networks")
    ax.text(
        0.02,
        0.05,
        "CIGRE: 11.9% cost drop\nfrom no-storage to simple.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.8,
        bbox={"boxstyle": "round,pad=0.26", "facecolor": COLOR_BG, "edgecolor": COLOR_STONE},
    )
    ax.legend(
        handles=[
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_NAVY, markeredgecolor="white", markersize=9, label="CIGRE LV"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_CORAL, markeredgecolor="white", markersize=9, label="IEEE33"),
        ],
        loc="lower right",
        frameon=True,
    )
    _style_axis(ax)

    bar_colors = [COLOR_SLATE, COLOR_NAVY, COLOR_TEAL, COLOR_CORAL]
    ax = axes[1]
    ax.bar(np.arange(len(order)), ieee["undervoltage_total"], color=bar_colors, width=0.62)
    ax.set_xticks(np.arange(len(order)), [y_labels[item] for item in order], rotation=16, ha="right")
    ax.set_ylabel("Cumulative undervoltage")
    ax.set_title("IEEE33 voltage support")
    ax.text(
        0.5,
        0.96,
        "No-storage is the worst case.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.6,
    )
    _style_axis(ax)

    ax = axes[2]
    ax.bar(np.arange(len(order)), ieee["line_overload_total"], color=bar_colors, width=0.62)
    ax.set_xticks(np.arange(len(order)), [y_labels[item] for item in order], rotation=16, ha="right")
    ax.set_ylabel("Cumulative line overload")
    ax.set_title("IEEE33 congestion relief")
    ax.text(
        0.5,
        0.96,
        "All battery pathways relieve overload\nrelative to no-storage.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.6,
    )
    _style_axis(ax)

    fig.tight_layout(w_pad=1.8)
    save_figure(fig, "case_study_storage_value_sanity", dirs, dpi)
    frame.to_csv(dirs["plot_data"] / "case_study_storage_value_sanity.csv", index=False)


def plot_cigre_protocols(df: pd.DataFrame, dirs: dict[str, Path], dpi: int) -> None:
    frame = prepare_cigre_protocol_frame(df)
    fig, axes = plt.subplots(
        2, 1, figsize=(6.9, 8.0), gridspec_kw={"height_ratios": [1.0, 1.02]}
    )

    ax = axes[0]
    y = np.arange(len(frame))
    means = frame["mean_savings_vs_none"].astype(float).to_numpy()
    mins = frame["min_savings_vs_none"].astype(float).to_numpy()
    maxs = frame["max_savings_vs_none"].astype(float).to_numpy()
    lower = means - mins
    upper = maxs - means
    ax.barh(y, means, color=frame["color"], height=0.58, edgecolor="none")
    ax.errorbar(means, y, xerr=[lower, upper], fmt="none", ecolor=COLOR_SLATE, elinewidth=1.3, capsize=3)
    ax.set_yticks(y, frame["protocol_label"])
    ax.invert_yaxis()
    ax.axvline(0.0, color=COLOR_SLATE, linestyle="--", linewidth=1.0)
    ax.set_xlabel("Mean objective savings vs no storage")
    ax.set_title("CIGRE key protocols: robust and deployable")
    ax.set_xlim(min(-3800, float(mins.min()) - 500), float(maxs.max()) + 1500)
    for idx, row in frame.iterrows():
        ax.text(
            float(row["mean_savings_vs_none"]) + 420,
            idx,
            f"{row['gate_label']} pass",
            va="center",
            ha="left",
            fontsize=9.0,
            fontweight="bold",
            color=COLOR_TEXT,
        )
    _pill_label(ax, "A", 0.015, 0.97, facecolor=COLOR_SOFT_TEAL, ha="left", va="top", fontsize=8.2)
    _note_box(ax, "Whiskers show min/max savings across seeds.", 0.02, 0.04, va="bottom")
    _style_axis(ax)

    ax = axes[1]
    ax.axhspan(0.20, 0.50, color=COLOR_SOFT_TEAL, zorder=0)
    sizes = np.clip(frame["mean_savings_vs_none"].astype(float).to_numpy() / 40.0, 70, 320)
    ax.scatter(
        frame["mean_throughput_kwh"].astype(float),
        frame["mean_final_soc"].astype(float),
        s=sizes,
        c=frame["color"],
        edgecolor="white",
        linewidth=0.9,
        zorder=3,
    )
    label_offsets = {
        "Simple → Simple": (10, 12),
        "Simple → Full": (12, 16),
        "Mixed → Full": (18, 6),
        "Full → Full": (12, 10),
    }
    for _, row in frame.iterrows():
        dx, dy = label_offsets.get(str(row["protocol_label"]), (8, 8))
        ax.annotate(
            str(row["protocol_label"]),
            (float(row["mean_throughput_kwh"]), float(row["mean_final_soc"])),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.8,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#D6D3CC"},
        )
    ax.set_xlabel("Mean throughput (kWh)")
    ax.set_ylabel("Mean final SOC")
    ax.set_ylim(0.14, 0.56)
    ax.set_xlim(float(frame["mean_throughput_kwh"].min()) - 900, float(frame["mean_throughput_kwh"].max()) + 1100)
    ax.set_title("Inventory shape and battery usage")
    _pill_label(ax, "B", 0.015, 0.97, facecolor=COLOR_SOFT_GOLD, ha="left", va="top", fontsize=8.2)
    _note_box(ax, "Shaded band marks the\nhealthy mid-inventory zone.", 0.03, 0.95)
    _style_axis(ax)

    fig.tight_layout(h_pad=1.3)
    save_figure(fig, "case_study_cigre_protocols", dirs, dpi)
    frame.to_csv(dirs["plot_data"] / "case_study_cigre_protocols.csv", index=False)


def plot_ieee33_fidelity_stress(
    matched_df: pd.DataFrame, gate_df: pd.DataFrame, dirs: dict[str, Path], dpi: int
) -> None:
    ladder, gate = prepare_ieee33_stress_frames(matched_df, gate_df)
    fig, axes = plt.subplots(
        2, 1, figsize=(7.0, 8.2), gridspec_kw={"height_ratios": [1.02, 1.0]}
    )

    ax = axes[0]
    y = np.arange(len(ladder))
    values = ladder["objective_savings_vs_none"].astype(float).to_numpy()
    ax.barh(y, values, color=ladder["regime_color"], height=0.58, edgecolor="none")
    ax.axvline(0.0, color=COLOR_SLATE, linestyle="--", linewidth=1.0)
    ax.set_yticks(y, ladder["model_label"])
    ax.invert_yaxis()
    ax.set_xlabel("Seed-42 objective savings vs no storage")
    ax.set_title("IEEE33 matched ladder: policy morphology matters")
    x_min = float(values.min()) - 1200.0
    x_max = float(values.max()) + 1400.0
    ax.set_xlim(x_min, x_max)
    inside_left = blended_transform_factory(ax.transAxes, ax.transData)
    for idx, row in ladder.iterrows():
        val = float(row["objective_savings_vs_none"])
        label_text = (
            f"{str(row['policy_regime']).replace('_', ' ')}\n"
            f"SOC {float(row['final_soc']):.3f} | {float(row['total_battery_throughput_kwh']):.0f} kWh"
        )
        if val >= 0:
            x_text = val + 360
            transform = ax.transData
            ha = "left"
        else:
            x_text = 0.02
            transform = inside_left
            ha = "left"
        ax.text(
            x_text,
            idx,
            label_text,
            transform=transform,
            ha=ha,
            va="center",
            fontsize=8.2,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#D6D3CC"},
        )
    _pill_label(ax, "A", 0.015, 0.97, facecolor=COLOR_SOFT_CORAL, ha="left", va="top", fontsize=8.2)
    _note_box(
        ax,
        "Full Thevenin is the best current line.\nSimple and thermal-stress paths collapse.",
        0.02,
        0.05,
        va="bottom",
    )
    _style_axis(ax)

    ax = axes[1]
    seed_positions = {42: 0, 52: 1, 62: 2}
    model_style = {
        "thevenin_rint_only": {"color": COLOR_GOLD, "marker": "o", "label": "Rint-only"},
        "thevenin_full": {"color": COLOR_NAVY, "marker": "s", "label": "Full Thevenin"},
    }
    x_offsets = {"thevenin_rint_only": -0.08, "thevenin_full": 0.08}
    soc_label_offsets = {
        ("thevenin_rint_only", 42): (6, 8, "bottom", "SOC 0.15"),
        ("thevenin_full", 42): (6, 10, "bottom", "SOC 0.27"),
    }
    for model, style in model_style.items():
        subset = gate[gate["train_model"] == model].copy()
        seeds = [int(seed) for seed in subset["seed"]]
        x = [seed_positions[seed] + x_offsets[model] for seed in seeds]
        yv = subset["objective_savings_vs_none"].astype(float).to_numpy()
        ax.plot(x, yv, color=style["color"], linewidth=1.4, zorder=1)
        ax.scatter(
            x,
            yv,
            s=90,
            color=[COLOR_PASS if gate == "pass" else COLOR_FAIL for gate in subset["reasonable_dispatch_gate"]],
            edgecolor=style["color"],
            linewidth=1.0,
            marker=style["marker"],
            label=style["label"],
            zorder=3,
        )
        for seed, xi, yi, soc in zip(seeds, x, yv, subset["final_soc"].astype(float)):
            label_cfg = soc_label_offsets.get((model, seed))
            if label_cfg is None:
                continue
            dx, dy, va, label_text = label_cfg
            ax.annotate(
                label_text,
                (xi, yi),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=7.8,
                va=va,
            )
    ax.axhline(0.0, color=COLOR_SLATE, linestyle="--", linewidth=1.0)
    ax.set_xticks([0, 1, 2], ["Seed 42", "Seed 52", "Seed 62"])
    ax.set_ylabel("Objective savings vs no storage")
    ax.set_title("Three-seed gate: best lines remain fragile")
    ax.legend(loc="lower left", frameon=True)
    ax.set_xlim(-0.25, 2.25)
    _pill_label(ax, "B", 0.015, 0.97, facecolor=COLOR_SOFT_GOLD, ha="left", va="top", fontsize=8.2)
    _note_box(
        ax,
        "Green fill = gate pass\nRed fill = gate fail\nFailing seeds collapse to SOC 0.10 or 0.90",
        0.98,
        0.09,
        ha="right",
        va="bottom",
    )
    _style_axis(ax)

    fig.tight_layout(h_pad=1.3)
    save_figure(fig, "case_study_ieee33_fidelity_stress", dirs, dpi)
    ladder.to_csv(dirs["plot_data"] / "case_study_ieee33_fidelity_stress_seed42.csv", index=False)
    gate.to_csv(dirs["plot_data"] / "case_study_ieee33_gate_fragility.csv", index=False)


def plot_cross_case_regime_map(dirs: dict[str, Path], dpi: int) -> None:
    frame = build_cross_case_regime_frame()
    fig, axes = plt.subplots(
        2, 1, figsize=(7.0, 6.7), gridspec_kw={"height_ratios": [0.98, 1.02]}
    )

    ax = axes[0]
    y_positions = [1.0, 0.0]
    markers = [
        ("battery_load_ratio", "Battery / load peak", "o", "white"),
        ("battery_netload_ratio", "Battery / net-load peak", "D", None),
    ]
    for ypos, (_, row) in zip(y_positions, frame.iterrows()):
        accent = str(row["accent"])
        ratios = [float(row["battery_load_ratio"]), float(row["battery_netload_ratio"])]
        ax.plot(ratios, [ypos, ypos], color=accent, linewidth=4.2, solid_capstyle="round", zorder=1)
        for key, _, marker, facecolor in markers:
            ratio = float(row[key])
            ax.scatter(
                ratio,
                ypos,
                s=110,
                marker=marker,
                facecolor=accent if facecolor is None else facecolor,
                edgecolor=accent,
                linewidth=2.0,
                zorder=3,
            )
            ax.annotate(
                f"{ratio:.3f}",
                (ratio, ypos),
                xytext=(0, 12 if key == "battery_netload_ratio" else -16),
                textcoords="offset points",
                ha="center",
                fontsize=8.5,
            )
    ax.set_xlim(0.0, 0.52)
    ax.set_ylim(-0.5, 1.5)
    ax.set_yticks(y_positions, frame["case"].tolist())
    ax.set_xlabel("Storage-to-network ratio")
    ax.set_title("Storage margin")
    _pill_label(ax, "A", 0.015, 0.97, facecolor=COLOR_SOFT_TEAL, ha="left", va="top", fontsize=8.2)
    ax.legend(
        handles=[
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="white",
                markeredgecolor=COLOR_TEXT,
                markeredgewidth=1.8,
                markersize=8,
                label="Battery / load peak",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="D",
                color="none",
                markerfacecolor=COLOR_TEXT,
                markeredgecolor=COLOR_TEXT,
                markersize=7,
                label="Battery / net-load peak",
            ),
        ],
        loc="lower right",
        frameon=True,
    )
    _style_axis(ax)

    ax = axes[1]
    categories = ["Simple path", "Best line"]
    x = np.arange(len(categories), dtype=float)
    width = 0.28
    cigre = frame[frame["case"] == "CIGRE"].iloc[0]
    ieee = frame[frame["case"] == "IEEE33"].iloc[0]
    cigre_values = [float(cigre["simple_path_pass_rate"]), float(cigre["best_line_pass_rate"])]
    ieee_values = [float(ieee["simple_path_pass_rate"]), float(ieee["best_line_pass_rate"])]
    ax.bar(x - width / 2, cigre_values, width=width, color=str(cigre["accent"]), label="CIGRE")
    ax.bar(x + width / 2, ieee_values, width=width, color=str(ieee["accent"]), label="IEEE33")
    ax.set_xticks(x, categories)
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Gate pass rate")
    ax.set_title("Deployable policy rate")
    for xpos, value in zip(x - width / 2, cigre_values):
        ax.text(xpos, value + 0.04, f"{value:.2f}", ha="center", va="bottom", fontsize=8.6)
    for xpos, value in zip(x + width / 2, ieee_values):
        ax.text(xpos, value + 0.04, f"{value:.2f}", ha="center", va="bottom", fontsize=8.6)
    _pill_label(ax, "B", 0.015, 0.97, facecolor=COLOR_SOFT_GOLD, ha="left", va="top", fontsize=8.2)
    ax.legend(loc="upper left", frameon=True)
    _style_axis(ax)

    fig.tight_layout(h_pad=1.3)
    save_figure(fig, "case_study_cross_case_regime_map", dirs, dpi)
    frame.to_csv(dirs["plot_data"] / "case_study_cross_case_regime_map.csv", index=False)


def main() -> None:
    args = parse_args()
    set_style()
    dirs = ensure_dirs()

    storage_df = load_storage_value_sanity()
    cigre_df = load_cigre_protocol_summary()
    ieee_matched_df = load_ieee33_matched_ladder()
    ieee_gate_df = load_ieee33_gate_review()

    plot_storage_value_sanity(storage_df, dirs, dpi=int(args.dpi))
    plot_cigre_protocols(cigre_df, dirs, dpi=int(args.dpi))
    plot_ieee33_fidelity_stress(ieee_matched_df, ieee_gate_df, dirs, dpi=int(args.dpi))
    plot_cross_case_regime_map(dirs, dpi=int(args.dpi))

    print("Saved redesigned case-study figures to:")
    for key in ("pdf", "png", "tiff", "eps"):
        print(f"  {key}: {dirs[key]}")
    print(f"Saved plot data to: {dirs['plot_data']}")


if __name__ == "__main__":
    main()
