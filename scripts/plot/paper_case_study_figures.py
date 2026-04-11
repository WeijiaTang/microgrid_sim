#!/usr/bin/env python3
"""Generate reviewer-aligned case-study figures for the revised paper.

This script replaces the legacy MG-RES / MG-CIGRE annual PBM-vs-EBM plotting flow
with figures that match the current manuscript:

1. Storage-value sanity across CIGRE LV and modified IEEE33.
2. IEEE33 cross-fidelity train-test cost matrix.
3. Mixed-fidelity probe on common full-Thevenin deployment.

Outputs are written to ``microgrid-paper/figures`` in PDF/PNG/TIFF/EPS at
publication-ready resolution, together with CSV exports of the plotted data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_ROOT = REPO_ROOT / "microgrid-paper" / "figures"
PLOT_DATA_ROOT = REPO_ROOT / "results" / "paper_case_study_plot_data"

COLOR_NAVY = "#1A3A5C"
COLOR_CORAL = "#E05A47"
COLOR_SAND = "#D4A96A"
COLOR_TEAL = "#2C7A7B"
COLOR_SLATE = "#546E7A"
COLOR_GRID = "#D9DFE5"
COLOR_TEXT = "#22313F"
COLOR_BG = "#FBFAF7"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate revised case-study figures for the paper.")
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Raster export resolution for PNG/TIFF outputs.",
    )
    return parser.parse_args()


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": COLOR_TEXT,
            "axes.labelcolor": COLOR_TEXT,
            "xtick.color": COLOR_TEXT,
            "ytick.color": COLOR_TEXT,
            "text.color": COLOR_TEXT,
            "axes.grid": True,
            "grid.color": COLOR_GRID,
            "grid.linewidth": 0.6,
            "grid.alpha": 1.0,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
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


def load_storage_value_sanity() -> pd.DataFrame:
    path = REPO_ROOT / "results" / "no_battery_sanity_ga_paper_balanced" / "summary.csv"
    df = _read_csv(path)
    keep = [
        "case",
        "battery_model",
        "final_cumulative_cost",
        "undervoltage_total",
        "line_overload_total",
        "final_soc",
    ]
    return df.loc[:, keep].copy()


def load_cross_fidelity_detail() -> pd.DataFrame:
    path = (
        REPO_ROOT
        / "results"
        / "fidelity_summary_tables_ieee33_paper_balanced_5k_regularized_multiseed"
        / "paper_key_metrics.csv"
    )
    return _read_csv(path)


def load_mixed_fidelity_summaries() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    candidate_dirs = [
        "post_patch_ieee33_paper_balanced_5k_3x3_multiseed",
        "thevenin_staged_lossonly90_then_thevenin10_lr5e5_multiseed",
        "thevenin_staged_lossonly80_then_thevenin20_lr5e5_multiseed",
        "thevenin_staged_lossonly80_then_thevenin20_lr2e5_multiseed",
        "thevenin_staged_lossonly80_then_thevenin20_lr1e5_multiseed",
        "thevenin_staged_lossonly70_then_thevenin30_lr5e5_multiseed",
    ]
    for dir_name in candidate_dirs:
        path = REPO_ROOT / "results" / dir_name / "summary.csv"
        if not path.exists():
            continue
        frame = _read_csv(path)
        frame["source_dir_name"] = dir_name
        rows.append(frame)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def plot_storage_value_sanity(df: pd.DataFrame, dirs: dict[str, Path], dpi: int) -> None:
    case_order = ["cigre", "ieee33"]
    model_order = ["none", "simple", "thevenin_loss_only", "thevenin"]
    label_map = {
        "none": "No storage",
        "simple": "Simple",
        "thevenin_loss_only": "Loss-aware",
        "thevenin": "Full Thevenin",
    }
    color_map = {
        "none": COLOR_SLATE,
        "simple": COLOR_NAVY,
        "thevenin_loss_only": COLOR_TEAL,
        "thevenin": COLOR_CORAL,
    }

    fig, axes = plt.subplots(1, 3, figsize=(11.8, 4.2), gridspec_kw={"width_ratios": [1.2, 1.0, 1.0]})

    x = np.arange(len(model_order), dtype=float)
    width = 0.34
    cigre = df[df["case"] == "cigre"].set_index("battery_model").reindex(model_order)
    ieee = df[df["case"] == "ieee33"].set_index("battery_model").reindex(model_order)

    cigre_norm = cigre["final_cumulative_cost"] / float(cigre.loc["none", "final_cumulative_cost"])
    ieee_norm = ieee["final_cumulative_cost"] / float(ieee.loc["none", "final_cumulative_cost"])

    ax = axes[0]
    ax.bar(x - width / 2, cigre_norm, width=width, color=COLOR_NAVY, label="CIGRE LV")
    ax.bar(x + width / 2, ieee_norm, width=width, color=COLOR_CORAL, label="Modified IEEE33")
    ax.set_xticks(x, [label_map[m] for m in model_order], rotation=16, ha="right")
    ax.set_ylabel("Normalized one-day cost\n(relative to no storage)")
    ax.set_title("Storage-value sanity")
    ax.legend(loc="upper left", frameon=True)
    ax.set_ylim(0.85, 1.02)
    ax.text(
        0.98,
        0.03,
        "Absolute cost reference:\nCIGRE 639.47 -> 571.71\nIEEE33 25020.61 -> 24453.86",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.4,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": COLOR_BG, "edgecolor": "#C9C1B8"},
    )

    ax = axes[1]
    ieee_subset = ieee.loc[model_order]
    ax.bar(x, ieee_subset["undervoltage_total"], color=[color_map[m] for m in model_order], width=0.62)
    ax.set_xticks(x, [label_map[m] for m in model_order], rotation=16, ha="right")
    ax.set_ylabel("Cumulative undervoltage")
    ax.set_title("IEEE33 voltage support")

    ax = axes[2]
    ax.bar(x, ieee_subset["line_overload_total"], color=[color_map[m] for m in model_order], width=0.62)
    ax.set_xticks(x, [label_map[m] for m in model_order], rotation=16, ha="right")
    ax.set_ylabel("Cumulative line overload")
    ax.set_title("IEEE33 congestion relief")

    fig.tight_layout(w_pad=1.6)
    save_figure(fig, "case_study_storage_value_sanity", dirs, dpi)
    df.to_csv(dirs["plot_data"] / "case_study_storage_value_sanity.csv", index=False)


def plot_cross_fidelity_heatmap(df: pd.DataFrame, dirs: dict[str, Path], dpi: int) -> None:
    focus = df[
        (df["case"] == "ieee33")
        & (df["regime"] == "network_stress")
        & (df["reward_profile"] == "paper_balanced")
        & (df["train_steps"] == 5000)
        & (df["eval_steps"] == 72)
    ].copy()

    order = ["simple", "thevenin_loss_only", "thevenin"]
    label_map = {
        "simple": "Simple",
        "thevenin_loss_only": "Loss-aware\nThevenin",
        "thevenin": "Full\nThevenin",
    }
    pivot_cost = (
        focus.pivot_table(index="train_model", columns="test_model", values="final_cumulative_cost", aggfunc="mean")
        .reindex(index=order, columns=order)
    )
    pivot_soc = (
        focus.pivot_table(index="train_model", columns="test_model", values="final_soc", aggfunc="mean")
        .reindex(index=order, columns=order)
    )

    values = pivot_cost.to_numpy(dtype=float)
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    cmap = LinearSegmentedColormap.from_list("paper_heat", ["#F6E9E2", "#F3C7AF", "#E58A62", "#C7543E"])

    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    im = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean deployment cost")

    ax.set_xticks(np.arange(len(order)), [label_map[item] for item in order])
    ax.set_yticks(np.arange(len(order)), [label_map[item] for item in order])
    ax.set_xlabel("Deployment environment")
    ax.set_ylabel("Training pathway")
    ax.set_title("IEEE33 cross-fidelity cost matrix")

    for row_idx, train_model in enumerate(order):
        for col_idx, test_model in enumerate(order):
            cost = pivot_cost.loc[train_model, test_model]
            soc = pivot_soc.loc[train_model, test_model]
            text_color = "white" if cost > (vmin + vmax) / 2.0 else COLOR_TEXT
            ax.text(
                col_idx,
                row_idx,
                f"{cost:,.0f}\nSOC {soc:.3f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
                fontweight="bold" if train_model == "thevenin_loss_only" else "normal",
            )

    best_rows = focus.groupby("test_model", as_index=False)["final_cumulative_cost"].min()
    note = (
        "Loss-aware training is best in\n"
        "all three deployment environments.\n"
        "Family means: 75632 < 75718 < 75917."
    )
    ax.text(
        1.02,
        0.02,
        note,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.6,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": COLOR_BG, "edgecolor": "#C9C1B8"},
    )

    fig.tight_layout()
    save_figure(fig, "case_study_ieee33_cross_fidelity_heatmap", dirs, dpi)
    pivot_cost.to_csv(dirs["plot_data"] / "case_study_ieee33_cross_fidelity_cost_matrix.csv")
    pivot_soc.to_csv(dirs["plot_data"] / "case_study_ieee33_cross_fidelity_soc_matrix.csv")
    best_rows.to_csv(dirs["plot_data"] / "case_study_ieee33_cross_fidelity_best_rows.csv", index=False)


def _mixed_family_label(row: pd.Series) -> str:
    train_model = str(row["train_model"])
    source_dir_name = str(row.get("source_dir_name", ""))
    if train_model == "simple":
        return "Pure simple"
    if train_model == "thevenin_loss_only":
        return "Pure loss-aware"
    if train_model == "thevenin":
        return "Pure full Thevenin"

    fractions_raw = row.get("mixed_fidelity_stage_fractions", "")
    lrs_raw = row.get("mixed_fidelity_stage_learning_rates", "")
    fractions = "" if pd.isna(fractions_raw) else str(fractions_raw).strip()
    lrs = "" if pd.isna(lrs_raw) else str(lrs_raw).strip()
    if not fractions:
        if "lossonly90_then_thevenin10" in source_dir_name:
            fractions = "0.9,0.1"
        elif "lossonly80_then_thevenin20" in source_dir_name:
            fractions = "0.8,0.2"
        elif "lossonly70_then_thevenin30" in source_dir_name:
            fractions = "0.7,0.3"
    normalized_lrs = lrs.replace(" ", "").lower()
    if fractions == "0.9,0.1" and ("5e-05" in normalized_lrs or "5e-5" in normalized_lrs):
        return "Mixed 0.9/0.1\n5e-5"
    if fractions == "0.8,0.2" and ("5e-05" in normalized_lrs or "5e-5" in normalized_lrs):
        return "Mixed 0.8/0.2\n5e-5"
    if fractions == "0.8,0.2" and ("2e-05" in normalized_lrs or "2e-5" in normalized_lrs):
        return "Mixed 0.8/0.2\n2e-5"
    if fractions == "0.8,0.2" and ("1e-05" in normalized_lrs or "1e-5" in normalized_lrs):
        return "Mixed 0.8/0.2\n1e-5"
    if fractions == "0.7,0.3" and ("5e-05" in normalized_lrs or "5e-5" in normalized_lrs):
        return "Mixed 0.7/0.3\n5e-5"
    return source_dir_name or train_model


def plot_mixed_fidelity_tradeoff(df: pd.DataFrame, dirs: dict[str, Path], dpi: int) -> None:
    focus = df[(df["test_model"] == "thevenin") & (df["case"] == "ieee33")].copy()
    if focus.empty:
        return

    group_cols = [
        "source_dir_name",
        "train_model",
        "mixed_fidelity_stage_fractions",
        "mixed_fidelity_stage_learning_rates",
    ]
    metrics = ["final_cumulative_cost", "final_soc", "total_battery_throughput_kwh"]
    agg = focus.groupby(group_cols, dropna=False)[metrics].mean().reset_index()
    agg["family_label"] = agg.apply(_mixed_family_label, axis=1)

    order = [
        "Pure simple",
        "Pure loss-aware",
        "Mixed 0.9/0.1\n5e-5",
        "Mixed 0.8/0.2\n5e-5",
        "Mixed 0.8/0.2\n2e-5",
        "Mixed 0.8/0.2\n1e-5",
        "Mixed 0.7/0.3\n5e-5",
        "Pure full Thevenin",
    ]
    agg["family_label"] = pd.Categorical(agg["family_label"], categories=order, ordered=True)
    agg = agg.sort_values("family_label").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    x = np.arange(len(agg), dtype=float)
    scatter = ax.scatter(
        agg["total_battery_throughput_kwh"],
        agg["final_cumulative_cost"],
        c=agg["final_soc"],
        cmap="viridis",
        s=120,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean final SOC")

    for idx, row in agg.iterrows():
        ax.annotate(
            str(row["family_label"]),
            (row["total_battery_throughput_kwh"], row["final_cumulative_cost"]),
            xytext=(6, 6 if idx % 2 == 0 else -14),
            textcoords="offset points",
            fontsize=8.6,
            color=COLOR_TEXT,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "#D2D7DD"},
        )

    pure_loss_row = agg[agg["family_label"] == "Pure loss-aware"].iloc[0]
    pure_full_row = agg[agg["family_label"] == "Pure full Thevenin"].iloc[0]
    ax.plot(
        [pure_loss_row["total_battery_throughput_kwh"], pure_full_row["total_battery_throughput_kwh"]],
        [pure_loss_row["final_cumulative_cost"], pure_full_row["final_cumulative_cost"]],
        color="#C8CDD3",
        lw=1.2,
        ls="--",
        zorder=1,
    )

    ax.set_xlabel("Mean battery throughput (kWh)")
    ax.set_ylabel("Full-Thevenin deployment cost")
    ax.set_title("Mixed-fidelity trade-off under full-Thevenin deployment")
    ax.text(
        0.02,
        0.98,
        "Mixed-fidelity improves over pure full Thevenin,\n"
        "but the best point remains near the loss-aware baseline.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.7,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": COLOR_BG, "edgecolor": "#C9C1B8"},
    )

    fig.tight_layout()
    save_figure(fig, "case_study_mixed_fidelity_tradeoff", dirs, dpi)
    agg.to_csv(dirs["plot_data"] / "case_study_mixed_fidelity_tradeoff.csv", index=False)


def main() -> None:
    args = parse_args()
    set_style()
    dirs = ensure_dirs()

    storage_df = load_storage_value_sanity()
    cross_df = load_cross_fidelity_detail()
    mixed_df = load_mixed_fidelity_summaries()

    plot_storage_value_sanity(storage_df, dirs, dpi=int(args.dpi))
    plot_cross_fidelity_heatmap(cross_df, dirs, dpi=int(args.dpi))
    plot_mixed_fidelity_tradeoff(mixed_df, dirs, dpi=int(args.dpi))

    print("Saved revised case-study figures to:")
    for key in ("pdf", "png", "tiff", "eps"):
        print(f"  {key}: {dirs[key]}")
    print(f"Saved plot data to: {dirs['plot_data']}")


if __name__ == "__main__":
    main()
