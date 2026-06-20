#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "ieee33" / "network_stress" / "simple"
FIG_ROOT = ROOT / "microgrid-paper" / "figures"
PDF = FIG_ROOT / "pdf"
PNG = FIG_ROOT / "png"
SVG = FIG_ROOT / "svg"
TIFF = FIG_ROOT / "tiff"
SOURCE = FIG_ROOT / "source_data"

for folder in (PDF, PNG, SVG, TIFF, SOURCE):
    folder.mkdir(parents=True, exist_ok=True)

MODEL_ORDER = ["simple", "thevenin_loss_only", "thevenin_rint_only", "thevenin_full"]
MODEL_LABELS = ["Simple\nEBM", "Thevenin\nloss", "Thevenin\nRint", "Thevenin\nfull"]
P4_MIDBAND_THRESHOLD = 0.05
P4_INFEASIBLE_THRESHOLD = 0.30

NAVY = "#20364F"
BLUE = "#3E6FA8"
TEAL = "#2A9D8F"
ORANGE = "#D97941"
PURPLE = "#8E5EA2"
RED = "#C44E52"
GOLD = "#B8872D"
GREEN = "#2F7D5B"
GREY = "#6B7280"
LIGHT = "#E5E7EB"
BLACK = "#1B1B1B"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 7.5,
        "axes.titlesize": 8,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 6.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.65,
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
        "xtick.major.size": 2.8,
        "ytick.major.size": 2.8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.bbox": "tight",
    }
)


def panel_label(ax, text: str) -> None:
    ax.text(
        -0.12,
        1.04,
        text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
        color=BLACK,
    )


def save_figure(fig, name: str) -> None:
    fig.savefig(PDF / f"{name}.pdf")
    fig.savefig(SVG / f"{name}.svg")
    fig.savefig(PNG / f"{name}.png", dpi=600)
    fig.savefig(TIFF / f"{name}.tiff", dpi=600)
    plt.close(fig)


def seed_summary_paths() -> list[Path]:
    paths: dict[int, Path] = {}
    for path in RESULTS.glob("seed_*_summary.csv"):
        seed = int(path.stem.split("_")[1])
        paths[seed] = path
    for path in RESULTS.glob("seed_*/summary.csv"):
        seed = int(path.parent.name.split("_")[1])
        paths[seed] = path
    if not paths:
        raise FileNotFoundError(f"No seed summary CSV files found under {RESULTS}")
    return [paths[seed] for seed in sorted(paths)]


def load_seed_summaries() -> pd.DataFrame:
    frames = []
    for path in seed_summary_paths():
        frame = pd.read_csv(path)
        if "seed" not in frame.columns:
            frame["seed"] = int(path.parent.name.split("_")[1])
        frame["source_file"] = path.relative_to(ROOT).as_posix()
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    required = {
        "seed",
        "test_model",
        "final_cumulative_objective_cost",
        "soc_midband_dwell_fraction",
        "infeasible_action_dwell_fraction",
        "peak_price_discharge_action_fraction",
        "valley_price_charge_action_fraction",
        "mean_abs_shield_delta",
        "shield_material_activation_fraction",
        "shield_strong_activation_fraction",
        "mean_abs_inventory_teacher_gap",
        "final_terminal_soc_deviation",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required summary columns: {missing}")
    df["p4_selected"] = (
        (df["soc_midband_dwell_fraction"] > P4_MIDBAND_THRESHOLD)
        & (df["infeasible_action_dwell_fraction"] < P4_INFEASIBLE_THRESHOLD)
    )
    return df


def simple_seed_rows(df: pd.DataFrame) -> pd.DataFrame:
    simple = df[df["test_model"].eq("simple")].copy()
    duplicate = simple["seed"].duplicated().any()
    if duplicate:
        raise ValueError("Duplicate simple-model rows found for at least one seed")
    return simple.sort_values("seed")


def p4_selected_seeds(df: pd.DataFrame) -> list[int]:
    seeds = simple_seed_rows(df).loc[lambda x: x["p4_selected"], "seed"].astype(int).tolist()
    if len(seeds) != 6:
        raise ValueError(f"Expected 6 P4-selected seeds from experiment data, found {len(seeds)}: {seeds}")
    return seeds


def p4_cross_fidelity(df: pd.DataFrame, seeds: list[int]) -> pd.DataFrame:
    subset = df[df["seed"].isin(seeds) & df["test_model"].isin(MODEL_ORDER)].copy()
    counts = subset.groupby("seed")["test_model"].nunique()
    incomplete = counts[counts != len(MODEL_ORDER)]
    if not incomplete.empty:
        raise ValueError(f"Missing cross-fidelity rows for seeds: {incomplete.to_dict()}")
    subset["model_order"] = subset["test_model"].map({m: i for i, m in enumerate(MODEL_ORDER)})
    return subset.sort_values(["seed", "model_order"])


def fig1_inventory_teacher_gap_elite(df: pd.DataFrame, seeds: list[int]) -> None:
    simple = simple_seed_rows(df)
    data = simple[simple["seed"].isin(seeds)].copy()
    data.to_csv(SOURCE / "fig1_inventory_teacher_gap_source.csv", index=False)

    summary = pd.DataFrame(
        {
            "metric": [
                "SOC mid-band dwell",
                "Peak discharge",
                "Valley charge",
                "Infeasible dwell",
                "Terminal SOC deviation",
                "Inventory-teacher gap",
            ],
            "mean": [
                data["soc_midband_dwell_fraction"].mean(),
                data["peak_price_discharge_action_fraction"].mean(),
                data["valley_price_charge_action_fraction"].mean(),
                data["infeasible_action_dwell_fraction"].mean(),
                data["final_terminal_soc_deviation"].mean(),
                data["mean_abs_inventory_teacher_gap"].mean(),
            ],
            "std": [
                data["soc_midband_dwell_fraction"].std(ddof=1),
                data["peak_price_discharge_action_fraction"].std(ddof=1),
                data["valley_price_charge_action_fraction"].std(ddof=1),
                data["infeasible_action_dwell_fraction"].std(ddof=1),
                data["final_terminal_soc_deviation"].std(ddof=1),
                data["mean_abs_inventory_teacher_gap"].std(ddof=1),
            ],
            "n": [len(data)] * 6,
        }
    )
    summary.to_csv(SOURCE / "fig1_inventory_teacher_gap_summary_source.csv", index=False)

    fig = plt.figure(figsize=(7.4, 3.25), constrained_layout=True)
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.15, 0.9, 1.0])
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    ax_c = fig.add_subplot(gs[2])

    x = np.arange(len(data))
    width = 0.36
    ax_a.bar(
        x - width / 2,
        data["mean_abs_inventory_teacher_gap"],
        width=width,
        color=TEAL,
        label="Inventory-teacher action gap",
    )
    ax_a.bar(
        x + width / 2,
        data["mean_abs_shield_delta"],
        width=width,
        color=PURPLE,
        label="Shield delta",
    )
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(data["seed"].astype(int).astype(str))
    ax_a.set_ylabel("Mean absolute normalized action gap")
    ax_a.set_xlabel("P4-selected seed")
    ax_a.grid(axis="y", color=LIGHT, linewidth=0.5)
    ax_a.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2)
    panel_label(ax_a, "a")

    terminal = data["final_terminal_soc_deviation"].to_numpy()
    ax_b.bar(x, terminal, color=GOLD, width=0.58)
    ax_b.axhline(terminal.mean(), color=BLACK, linewidth=0.8, linestyle=(0, (3, 2)))
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(data["seed"].astype(int).astype(str))
    ax_b.set_ylim(0, max(0.45, terminal.max() * 1.16))
    ax_b.set_ylabel("Final SOC deviation")
    ax_b.set_xlabel("P4-selected seed")
    ax_b.grid(axis="y", color=LIGHT, linewidth=0.5)
    ax_b.text(len(x) - 0.52, terminal.mean() + 0.008, "mean", ha="right", va="bottom", fontsize=6.4, color=BLACK)
    for xi, value in zip(x, terminal):
        ax_b.text(xi, value + 0.012, f"{value:.2f}", ha="center", va="bottom", fontsize=6.1)
    panel_label(ax_b, "b")

    plot_summary = summary.copy()
    colors = [TEAL, BLUE, GOLD, ORANGE, PURPLE, GREEN]
    y = np.arange(len(plot_summary))[::-1]
    ax_c.barh(y, plot_summary["mean"], xerr=plot_summary["std"], color=colors, alpha=0.86, capsize=2)
    ax_c.set_yticks(y)
    ax_c.set_yticklabels(plot_summary["metric"])
    ax_c.set_xlim(0, 0.62)
    ax_c.set_xlabel("Dimensionless diagnostic value")
    ax_c.grid(axis="x", color=LIGHT, linewidth=0.5)
    for yi, value, spread in zip(y, plot_summary["mean"], plot_summary["std"]):
        label_x = min(float(value + spread + 0.02), 0.58)
        ax_c.text(label_x, yi, f"{value:.2f}", va="center", fontsize=6.5)
    panel_label(ax_c, "c")

    save_figure(fig, "fig1_inventory_teacher_gap_elite")


def fig2_cross_fidelity_elite(df: pd.DataFrame, seeds: list[int]) -> None:
    data = p4_cross_fidelity(df, seeds)
    data.to_csv(SOURCE / "fig2_cross_fidelity_elite_source.csv", index=False)
    data.to_csv(SOURCE / "fig3_cross_fidelity_heatmap_source.csv", index=False)

    fig = plt.figure(figsize=(7.2, 3.3), constrained_layout=True)
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.15, 1.0])
    ax_cost = fig.add_subplot(gs[0])
    ax_morph = fig.add_subplot(gs[1])
    model_x = np.arange(len(MODEL_ORDER))
    palette = [BLUE, TEAL, ORANGE, PURPLE, GREEN, RED]

    for color, seed in zip(palette, seeds):
        rows = data[data["seed"].eq(seed)].sort_values("model_order")
        ax_cost.plot(
            model_x,
            rows["final_cumulative_objective_cost"] / 1e6,
            marker="o",
            linewidth=1.1,
            markersize=4.0,
            color=color,
            alpha=0.88,
            label=f"Seed {seed}",
        )
    grouped_cost = data.groupby("model_order")["final_cumulative_objective_cost"]
    mean_cost = grouped_cost.mean().reindex(model_x) / 1e6
    std_cost = grouped_cost.std(ddof=1).reindex(model_x) / 1e6
    ax_cost.plot(model_x, mean_cost, color=BLACK, linewidth=1.8, linestyle="--", label="Mean")
    ax_cost.fill_between(model_x, mean_cost - std_cost, mean_cost + std_cost, color=BLACK, alpha=0.09)
    ax_cost.set_xticks(model_x)
    ax_cost.set_xticklabels(MODEL_LABELS)
    ax_cost.set_ylabel("Objective cost (M¥)")
    ax_cost.grid(axis="y", color=LIGHT, linewidth=0.5)
    ax_cost.legend(
        frameon=False,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        columnspacing=0.8,
        handlelength=1.5,
    )
    panel_label(ax_cost, "a")

    morphology = data.copy()
    morphology["morphology_score"] = 100 * (
        morphology["soc_midband_dwell_fraction"] - morphology["infeasible_action_dwell_fraction"]
    )
    morph_simple = morphology[morphology["test_model"].eq("simple")].set_index("seed")
    morph_full = morphology[morphology["test_model"].eq("thevenin_full")].set_index("seed")
    for color, seed in zip(palette, seeds):
        y0 = float(morph_simple.loc[seed, "morphology_score"])
        y1 = float(morph_full.loc[seed, "morphology_score"])
        ax_morph.plot([0, 1], [y0, y1], marker="o", linewidth=1.2, markersize=4.2, color=color)
        ax_morph.text(-0.04, y0, str(seed), ha="right", va="center", color=color, fontsize=6.5)
        ax_morph.text(1.04, y1, f"{y1:.1f}", ha="left", va="center", color=color, fontsize=6.5)
    ax_morph.axhline(0, color=GREY, linewidth=0.6, linestyle=(0, (3, 2)))
    ax_morph.set_xlim(-0.16, 1.18)
    ax_morph.set_xticks([0, 1])
    ax_morph.set_xticklabels(["Simple EBM", "Thevenin full"])
    ax_morph.set_ylabel("Morphology score\n(mid-band% − infeasible%)")
    ax_morph.grid(axis="y", color=LIGHT, linewidth=0.5)
    panel_label(ax_morph, "b")

    save_figure(fig, "fig2_cross_fidelity_elite")


def fig3_morphology_density(df: pd.DataFrame, seeds: list[int]) -> None:
    data = simple_seed_rows(df)
    selected = data[data["seed"].isin(seeds)].copy()
    selected.to_csv(SOURCE / "fig3_morphology_density_elite_source.csv", index=False)
    selected.to_csv(SOURCE / "fig2_morphology_scatter_source.csv", index=False)

    fig = plt.figure(figsize=(7.4, 3.15), constrained_layout=True)
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.0, 1.18], wspace=0.18)
    ax_scatter = fig.add_subplot(gs[0, 0])
    ax_profile = fig.add_subplot(gs[0, 1])

    ax_scatter.add_patch(
        Rectangle(
            (0, P4_MIDBAND_THRESHOLD * 100),
            P4_INFEASIBLE_THRESHOLD * 100,
            55 - P4_MIDBAND_THRESHOLD * 100,
            facecolor=TEAL,
            edgecolor="none",
            alpha=0.08,
            zorder=0,
        )
    )
    ax_scatter.axvline(P4_INFEASIBLE_THRESHOLD * 100, color=RED, linewidth=1.0, linestyle=(0, (4, 2)))
    ax_scatter.axhline(P4_MIDBAND_THRESHOLD * 100, color=BLUE, linewidth=1.0, linestyle=(0, (4, 2)))

    scatter = ax_scatter.scatter(
        selected["infeasible_action_dwell_fraction"] * 100,
        selected["soc_midband_dwell_fraction"] * 100,
        s=118,
        marker="o",
        linewidth=0.8,
        edgecolor=BLACK,
        c=selected["final_cumulative_objective_cost"] / 1e6,
        cmap="viridis_r",
        zorder=3,
    )

    label_offsets = {
        42: (0.8, 1.7),
        44: (0.8, 1.7),
        81: (0.8, -3.0),
        83: (0.8, 1.7),
        84: (0.9, -2.7),
        93: (0.8, 1.5),
    }
    for _, row in selected.iterrows():
        seed = int(row["seed"])
        dx, dy = label_offsets.get(seed, (0.8, 1.0))
        ax_scatter.text(
            row["infeasible_action_dwell_fraction"] * 100 + dx,
            row["soc_midband_dwell_fraction"] * 100 + dy,
            str(seed),
            fontsize=6.9,
            color=BLACK,
        )
    ax_scatter.text(
        29.2,
        52.5,
        "infeasible gate",
        ha="right",
        va="top",
        fontsize=6.5,
        color=RED,
    )
    ax_scatter.text(
        16.0,
        P4_MIDBAND_THRESHOLD * 100 + 1.2,
        "mid-band gate",
        ha="left",
        va="bottom",
        fontsize=6.5,
        color=BLUE,
    )
    ax_scatter.set_xlim(-1.5, 31.5)
    ax_scatter.set_ylim(-1.5, 55)
    ax_scatter.set_xlabel("Infeasible-action dwell (%)")
    ax_scatter.set_ylabel("SOC mid-band dwell (%)")
    ax_scatter.grid(color=LIGHT, linewidth=0.45)
    panel_label(ax_scatter, "a")

    profile = selected.set_index("seed").loc[seeds]
    y = np.arange(len(seeds))
    midband = profile["soc_midband_dwell_fraction"] * 100
    infeasible = profile["infeasible_action_dwell_fraction"] * 100
    ax_profile.barh(y + 0.17, midband, height=0.28, color=TEAL, label="Mid-band dwell")
    ax_profile.barh(y - 0.17, infeasible, height=0.28, color=ORANGE, label="Infeasible dwell")
    ax_profile.axvline(P4_MIDBAND_THRESHOLD * 100, color=BLUE, linewidth=0.8, linestyle=(0, (4, 2)))
    ax_profile.axvline(P4_INFEASIBLE_THRESHOLD * 100, color=RED, linewidth=0.8, linestyle=(0, (4, 2)))
    for yi, value in zip(y, midband):
        ax_profile.text(value + 0.8, yi + 0.17, f"{value:.1f}", va="center", fontsize=6.2, color=BLACK)
    for yi, value in zip(y, infeasible):
        ax_profile.text(value + 0.8, yi - 0.17, f"{value:.1f}", va="center", fontsize=6.2, color=BLACK)
    ax_profile.set_yticks(y)
    ax_profile.set_yticklabels([str(seed) for seed in seeds])
    ax_profile.invert_yaxis()
    ax_profile.set_xlim(0, 55)
    ax_profile.set_xlabel("Dwell fraction (%)")
    ax_profile.set_ylabel("P4-selected seed")
    ax_profile.grid(axis="x", color=LIGHT, linewidth=0.45)
    ax_profile.legend(frameon=False, loc="upper right")
    panel_label(ax_profile, "b")

    cbar = fig.colorbar(scatter, ax=ax_scatter, orientation="horizontal", pad=0.16, fraction=0.052)
    cbar.set_label("Objective cost (M¥)", labelpad=1.5)
    cbar.ax.tick_params(labelsize=5.8, length=2)

    save_figure(fig, "fig3_morphology_density_elite")


def fig5_candidate_pool_screening(df: pd.DataFrame, seeds: list[int]) -> None:
    data = simple_seed_rows(df).copy()
    data["p4_selected"] = data["seed"].isin(seeds)
    source_cols = [
        "seed",
        "p4_selected",
        "soc_midband_dwell_fraction",
        "infeasible_action_dwell_fraction",
        "final_cumulative_objective_cost",
        "mean_abs_inventory_teacher_gap",
        "final_terminal_soc_deviation",
        "peak_price_discharge_action_fraction",
    ]
    data[source_cols].to_csv(SOURCE / "fig5_candidate_pool_screening_source.csv", index=False)

    fig, ax = plt.subplots(figsize=(3.52, 2.85), constrained_layout=True)
    ax.add_patch(
        Rectangle(
            (0, P4_MIDBAND_THRESHOLD * 100),
            P4_INFEASIBLE_THRESHOLD * 100,
            55 - P4_MIDBAND_THRESHOLD * 100,
            facecolor=TEAL,
            edgecolor="none",
            alpha=0.08,
            zorder=0,
        )
    )
    ax.axvline(P4_INFEASIBLE_THRESHOLD * 100, color=RED, linewidth=0.9, linestyle=(0, (4, 2)))
    ax.axhline(P4_MIDBAND_THRESHOLD * 100, color=BLUE, linewidth=0.9, linestyle=(0, (4, 2)))

    rejected = data[~data["p4_selected"]]
    selected = data[data["p4_selected"]]
    ax.scatter(
        rejected["infeasible_action_dwell_fraction"] * 100,
        rejected["soc_midband_dwell_fraction"] * 100,
        s=42,
        color="#B8C0CC",
        edgecolor="white",
        linewidth=0.5,
        alpha=0.86,
        zorder=2,
    )
    ax.scatter(
        selected["infeasible_action_dwell_fraction"] * 100,
        selected["soc_midband_dwell_fraction"] * 100,
        s=76,
        color=TEAL,
        edgecolor=BLACK,
        linewidth=0.65,
        zorder=3,
    )
    for _, row in selected.iterrows():
        ax.text(
            row["infeasible_action_dwell_fraction"] * 100 + 1.2,
            row["soc_midband_dwell_fraction"] * 100 + 1.1,
            str(int(row["seed"])),
            fontsize=6.3,
            color=BLACK,
        )

    ax.text(
        7.0,
        52.0,
        "P4 pass region",
        ha="left",
        va="top",
        fontsize=6.5,
        color=TEAL,
    )
    ax.set_xlim(-2, 105)
    ax.set_ylim(-2, 55)
    ax.set_xlabel("Infeasible-action dwell (%)")
    ax.set_ylabel("SOC mid-band dwell (%)")
    ax.grid(color=LIGHT, linewidth=0.45)
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=TEAL, markeredgecolor=BLACK, markersize=5.8, label="P4-selected"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#B8C0CC", markeredgecolor="white", markersize=5.2, label="Not selected"),
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="upper right")
    ax.text(
        0.98,
        0.08,
        f"{len(selected)} / {len(data)} candidates selected",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.4,
        color=BLACK,
    )

    save_figure(fig, "fig5_candidate_pool_screening")

def main() -> None:
    df = load_seed_summaries()
    seeds = p4_selected_seeds(df)
    fig1_inventory_teacher_gap_elite(df, seeds)
    fig2_cross_fidelity_elite(df, seeds)
    fig3_morphology_density(df, seeds)
    fig5_candidate_pool_screening(df, seeds)
    print(f"Generated data-driven elite figures for {len(seeds)} P4-selected seeds; seeds={seeds}")


if __name__ == "__main__":
    main()
