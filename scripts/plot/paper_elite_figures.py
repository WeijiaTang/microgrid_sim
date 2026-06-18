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
SUCCESS_MIDBAND_THRESHOLD = 0.05
SUCCESS_INFEASIBLE_THRESHOLD = 0.30

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
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required summary columns: {missing}")
    df["success"] = (
        (df["soc_midband_dwell_fraction"] > SUCCESS_MIDBAND_THRESHOLD)
        & (df["infeasible_action_dwell_fraction"] < SUCCESS_INFEASIBLE_THRESHOLD)
    )
    return df


def simple_seed_rows(df: pd.DataFrame) -> pd.DataFrame:
    simple = df[df["test_model"].eq("simple")].copy()
    duplicate = simple["seed"].duplicated().any()
    if duplicate:
        raise ValueError("Duplicate simple-model rows found for at least one seed")
    return simple.sort_values("seed")


def success_seeds(df: pd.DataFrame) -> list[int]:
    seeds = simple_seed_rows(df).loc[lambda x: x["success"], "seed"].astype(int).tolist()
    if len(seeds) != 6:
        raise ValueError(f"Expected 6 successful seeds from experiment data, found {len(seeds)}: {seeds}")
    return seeds


def success_cross_fidelity(df: pd.DataFrame, seeds: list[int]) -> pd.DataFrame:
    subset = df[df["seed"].isin(seeds) & df["test_model"].isin(MODEL_ORDER)].copy()
    counts = subset.groupby("seed")["test_model"].nunique()
    incomplete = counts[counts != len(MODEL_ORDER)]
    if not incomplete.empty:
        raise ValueError(f"Missing cross-fidelity rows for seeds: {incomplete.to_dict()}")
    subset["model_order"] = subset["test_model"].map({m: i for i, m in enumerate(MODEL_ORDER)})
    return subset.sort_values(["seed", "model_order"])


def fig1_shield_dependence_elite(df: pd.DataFrame, seeds: list[int]) -> None:
    simple = simple_seed_rows(df)
    data = simple[simple["seed"].isin(seeds)].copy()
    data.to_csv(SOURCE / "fig1_shield_dependence_source.csv", index=False)

    summary = pd.DataFrame(
        {
            "metric": [
                "SOC mid-band dwell",
                "Peak discharge",
                "Valley charge",
                "Infeasible dwell",
                "Mean shield delta",
                "Inventory-teacher gap",
            ],
            "mean": [
                data["soc_midband_dwell_fraction"].mean(),
                data["peak_price_discharge_action_fraction"].mean(),
                data["valley_price_charge_action_fraction"].mean(),
                data["infeasible_action_dwell_fraction"].mean(),
                data["mean_abs_shield_delta"].mean(),
                data["mean_abs_inventory_teacher_gap"].mean(),
            ],
            "std": [
                data["soc_midband_dwell_fraction"].std(ddof=1),
                data["peak_price_discharge_action_fraction"].std(ddof=1),
                data["valley_price_charge_action_fraction"].std(ddof=1),
                data["infeasible_action_dwell_fraction"].std(ddof=1),
                data["mean_abs_shield_delta"].std(ddof=1),
                data["mean_abs_inventory_teacher_gap"].std(ddof=1),
            ],
            "n": [len(data)] * 6,
        }
    )
    summary.to_csv(SOURCE / "fig1_shield_dependence_summary_source.csv", index=False)

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
        label="Inventory-teacher gap",
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
    ax_a.set_ylabel("Mean absolute action gap")
    ax_a.set_xlabel("Successful seed")
    ax_a.grid(axis="y", color=LIGHT, linewidth=0.5)
    ax_a.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2)
    panel_label(ax_a, "a")

    activation_metrics = [
        ("Any shield", "shield_activation_fraction"),
        ("Material", "shield_material_activation_fraction"),
        ("Strong", "shield_strong_activation_fraction"),
    ]
    act_means = [data[col].mean() for _, col in activation_metrics]
    act_stds = [data[col].std(ddof=1) for _, col in activation_metrics]
    ax_b.bar(
        np.arange(len(activation_metrics)),
        act_means,
        yerr=act_stds,
        color=[GREY, GOLD, RED],
        capsize=3,
        width=0.58,
    )
    ax_b.set_xticks(np.arange(len(activation_metrics)))
    ax_b.set_xticklabels([label for label, _ in activation_metrics], rotation=25, ha="right")
    ax_b.set_ylim(0, 0.04)
    ax_b.set_ylabel("Activation fraction")
    ax_b.grid(axis="y", color=LIGHT, linewidth=0.5)
    ax_b.text(0.5, 0.72, "Shield activation = 0\nin this 16-seed dataset", transform=ax_b.transAxes, ha="center", va="center")
    panel_label(ax_b, "b")

    plot_summary = summary.copy()
    colors = [TEAL, BLUE, GOLD, ORANGE, PURPLE, GREEN]
    y = np.arange(len(plot_summary))[::-1]
    ax_c.barh(y, plot_summary["mean"], xerr=plot_summary["std"], color=colors, alpha=0.86, capsize=2)
    ax_c.set_yticks(y)
    ax_c.set_yticklabels(plot_summary["metric"])
    ax_c.set_xlim(0, 0.62)
    ax_c.set_xlabel("Fraction / normalized gap")
    ax_c.grid(axis="x", color=LIGHT, linewidth=0.5)
    for yi, value, spread in zip(y, plot_summary["mean"], plot_summary["std"]):
        label_x = min(float(value + spread + 0.02), 0.58)
        ax_c.text(label_x, yi, f"{value:.2f}", va="center", fontsize=6.5)
    panel_label(ax_c, "c")

    save_figure(fig, "fig1_shield_dependence_elite")


def fig2_cross_fidelity_elite(df: pd.DataFrame, seeds: list[int]) -> None:
    data = success_cross_fidelity(df, seeds)
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


def fig3_morphology_density(df: pd.DataFrame) -> None:
    data = simple_seed_rows(df)
    data.to_csv(SOURCE / "fig3_morphology_density_elite_source.csv", index=False)
    data.to_csv(SOURCE / "fig2_morphology_scatter_source.csv", index=False)

    success = data[data["success"]].copy()
    failure = data[~data["success"]].copy()

    fig, ax = plt.subplots(figsize=(3.6, 3.2), constrained_layout=True)
    ax.add_patch(
        Rectangle(
            (0, SUCCESS_MIDBAND_THRESHOLD * 100),
            SUCCESS_INFEASIBLE_THRESHOLD * 100,
            55 - SUCCESS_MIDBAND_THRESHOLD * 100,
            facecolor=TEAL,
            edgecolor="none",
            alpha=0.08,
            zorder=0,
        )
    )
    ax.axvline(SUCCESS_INFEASIBLE_THRESHOLD * 100, color=RED, linewidth=1.0, linestyle=(0, (4, 2)))
    ax.axhline(SUCCESS_MIDBAND_THRESHOLD * 100, color=BLUE, linewidth=1.0, linestyle=(0, (4, 2)))

    ax.scatter(
        failure["infeasible_action_dwell_fraction"] * 100,
        failure["soc_midband_dwell_fraction"] * 100,
        s=32,
        marker="x",
        linewidth=1.3,
        color=GREY,
        label=f"Failure (n={len(failure)})",
    )
    dense_failure = failure[
        (failure["infeasible_action_dwell_fraction"] > 0.95)
        & (failure["soc_midband_dwell_fraction"] < 0.01)
    ]
    if len(dense_failure) > 1:
        ax.text(
            98,
            3.2,
            f"{len(dense_failure)} overlapping\nfailures",
            ha="right",
            va="bottom",
            fontsize=5.8,
            color=GREY,
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.75),
        )
    scatter = ax.scatter(
        success["infeasible_action_dwell_fraction"] * 100,
        success["soc_midband_dwell_fraction"] * 100,
        s=58,
        marker="o",
        linewidth=0.7,
        edgecolor=BLACK,
        c=success["final_cumulative_objective_cost"] / 1e6,
        cmap="viridis_r",
        label=f"Success (n={len(success)})",
        zorder=3,
    )

    label_offsets = {
        42: (1.4, 1.4),
        44: (1.2, 1.0),
        81: (1.3, -1.8),
        83: (1.2, 1.2),
        84: (1.1, -1.7),
        93: (1.1, 1.1),
    }
    for _, row in success.iterrows():
        seed = int(row["seed"])
        dx, dy = label_offsets.get(seed, (1.0, 1.0))
        ax.text(
            row["infeasible_action_dwell_fraction"] * 100 + dx,
            row["soc_midband_dwell_fraction"] * 100 + dy,
            str(seed),
            fontsize=6.6,
            color=BLACK,
        )
    ax.text(
        0.97,
        0.97,
        f"Success: {len(success)}/16\nGates: mid-band >5%, infeasible <30%",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.3,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=LIGHT),
    )
    ax.set_xlim(-3, 105)
    ax.set_ylim(-2, 55)
    ax.set_xlabel("Infeasible-action dwell (%)")
    ax.set_ylabel("SOC mid-band dwell (%)")
    ax.grid(color=LIGHT, linewidth=0.45)
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(0.98, 0.83), handletextpad=0.4)
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.055, pad=0.02)
    cbar.set_label("Objective cost (M¥)")

    save_figure(fig, "fig3_morphology_density_elite")


def main() -> None:
    df = load_seed_summaries()
    seeds = success_seeds(df)
    fig1_shield_dependence_elite(df, seeds)
    fig2_cross_fidelity_elite(df, seeds)
    fig3_morphology_density(df)
    print(f"Generated data-driven elite figures from {len(simple_seed_rows(df))} seeds; successes={seeds}")


if __name__ == "__main__":
    main()
