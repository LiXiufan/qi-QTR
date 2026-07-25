from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mplconfig_codex"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


SCRIPT_DIR = Path(__file__).resolve().parent

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 20,
        "axes.labelsize": 20,
        "axes.titlesize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 16,
        "legend.frameon": False,
        "axes.linewidth": 1.4,
        "lines.linewidth": 2,
        "xtick.major.width": 1.6,
        "ytick.major.width": 1.6,
        "xtick.major.size": 10,
        "ytick.major.size": 10,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": False,
        "ytick.right": False,
        "figure.autolayout": True,
    }
)


METHOD_COLORS = {
    "Expectation": "#1f77b4",
    "Fixed QTL": "#ff7f0e",
    "Ascending QTL": "#2ca02c",
}


def build_plot(data_path: Path, output_png: Path, output_pdf: Path) -> None:
    transfer_seed_df = pd.read_csv(data_path)
    stacked_optimal_mass_df = (
        transfer_seed_df.groupby(["n", "p", "objective"])
        .agg(mean_optimal_mass=("final_optimal_mass", "mean"))
        .reset_index()
        .sort_values(["n", "p", "objective"])
    )

    n_values = sorted(stacked_optimal_mass_df["n"].unique())
    p_values = sorted(stacked_optimal_mass_df["p"].unique())
    n_gap = 0.92
    p_offsets = np.array([-0.16, 0.0, 0.16])
    bar_width = 0.15
    n_centers = np.arange(len(n_values), dtype=float) * n_gap

    plot_y_min = max(0.0, float(stacked_optimal_mass_df["mean_optimal_mass"].min()) - 0.02)
    plot_y_max = min(1.0, float(stacked_optimal_mass_df["mean_optimal_mass"].max()) + 0.05)

    fig, ax = plt.subplots(figsize=(7, 6.0))

    for n_center, n in zip(n_centers, n_values):
        for p_offset, p in zip(p_offsets, p_values):
            x_pos = n_center + p_offset
            subset = (
                stacked_optimal_mass_df[(stacked_optimal_mass_df["n"] == n) & (stacked_optimal_mass_df["p"] == p)]
                .set_index("objective")
            )

            method_values = {
                "Expectation": float(subset.loc["Expectation", "mean_optimal_mass"]),
                "Fixed QTL": float(subset.loc["Fixed QTL", "mean_optimal_mass"]),
                "Ascending QTL": float(subset.loc["Ascending QTL", "mean_optimal_mass"]),
            }
            sorted_methods = sorted(method_values.items(), key=lambda item: item[1])

            previous_value = 0.0
            for method_name, method_value in sorted_methods:
                segment_height = method_value - previous_value
                if segment_height <= 1e-12:
                    previous_value = method_value
                    continue
                ax.bar(
                    x_pos,
                    segment_height,
                    width=bar_width,
                    bottom=previous_value,
                    color=METHOD_COLORS[method_name],
                    edgecolor="white",
                    linewidth=0.8,
                    alpha=0.94,
                )
                previous_value = method_value

            ax.scatter(
                [x_pos, x_pos, x_pos],
                [
                    method_values["Expectation"],
                    method_values["Fixed QTL"],
                    method_values["Ascending QTL"],
                ],
                color=[
                    METHOD_COLORS["Expectation"],
                    METHOD_COLORS["Fixed QTL"],
                    METHOD_COLORS["Ascending QTL"],
                ],
                s=28,
                edgecolor="white",
                linewidth=0.8,
                zorder=5,
            )

            ax.text(
                x_pos,
                -0.015,
                f"p={p}",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=14,
            )

    legend_handles = [
        Patch(facecolor=METHOD_COLORS["Expectation"], edgecolor="white", label="Expectation"),
        Patch(facecolor=METHOD_COLORS["Fixed QTL"], edgecolor="white", label=r"Fixed-$\gamma$"),
        Patch(facecolor=METHOD_COLORS["Ascending QTL"], edgecolor="white", label=r"Ascending-$\gamma$"),
    ]

    for left, right in zip(n_centers[:-1], n_centers[1:]):
        ax.axvline((left + right) / 2.0, color="#d7d7d7", linewidth=1.0, alpha=0.8)

    ax.set_xticks(n_centers)
    ax.set_xticklabels([f"n={n}" for n in n_values], fontsize=18)
    ax.tick_params(axis="x", length=0, pad=28)
    ax.set_ylabel("Mean Optimal Mass", fontsize=16)
    ax.set_ylim(plot_y_min, plot_y_max)
    ax.grid(axis="y", alpha=0.18)
    ax.legend(handles=legend_handles, frameon=False, loc="upper right", fontsize=14)

    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.4)

    fig.tight_layout(rect=[0.0, 0.10, 1.0, 0.95])
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot the MaxCut comparison panel for mean optimal mass.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=SCRIPT_DIR / "data_and_figures" / "maxcut_compare_avg_shot5000.csv",
    )
    parser.add_argument("--output-png", type=Path, default=SCRIPT_DIR / "maxcut_mean_optimal_mass_plot.png")
    parser.add_argument("--output-pdf", type=Path, default=SCRIPT_DIR / "maxcut_mean_optimal_mass_plot.pdf")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    build_plot(args.data_path.resolve(), args.output_png.resolve(), args.output_pdf.resolve())


if __name__ == "__main__":
    main()
