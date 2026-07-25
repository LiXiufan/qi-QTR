"""Create Figure (c) directly from the scale-benchmark CSV file."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from plotting import (
    apply_plot_style,
    plt,
    require_columns,
    save_figure,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data_and_figures"
METHOD_COLORS = {
    "Expectation": "#1f77b4",
    "Fixed QTL": "#ff7f0e",
    "Ascending QTL": "#2ca02c",
}


def standard_error(values: pd.Series | np.ndarray) -> float:
    """Calculate graph-level sample SEM without experiment dependencies."""
    array = np.asarray(values, dtype=float)
    if len(array) <= 1:
        return 0.0
    return float(array.std(ddof=1) / np.sqrt(len(array)))


def aggregate_optimal_mass(data_path: Path) -> pd.DataFrame:
    """Return mean and SEM optimal mass for every size/depth/method."""
    data_path = Path(data_path)
    frame = pd.read_csv(data_path)
    if {"mean_optimal_mass", "sem_optimal_mass"}.issubset(frame.columns):
        require_columns(
            frame,
            {"n", "p", "objective"},
            data_path,
        )
        return frame[
            [
                "n",
                "p",
                "objective",
                "mean_optimal_mass",
                "sem_optimal_mass",
            ]
        ].copy()

    require_columns(
        frame,
        {"n", "p", "objective", "final_optimal_mass"},
        data_path,
    )
    return (
        frame.groupby(["n", "p", "objective"])
        .agg(
            mean_optimal_mass=("final_optimal_mass", "mean"),
            sem_optimal_mass=("final_optimal_mass", standard_error),
        )
        .reset_index()
    )


def plot_scale_benchmark(
    data_path: Path,
    output_png: Path,
    output_pdf: Path,
) -> None:
    """Plot the original layered optimal-mass benchmark presentation."""
    apply_plot_style()
    plt.rcParams.update(
        {
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
        }
    )
    summary = aggregate_optimal_mass(data_path)
    n_values = sorted(summary["n"].unique())
    depth_values = sorted(summary["p"].unique())
    if len(depth_values) != 3:
        raise ValueError(
            "The original Figure (c) style requires exactly three depths."
        )
    n_gap = 0.92
    depth_offsets = np.array([-0.16, 0.0, 0.16])
    bar_width = 0.15
    n_centers = np.arange(len(n_values), dtype=float) * n_gap
    y_minimum = max(
        0.0,
        float(summary["mean_optimal_mass"].min()) - 0.02,
    )
    y_maximum = min(
        1.0,
        float(summary["mean_optimal_mass"].max()) + 0.05,
    )

    figure, axis = plt.subplots(figsize=(7.0, 6.0))
    for n_center, n_value in zip(n_centers, n_values):
        for depth_offset, depth in zip(depth_offsets, depth_values):
            x_position = n_center + depth_offset
            subset = summary[
                (summary["n"] == n_value) & (summary["p"] == depth)
            ].set_index("objective")
            method_values = {
                method: float(
                    subset.loc[method, "mean_optimal_mass"]
                )
                for method in METHOD_COLORS
            }

            previous_value = 0.0
            for method, method_value in sorted(
                method_values.items(),
                key=lambda item: item[1],
            ):
                segment_height = method_value - previous_value
                if segment_height > 1e-12:
                    axis.bar(
                        x_position,
                        segment_height,
                        width=bar_width,
                        bottom=previous_value,
                        color=METHOD_COLORS[method],
                        edgecolor="white",
                        linewidth=0.8,
                        alpha=0.94,
                    )
                previous_value = method_value

            axis.scatter(
                [x_position] * 3,
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
            axis.text(
                x_position,
                -0.015,
                f"p={depth}",
                transform=axis.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=14,
            )

    legend_handles = [
        Patch(
            facecolor=METHOD_COLORS["Expectation"],
            edgecolor="white",
            label="Expectation",
        ),
        Patch(
            facecolor=METHOD_COLORS["Fixed QTL"],
            edgecolor="white",
            label=r"Fixed-$\gamma$",
        ),
        Patch(
            facecolor=METHOD_COLORS["Ascending QTL"],
            edgecolor="white",
            label=r"Ascending-$\gamma$",
        ),
    ]
    for left, right in zip(n_centers[:-1], n_centers[1:]):
        axis.axvline(
            (left + right) / 2.0,
            color="#d7d7d7",
            linewidth=1.0,
            alpha=0.8,
        )

    axis.set_xticks(n_centers)
    axis.set_xticklabels([f"n={n_value}" for n_value in n_values], fontsize=18)
    axis.tick_params(axis="x", length=0, pad=28)
    axis.set_ylabel("Mean Optimal Mass", fontsize=16)
    axis.set_ylim(y_minimum, y_maximum)
    axis.grid(axis="y", alpha=0.18)
    axis.legend(
        handles=legend_handles,
        frameon=False,
        loc="upper right",
        fontsize=14,
    )
    for spine in axis.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.4)

    figure.tight_layout(rect=[0.0, 0.10, 1.0, 0.95])
    save_figure(figure, output_png, output_pdf)


def build_parser() -> argparse.ArgumentParser:
    """Build the Figure (c) command-line interface."""
    parser = argparse.ArgumentParser(
        description="Plot Figure (c) from a scale-benchmark CSV."
    )
    parser.add_argument(
        "--data-csv",
        type=Path,
        default=DATA_DIR / "maxcut_compare_avg_shot5000.csv",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=DATA_DIR / "maxcut_mean_optimal_mass_plot.png",
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=DATA_DIR / "maxcut_mean_optimal_mass_plot.pdf",
    )
    return parser


def main() -> None:
    """Load command-line paths and generate Figure (c)."""
    arguments = build_parser().parse_args()
    plot_scale_benchmark(
        arguments.data_csv,
        arguments.output_png,
        arguments.output_pdf,
    )


if __name__ == "__main__":
    main()
