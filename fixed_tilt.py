"""Create Figure (a) directly from the fixed-tilt summary CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.optimize import least_squares

from plotting import (
    apply_plot_style,
    load_curve_data,
    plt,
    save_figure,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data_and_figures"
FIT_GAMMA_MAX = 4.0
DENSE_POINTS = 800
DEFAULT_DATASETS = [
    (
        "shot=1000",
        DATA_DIR / "fixed_gamma_shot_1000.csv",
        "#1f77b4",
        "o",
        "-",
    ),
    (
        "shot=5000",
        DATA_DIR / "fixed_gamma_shot_5000.csv",
        "#ff7f0e",
        "s",
        "--",
    ),
    (
        "shot=10000",
        DATA_DIR / "fixed_gamma_shot_10000.csv",
        "#1b8a5a",
        "^",
        "-.",
    ),
]


def exp_quad_model(x, a, b, c, d):
    """Original shifted exponential-quadratic display model."""
    x = np.asarray(x, dtype=float)
    return d + (a + b * x) * np.exp(-c * x * x)


def exp_quad_first_derivative(x, a, b, c):
    """First derivative used by the original shape constraints."""
    x = np.asarray(x, dtype=float)
    return np.exp(-c * x * x) * (
        b - 2.0 * c * x * (a + b * x)
    )


def exp_quad_second_derivative(x, a, b, c):
    """Second derivative used by the original curvature constraints."""
    x = np.asarray(x, dtype=float)
    return np.exp(-c * x * x) * (
        -2.0 * c * a
        - 6.0 * b * c * x
        + 4.0 * c * c * x * x * (a + b * x)
    )


def reference_lorentzian(x, g, h):
    """Lorentzian component of the stored reference curves."""
    x = np.asarray(x, dtype=float)
    return (h * h) / ((x - g) ** 2 + h * h)


def reference_curve_value(x, row):
    """Evaluate one stored poly-tail reference curve."""
    return float(
        row["a"]
        + row["b"] * x
        + row.get("c", 0.0) / (x + 1.0)
        + row["d"] * reference_lorentzian(x, row["g"], row["h"])
    )


def load_reference_shapes(path: Path) -> dict[str, dict[str, float]]:
    """Load the original matching targets at the origin, tail, and peak."""
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    targets: dict[str, dict[str, float]] = {}
    for _, row in frame.iterrows():
        label = str(row["dataset"])
        targets[label] = {
            "y0": reference_curve_value(0.0, row),
            "y3": reference_curve_value(3.0, row),
            "y4": reference_curve_value(4.0, row),
            "peak_gamma": float(row["peak_gamma"]),
            "peak_value": float(row["peak_value"]),
        }
    return targets


def fit_single_dataset(
    label: str,
    gamma: np.ndarray,
    ratio: np.ndarray,
    sem: np.ndarray,
    reference_targets: dict[str, dict[str, float]],
) -> dict[str, object]:
    """Run the original robust multi-start curve-matching procedure."""
    reference = reference_targets.get(label)
    ratio_at_zero = float(ratio[0])
    tail_ratio = float(ratio[-1])
    data_peak_index = int(np.argmax(ratio))
    data_peak_gamma = float(gamma[data_peak_index])
    data_peak_ratio = float(ratio[data_peak_index])

    d_guess = np.clip(tail_ratio - 0.0025, 0.68, 0.74)
    a_guess = np.clip(ratio_at_zero - d_guess, 0.02, 0.08)
    b_guess = np.clip(
        max(
            0.01,
            (data_peak_ratio - ratio_at_zero)
            / max(data_peak_gamma, 0.25)
            + 0.02,
        ),
        0.004,
        0.10,
    )
    starts = [
        np.array([a_guess, b_guess, 0.10, d_guess], dtype=float),
        np.array(
            [a_guess * 1.05, b_guess * 1.25, 0.08, d_guess * 0.998],
            dtype=float,
        ),
        np.array(
            [
                a_guess * 0.95,
                b_guess * 0.90,
                0.14,
                min(0.74, d_guess + 0.002),
            ],
            dtype=float,
        ),
        np.array(
            [
                a_guess,
                min(0.12, b_guess * 1.45),
                0.06,
                max(0.68, d_guess - 0.003),
            ],
            dtype=float,
        ),
    ]
    bounds = ([0.015, 0.0, 0.01, 0.68], [0.10, 0.14, 0.50, 0.74])
    minimum_lift = 0.010 if label == "shot=5000" else 0.007
    safe_sem = np.maximum(np.asarray(sem, dtype=float), 1e-3)

    def residuals(parameters):
        a, b, c, d = parameters
        data_residuals = (
            exp_quad_model(gamma, a, b, c, d) - ratio
        ) / safe_sem

        constraint_gamma = np.linspace(0.0, FIT_GAMMA_MAX, 2001)
        constraint_values = exp_quad_model(
            constraint_gamma,
            a,
            b,
            c,
            d,
        )
        peak_index = int(np.argmax(constraint_values))
        peak_gamma = float(constraint_gamma[peak_index])
        peak_value = float(constraint_values[peak_index])
        lift = peak_value - float(exp_quad_model(0.0, a, b, c, d))

        penalties = [
            40.0 * max(0.0, 0.35 - peak_gamma),
            40.0 * max(0.0, peak_gamma - 0.60),
            18.0
            * max(
                0.0,
                -float(exp_quad_first_derivative(0.20, a, b, c)),
            ),
            18.0
            * max(
                0.0,
                -float(exp_quad_first_derivative(0.35, a, b, c)),
            ),
            18.0
            * max(
                0.0,
                float(exp_quad_first_derivative(0.95, a, b, c)),
            ),
            18.0
            * max(
                0.0,
                float(exp_quad_first_derivative(1.30, a, b, c)),
            ),
            10.0
            * max(
                0.0,
                -float(exp_quad_second_derivative(1.50, a, b, c)),
            ),
            10.0
            * max(
                0.0,
                -float(exp_quad_second_derivative(2.40, a, b, c)),
            ),
            10.0
            * max(
                0.0,
                -float(exp_quad_second_derivative(3.00, a, b, c)),
            ),
            22.0 * max(0.0, minimum_lift - lift),
        ]

        if reference is not None:
            penalties.extend(
                [
                    18.0
                    * (
                        float(exp_quad_model(0.0, a, b, c, d))
                        - reference["y0"]
                    ),
                    12.0
                    * (
                        float(exp_quad_model(3.0, a, b, c, d))
                        - reference["y3"]
                    ),
                    18.0
                    * (
                        float(exp_quad_model(4.0, a, b, c, d))
                        - reference["y4"]
                    ),
                    20.0 * (peak_gamma - reference["peak_gamma"]),
                    16.0 * (peak_value - reference["peak_value"]),
                ]
            )
        return np.concatenate(
            [data_residuals, np.asarray(penalties, dtype=float)]
        )

    best_result = None
    best_score = np.inf
    for start in starts:
        result = least_squares(
            residuals,
            x0=start,
            bounds=bounds,
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=40000,
        )
        score = float(np.sum(residuals(result.x) ** 2))
        if score < best_score:
            best_result = result
            best_score = score

    if best_result is None:
        raise RuntimeError(f"Curve fitting failed for {label}.")
    a, b, c, d = map(float, best_result.x)
    dense_gamma = np.linspace(0.0, FIT_GAMMA_MAX, DENSE_POINTS)
    dense_ratio = exp_quad_model(dense_gamma, a, b, c, d)
    peak_index = int(np.argmax(dense_ratio))
    return {
        "model_name": "exp_quad_shifted",
        "model_formula": r"$d + (a+bx)e^{-cx^2}$",
        "a": a,
        "b": b,
        "c": c,
        "d": d,
        "weighted_sse": best_score,
        "gamma_dense": dense_gamma,
        "ratio_dense": dense_ratio,
        "peak_gamma": float(dense_gamma[peak_index]),
        "peak_value": float(dense_ratio[peak_index]),
    }


def plot_fixed_tilt(
    datasets: list[tuple[str, Path, str, str, str]],
    output_png: Path,
    output_pdf: Path,
    reference_shape_path: Path,
    fit_summary_path: Path,
) -> None:
    """Fit and plot final approximation ratio against fixed tilt."""
    apply_plot_style()
    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.labelsize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 15,
            "axes.linewidth": 1.2,
            "lines.linewidth": 2.0,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "xtick.major.size": 7,
            "ytick.major.size": 7,
        }
    )
    figure, axis = plt.subplots(figsize=(8.2, 6.0))
    legend_handles = []
    fit_summary_rows = []
    reference_targets = load_reference_shapes(reference_shape_path)

    for label, csv_path, color, marker, line_style in datasets:
        gamma, ratio, sem = load_curve_data(csv_path)
        fit = fit_single_dataset(
            label,
            gamma,
            ratio,
            sem,
            reference_targets,
        )
        axis.plot(
            fit["gamma_dense"],
            fit["ratio_dense"],
            color=color,
            linewidth=1.9,
            linestyle=line_style,
            solid_capstyle="round",
            zorder=2,
        )
        axis.scatter(
            gamma,
            ratio,
            color=color,
            marker=marker,
            s=48,
            edgecolors="white",
            linewidths=0.8,
            zorder=4,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linestyle=line_style,
                linewidth=1.9,
                marker=marker,
                markersize=8,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.8,
                label=label,
            )
        )
        fit_summary_rows.append(
            {
                "dataset": label,
                "model_name": fit["model_name"],
                "model_formula": fit["model_formula"],
                "a": fit["a"],
                "b": fit["b"],
                "c": fit["c"],
                "d": fit["d"],
                "weighted_sse": fit["weighted_sse"],
                "peak_gamma": fit["peak_gamma"],
                "peak_value": fit["peak_value"],
            }
        )

    axis.set(
        xlabel=r"Tilt parameter $|\gamma|$",
        ylabel="Mean Final Ratio",
        xlim=(0.0, 4.0),
        ylim=(0.72, 0.78),
    )
    axis.set_facecolor("white")
    axis.grid(
        axis="y",
        color="#d9d9d9",
        linestyle=(0, (4, 3)),
        linewidth=1.0,
    )
    axis.grid(axis="x", visible=False)
    axis.spines["top"].set_visible(True)
    axis.spines["right"].set_visible(True)
    axis.spines["left"].set_linewidth(1.2)
    axis.spines["bottom"].set_linewidth(1.2)
    axis.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=True,
        fancybox=True,
        framealpha=0.96,
        facecolor="white",
        edgecolor="#c8c8c8",
        borderpad=0.6,
        labelspacing=0.7,
        handlelength=2.0,
    )
    save_figure(figure, output_png, output_pdf, dpi=600)
    fit_summary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fit_summary_rows).to_csv(fit_summary_path, index=False)


def build_parser() -> argparse.ArgumentParser:
    """Build the Figure (a) command-line interface."""
    parser = argparse.ArgumentParser(
        description="Plot Figure (a) from fixed-tilt CSV summaries."
    )
    parser.add_argument(
        "--shot-1000",
        type=Path,
        default=DEFAULT_DATASETS[0][1],
    )
    parser.add_argument(
        "--shot-5000",
        type=Path,
        default=DEFAULT_DATASETS[1][1],
    )
    parser.add_argument(
        "--shot-10000",
        type=Path,
        default=DEFAULT_DATASETS[2][1],
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=DATA_DIR / "fixed_plot_results.png",
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=DATA_DIR / "fixed_plot_results.pdf",
    )
    parser.add_argument(
        "--reference-shape-csv",
        type=Path,
        default=DATA_DIR / "curve_fitting_poly_tail_summary.csv",
    )
    parser.add_argument(
        "--fit-summary-csv",
        type=Path,
        default=DATA_DIR / "fixed_plot_fit_summary.csv",
    )
    return parser


def main() -> None:
    """Load command-line paths and generate Figure (a)."""
    arguments = build_parser().parse_args()
    datasets = [
        ("shot=1000", arguments.shot_1000, "#1f77b4", "o", "-"),
        ("shot=5000", arguments.shot_5000, "#ff7f0e", "s", "--"),
        ("shot=10000", arguments.shot_10000, "#1b8a5a", "^", "-."),
    ]
    plot_fixed_tilt(
        datasets,
        arguments.output_png,
        arguments.output_pdf,
        arguments.reference_shape_csv,
        arguments.fit_summary_csv,
    )


if __name__ == "__main__":
    main()
