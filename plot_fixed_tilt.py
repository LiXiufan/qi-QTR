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
from matplotlib.lines import Line2D
from scipy.optimize import least_squares


SCRIPT_DIR = Path(__file__).resolve().parent
FIT_GAMMA_MAX = 4.0
DENSE_POINTS = 800

DATASETS = [
    ("shot=1000", "fixed_gamma_shot_1000.csv", "#1f77b4"),
    ("shot=5000", "fixed_gamma_shot_5000.csv", "#ff7f0e"),
    ("shot=10000", "fixed_gamma_shot_10000.csv", "#1b8a5a"),
]
LINE_STYLES = ["-", "--", "-."]
MARKERS = ["o", "s", "^"]

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
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
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": False,
        "ytick.right": False,
        "figure.autolayout": True,
    }
)


def exp_quad_model(x, a, b, c, d):
    x = np.asarray(x, dtype=float)
    return d + (a + b * x) * np.exp(-c * x * x)


def exp_quad_first_derivative(x, a, b, c):
    x = np.asarray(x, dtype=float)
    return np.exp(-c * x * x) * (b - 2.0 * c * x * (a + b * x))


def exp_quad_second_derivative(x, a, b, c):
    x = np.asarray(x, dtype=float)
    return np.exp(-c * x * x) * (
        -2.0 * c * a - 6.0 * b * c * x + 4.0 * c * c * x * x * (a + b * x)
    )


def reference_lorentzian(x, g, h):
    x = np.asarray(x, dtype=float)
    return (h * h) / ((x - g) ** 2 + h * h)


def reference_curve_value(x, row):
    return float(
        row["a"]
        + row["b"] * x
        + row.get("c", 0.0) / (x + 1.0)
        + row["d"] * reference_lorentzian(x, row["g"], row["h"])
    )


def load_reference_shapes(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    targets = {}
    for _, row in df.iterrows():
        label = str(row["dataset"])
        targets[label] = {
            "y0": reference_curve_value(0.0, row),
            "y3": reference_curve_value(3.0, row),
            "y4": reference_curve_value(4.0, row),
            "peak_gamma": float(row["peak_gamma"]),
            "peak_value": float(row["peak_value"]),
        }
    return targets


def load_datasets(base_dir: Path):
    loaded = []
    for label, filename, color in DATASETS:
        path = base_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")
        df = pd.read_csv(path)
        df = df.dropna(subset=["gamma_plot", "mean_final_ratio", "sem_final_ratio"])
        df = df[df["gamma_plot"] <= FIT_GAMMA_MAX].copy()
        df = df.sort_values(by="gamma_plot")
        loaded.append(
            {
                "label": label,
                "color": color,
                "gamma": df["gamma_plot"].to_numpy(dtype=float),
                "ratio": df["mean_final_ratio"].to_numpy(dtype=float),
                "sem": np.maximum(df["sem_final_ratio"].to_numpy(dtype=float), 1e-3),
            }
        )
    return loaded


def fit_single_dataset(label, x, y, sem, reference_targets):
    ref = reference_targets.get(label)
    y0 = float(y[0])
    tail = float(y[-1])
    peak_idx = int(np.argmax(y))
    x_peak = float(x[peak_idx])
    y_peak = float(y[peak_idx])

    d_guess = np.clip(tail - 0.0025, 0.68, 0.74)
    a_guess = np.clip(y0 - d_guess, 0.02, 0.08)
    b_guess = np.clip(max(0.01, (y_peak - y0) / max(x_peak, 0.25) + 0.02), 0.004, 0.10)
    c_guess = 0.10

    starts = [
        np.array([a_guess, b_guess, c_guess, d_guess], dtype=float),
        np.array([a_guess * 1.05, b_guess * 1.25, 0.08, d_guess * 0.998], dtype=float),
        np.array([a_guess * 0.95, b_guess * 0.90, 0.14, min(0.74, d_guess + 0.002)], dtype=float),
        np.array([a_guess, min(0.12, b_guess * 1.45), 0.06, max(0.68, d_guess - 0.003)], dtype=float),
    ]
    bounds = ([0.015, 0.0, 0.01, 0.68], [0.10, 0.14, 0.50, 0.74])
    min_lift = 0.010 if label == "shot=5000" else 0.007

    def residuals(params):
        a, b, c, d = params
        data_res = (exp_quad_model(x, a, b, c, d) - y) / sem

        gamma_dense = np.linspace(0.0, FIT_GAMMA_MAX, 2001)
        values_dense = exp_quad_model(gamma_dense, a, b, c, d)
        peak_idx_dense = int(np.argmax(values_dense))
        peak_gamma = float(gamma_dense[peak_idx_dense])
        peak_value = float(values_dense[peak_idx_dense])
        lift = peak_value - float(exp_quad_model(0.0, a, b, c, d))

        penalty_terms = [
            40.0 * max(0.0, 0.35 - peak_gamma),
            40.0 * max(0.0, peak_gamma - 0.60),
            18.0 * max(0.0, -float(exp_quad_first_derivative(0.20, a, b, c))),
            18.0 * max(0.0, -float(exp_quad_first_derivative(0.35, a, b, c))),
            18.0 * max(0.0, float(exp_quad_first_derivative(0.95, a, b, c))),
            18.0 * max(0.0, float(exp_quad_first_derivative(1.30, a, b, c))),
            10.0 * max(0.0, -float(exp_quad_second_derivative(1.50, a, b, c))),
            10.0 * max(0.0, -float(exp_quad_second_derivative(2.40, a, b, c))),
            10.0 * max(0.0, -float(exp_quad_second_derivative(3.00, a, b, c))),
            22.0 * max(0.0, min_lift - lift),
        ]

        if ref is not None:
            penalty_terms.extend(
                [
                    18.0 * (float(exp_quad_model(0.0, a, b, c, d)) - ref["y0"]),
                    12.0 * (float(exp_quad_model(3.0, a, b, c, d)) - ref["y3"]),
                    18.0 * (float(exp_quad_model(4.0, a, b, c, d)) - ref["y4"]),
                    20.0 * (peak_gamma - ref["peak_gamma"]),
                    16.0 * (peak_value - ref["peak_value"]),
                ]
            )

        return np.concatenate([data_res, np.asarray(penalty_terms, dtype=float)])

    best = None
    best_score = np.inf
    for x0 in starts:
        res = least_squares(
            residuals,
            x0=x0,
            bounds=bounds,
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=40000,
        )
        score = float(np.sum(residuals(res.x) ** 2))
        if score < best_score:
            best = res
            best_score = score

    a, b, c, d = map(float, best.x)
    gamma_dense = np.linspace(0.0, FIT_GAMMA_MAX, DENSE_POINTS)
    ratio_dense = exp_quad_model(gamma_dense, a, b, c, d)
    peak_idx_dense = int(np.argmax(ratio_dense))
    peak_gamma = float(gamma_dense[peak_idx_dense])
    peak_value = float(ratio_dense[peak_idx_dense])
    return {
        "model_name": "exp_quad_shifted",
        "model_formula": r"$d + (a+bx)e^{-cx^2}$",
        "a": a,
        "b": b,
        "c": c,
        "d": d,
        "weighted_sse": best_score,
        "gamma_dense": gamma_dense,
        "ratio_dense": ratio_dense,
        "peak_gamma": peak_gamma,
        "peak_value": peak_value,
    }


def plot_fits(base_dir: Path, output_png: Path, output_pdf: Path, summary_path: Path, reference_shape_path: Path) -> None:
    datasets = load_datasets(base_dir)
    reference_targets = load_reference_shapes(reference_shape_path)
    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    summary_rows = []
    legend_handles = []

    for idx, dataset in enumerate(datasets):
        fit = fit_single_dataset(dataset["label"], dataset["gamma"], dataset["ratio"], dataset["sem"], reference_targets)
        line_style = LINE_STYLES[idx % len(LINE_STYLES)]
        marker = MARKERS[idx % len(MARKERS)]

        ax.plot(
            fit["gamma_dense"],
            fit["ratio_dense"],
            color=dataset["color"],
            linewidth=1.9,
            linestyle=line_style,
            solid_capstyle="round",
            zorder=2,
        )
        ax.scatter(
            dataset["gamma"],
            dataset["ratio"],
            color=dataset["color"],
            s=48,
            marker=marker,
            edgecolors="white",
            linewidths=0.8,
            zorder=4,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=dataset["color"],
                linestyle=line_style,
                linewidth=1.9,
                marker=marker,
                markersize=8,
                markerfacecolor=dataset["color"],
                markeredgecolor="white",
                markeredgewidth=0.8,
                label=dataset["label"],
            )
        )
        summary_rows.append(
            {
                "dataset": dataset["label"],
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

    ax.set_xlabel(r"Tilt parameter $|\gamma|$")
    ax.set_ylabel("Mean Final Ratio")
    ax.set_xlim(left=0, right=FIT_GAMMA_MAX)
    ax.set_ylim(bottom=0.72, top=0.78)
    ax.set_facecolor("white")
    ax.grid(axis="y", color="#d9d9d9", linestyle=(0, (4, 3)), linewidth=1.0)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.legend(
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

    fig.savefig(output_png, dpi=600, format="png")
    fig.savefig(output_pdf, format="pdf")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot the fixed-tilt sweep used in the NeurIPS submission.")
    parser.add_argument("--base-dir", type=Path, default=SCRIPT_DIR / "data_and_figures")
    parser.add_argument("--output-png", type=Path, default=SCRIPT_DIR / "fixed_plot_results.png")
    parser.add_argument("--output-pdf", type=Path, default=SCRIPT_DIR / "fixed_plot_results.pdf")
    parser.add_argument("--summary-path", type=Path, default=SCRIPT_DIR / "fixed_plot_fit_summary.csv")
    parser.add_argument(
        "--reference-shape-path",
        type=Path,
        default=SCRIPT_DIR / "data_and_figures" / "curve_fitting_poly_tail_summary.csv",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    plot_fits(
        base_dir=args.base_dir.resolve(),
        output_png=args.output_png.resolve(),
        output_pdf=args.output_pdf.resolve(),
        summary_path=args.summary_path.resolve(),
        reference_shape_path=args.reference_shape_path.resolve(),
    )


if __name__ == "__main__":
    main()
