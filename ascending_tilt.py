"""Create Figure (b) with stable data-driven cubic curve fits."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.optimize import least_squares

from plotting import (
    apply_plot_style,
    plt,
    require_columns,
    save_figure,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data_and_figures"
FIT_GAMMA_MAX = 4.0
DENSE_POINTS = 800


@dataclass
class CurveResult:
    """Raw points, fitted curve, uncertainty, and model parameters."""

    label: str
    model_name: str
    gamma_data: np.ndarray
    ratio_data: np.ndarray
    sem_data: np.ndarray
    gamma_dense: np.ndarray
    ratio_dense: np.ndarray
    params: dict[str, float | str]


def cubic_model(x, a0, a1, a2, a3):
    """Evaluate a cubic polynomial in increasing coefficient order."""
    x = np.asarray(x, dtype=float)
    return a0 + a1 * x + a2 * x**2 + a3 * x**3


def load_final_ratio_data(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the final-ratio series used by Figure (b)."""
    path = Path(path)
    frame = pd.read_csv(path)
    require_columns(
        frame,
        {"gamma_plot", "mean_final_ratio", "sem_final_ratio"},
        path,
    )
    frame = (
        frame.dropna(
            subset=[
                "gamma_plot",
                "mean_final_ratio",
                "sem_final_ratio",
            ]
        )
        .loc[lambda data: data["gamma_plot"] <= FIT_GAMMA_MAX]
        .sort_values("gamma_plot")
        .drop_duplicates("gamma_plot")
    )
    if len(frame) < 4:
        raise ValueError(
            f"{path} requires at least four points for a cubic fit."
        )
    return (
        frame["gamma_plot"].to_numpy(dtype=float),
        frame["mean_final_ratio"].to_numpy(dtype=float),
        np.maximum(
            frame["sem_final_ratio"].to_numpy(dtype=float),
            1e-3,
        ),
    )


def fit_weighted_cubic(
    gamma: np.ndarray,
    ratio: np.ndarray,
    sem: np.ndarray,
) -> dict[str, object]:
    """Fit a robust cubic using inverse-SEM normalized residuals."""
    initial_coefficients = np.polynomial.polynomial.polyfit(
        gamma,
        ratio,
        deg=3,
        w=1.0 / sem,
    )

    def residuals(coefficients):
        return (
            cubic_model(gamma, *coefficients) - ratio
        ) / sem

    fit = least_squares(
        residuals,
        x0=initial_coefficients,
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=10000,
    )
    if not fit.success:
        raise RuntimeError(f"Cubic fitting failed: {fit.message}")

    a0, a1, a2, a3 = map(float, fit.x)
    dense_gamma = np.linspace(0.0, FIT_GAMMA_MAX, DENSE_POINTS)
    dense_ratio = cubic_model(
        dense_gamma,
        a0,
        a1,
        a2,
        a3,
    )
    peak_index = int(np.argmax(dense_ratio))
    return {
        "model_formula": (
            r"$a_0+a_1|\gamma|+a_2|\gamma|^2+a_3|\gamma|^3$"
        ),
        "a0": a0,
        "a1": a1,
        "a2": a2,
        "a3": a3,
        "weighted_sse": float(np.sum(residuals(fit.x) ** 2)),
        "peak_gamma": float(dense_gamma[peak_index]),
        "peak_value": float(dense_ratio[peak_index]),
        "tail_value": float(dense_ratio[-1]),
        "gamma_dense": dense_gamma,
        "ratio_dense": dense_ratio,
    }


def fit_curve_result(label: str, data_path: Path) -> CurveResult:
    """Load and fit one Figure (b) final-ratio series."""
    gamma, ratio, sem = load_final_ratio_data(data_path)
    fit = fit_weighted_cubic(gamma, ratio, sem)
    return CurveResult(
        label=label,
        model_name="robust_sem_weighted_cubic",
        gamma_data=gamma,
        ratio_data=ratio,
        sem_data=sem,
        gamma_dense=fit["gamma_dense"],
        ratio_dense=fit["ratio_dense"],
        params={
            key: value
            for key, value in fit.items()
            if key not in {"gamma_dense", "ratio_dense"}
        },
    )


def calculate_sem(result: CurveResult) -> float:
    """Calculate the constant-width RMS SEM display band."""
    return float(np.sqrt(np.mean(np.square(result.sem_data))))


def fit_summary_rows(
    fixed_result: CurveResult,
    ascending_result: CurveResult,
) -> list[dict[str, float | str]]:
    """Create one fit-summary row for each curve."""
    rows = []
    for dataset, result in [
        ("fixed", fixed_result),
        ("ascending", ascending_result),
    ]:
        row: dict[str, float | str] = {
            "dataset": dataset,
            "model": result.model_name,
        }
        row.update(result.params)
        rows.append(row)
    return rows


def make_plot(
    fixed_result: CurveResult,
    ascending_result: CurveResult,
    output_png: Path,
    output_pdf: Path,
) -> None:
    """Render the fitted curves with the established Figure (b) style."""
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
    figure, axis = plt.subplots(figsize=(8.0, 6.0))
    fixed_sem = calculate_sem(fixed_result)
    ascending_sem = calculate_sem(ascending_result)

    axis.fill_between(
        fixed_result.gamma_dense,
        fixed_result.ratio_dense - fixed_sem,
        fixed_result.ratio_dense + fixed_sem,
        color="#1f4e79",
        alpha=0.08,
        zorder=1,
    )
    axis.fill_between(
        ascending_result.gamma_dense,
        ascending_result.ratio_dense - ascending_sem,
        ascending_result.ratio_dense + ascending_sem,
        color="#EF7C00",
        alpha=0.08,
        zorder=1,
    )
    axis.plot(
        fixed_result.gamma_dense,
        fixed_result.ratio_dense,
        color="#1f4e79",
        linestyle=(0, (4, 3)),
        linewidth=2.2,
        zorder=3,
    )
    axis.plot(
        ascending_result.gamma_dense,
        ascending_result.ratio_dense,
        color="#EF7C00",
        linestyle="-",
        linewidth=2.2,
        zorder=3,
    )
    axis.scatter(
        fixed_result.gamma_data,
        fixed_result.ratio_data,
        marker="o",
        color="#1f4e79",
        s=70,
        edgecolors="white",
        linewidths=0.5,
        zorder=4,
    )
    axis.scatter(
        ascending_result.gamma_data,
        ascending_result.ratio_data,
        marker="s",
        color="#EF7C00",
        s=70,
        edgecolors="white",
        linewidths=0.5,
        zorder=4,
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color="#1f4e79",
            linestyle=(0, (4, 3)),
            linewidth=2.2,
            marker="o",
            markersize=8,
            markerfacecolor="#1f4e79",
            markeredgewidth=0.0,
            label=r"Fixed-$\gamma$",
        ),
        Line2D(
            [0],
            [0],
            color="#EF7C00",
            linestyle="-",
            linewidth=2.2,
            marker="s",
            markersize=8,
            markerfacecolor="#EF7C00",
            markeredgewidth=0.0,
            label=r"Ascending-$\gamma$",
        ),
    ]
    axis.set(
        xlabel=r"Tilt parameter $|\gamma|$",
        ylabel="Mean Final Ratio",
        xlim=(0.0, 4.0),
        ylim=(0.65, 0.865),
    )
    axis.set_yticks(np.arange(0.65, 0.851, 0.05))
    axis.grid(
        axis="y",
        color="#cfcfcf",
        linestyle=(0, (3, 3)),
        linewidth=1.0,
    )
    axis.grid(axis="x", visible=False)
    axis.spines["top"].set_visible(True)
    axis.spines["right"].set_visible(True)
    for spine in axis.spines.values():
        spine.set_linewidth(1.2)
    axis.legend(handles=legend_handles, loc="upper right")
    save_figure(figure, output_png, output_pdf, dpi=600)


def plot_ascending_tilt(
    fixed_csv: Path,
    ascending_csv: Path,
    output_png: Path,
    output_pdf: Path,
    fit_summary_csv: Path,
) -> None:
    """Fit both data series, render Figure (b), and save parameters."""
    fixed_result = fit_curve_result(r"Fixed $\gamma$", fixed_csv)
    ascending_result = fit_curve_result(
        r"Ascending $\gamma$",
        ascending_csv,
    )
    make_plot(
        fixed_result,
        ascending_result,
        output_png,
        output_pdf,
    )
    fit_summary_csv = Path(fit_summary_csv)
    fit_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        fit_summary_rows(fixed_result, ascending_result)
    ).to_csv(fit_summary_csv, index=False)


def build_parser() -> argparse.ArgumentParser:
    """Build the Figure (b) command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Fit robust cubic curves and plot Figure (b)."
        )
    )
    parser.add_argument(
        "--fixed-csv",
        type=Path,
        default=DATA_DIR / "fixed_gamma_shot_5000.csv",
    )
    parser.add_argument(
        "--ascending-csv",
        type=Path,
        default=(
            DATA_DIR
            / "schedule_gamma_restart_group(shots5000)all.csv"
        ),
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=DATA_DIR / "schedule_gamma_expquad_fit.png",
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=DATA_DIR / "schedule_gamma_expquad_fit.pdf",
    )
    parser.add_argument(
        "--fit-summary-csv",
        type=Path,
        default=DATA_DIR / "schedule_gamma_expquad_fit_summary.csv",
    )
    return parser


def main() -> None:
    """Load command-line paths and regenerate Figure (b)."""
    arguments = build_parser().parse_args()
    plot_ascending_tilt(
        arguments.fixed_csv,
        arguments.ascending_csv,
        arguments.output_png,
        arguments.output_pdf,
        arguments.fit_summary_csv,
    )


if __name__ == "__main__":
    main()
