"""Create Figure (b) with stable asymmetric rise-and-decay curve fits."""

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
DATA_DIR = SCRIPT_DIR / "figure_b_latest"
DENSE_POINTS = 800
PLOT_GAMMA_MAX = 36.0


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


def asymmetric_rise_decay_model(
    x,
    peak_gamma,
    peak_value,
    left_curvature,
    decay_amplitude,
    decay_scale,
    tail_slope,
):
    """Evaluate a rising quadratic followed by monotone nonlinear decay."""
    x = np.asarray(x, dtype=float)
    displacement = x - peak_gamma
    rising = peak_value - left_curvature * displacement**2
    decay_distance = np.maximum(displacement, 0.0)
    falling = (
        peak_value
        - decay_amplitude
        * (1.0 - np.exp(-decay_distance / decay_scale))
        - tail_slope * decay_distance
    )
    return np.where(displacement <= 0.0, rising, falling)


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
        .loc[
            lambda data: data["gamma_plot"].between(
                0.0,
                PLOT_GAMMA_MAX,
            )
        ]
        .sort_values("gamma_plot")
        .drop_duplicates("gamma_plot")
    )
    if len(frame) < 3:
        raise ValueError(
            f"{path} requires at least three points for a curve fit."
        )
    return (
        frame["gamma_plot"].to_numpy(dtype=float),
        frame["mean_final_ratio"].to_numpy(dtype=float),
        np.maximum(
            frame["sem_final_ratio"].to_numpy(dtype=float),
            1e-3,
        ),
    )


def fit_asymmetric_rise_decay(
    gamma: np.ndarray,
    ratio: np.ndarray,
    sem: np.ndarray,
) -> dict[str, object]:
    """Fit a peak-anchored, monotone-decay visual matching function."""
    peak_index = int(np.argmax(ratio))
    if peak_index == 0 or peak_index == len(gamma) - 1:
        raise ValueError(
            "The asymmetric rise-and-decay fit requires an interior peak."
        )

    peak_gamma = float(gamma[peak_index])
    peak_value = float(ratio[peak_index])
    baseline_value = float(ratio[0])
    left_span = max(
        peak_gamma - float(gamma[0]),
        np.finfo(float).eps,
    )
    left_curvature = max(
        (peak_value - baseline_value) / left_span**2,
        0.0,
    )
    right_span = float(gamma[-1] - peak_gamma)
    total_drop = max(peak_value - float(ratio[-1]), 1e-6)

    def residuals(parameters):
        decay_amplitude, decay_scale, tail_slope = parameters
        return (
            asymmetric_rise_decay_model(
                gamma,
                peak_gamma,
                peak_value,
                left_curvature,
                decay_amplitude,
                decay_scale,
                tail_slope,
            )
            - ratio
        ) / sem

    upper_scale = max(10.0 * right_span, 1.0)
    best_fit = None
    for decay_scale in [
        0.5,
        1.0,
        2.0,
        4.0,
        8.0,
        16.0,
        32.0,
        64.0,
    ]:
        if decay_scale >= upper_scale:
            continue
        saturation = max(
            1.0 - np.exp(-right_span / decay_scale),
            1e-6,
        )
        for tail_fraction in [0.0, 0.25, 0.5, 0.75, 1.0]:
            tail_slope = tail_fraction * total_drop / right_span
            remaining_drop = max(
                total_drop - tail_slope * right_span,
                1e-6,
            )
            decay_amplitude = remaining_drop / saturation
            candidate = least_squares(
                residuals,
                x0=[
                    min(decay_amplitude, 0.999),
                    decay_scale,
                    tail_slope,
                ],
                bounds=(
                    [0.0, 0.05, 0.0],
                    [1.0, upper_scale, 0.1],
                ),
                loss="soft_l1",
                f_scale=1.0,
                max_nfev=10000,
            )
            if (
                candidate.success
                and (
                    best_fit is None
                    or candidate.cost < best_fit.cost
                )
            ):
                best_fit = candidate
    if best_fit is None:
        raise RuntimeError("Asymmetric rise-and-decay fitting failed.")

    decay_amplitude, decay_scale, tail_slope = map(
        float,
        best_fit.x,
    )
    dense_gamma = np.linspace(
        float(np.min(gamma)),
        float(np.max(gamma)),
        DENSE_POINTS,
    )
    dense_ratio = asymmetric_rise_decay_model(
        dense_gamma,
        peak_gamma,
        peak_value,
        left_curvature,
        decay_amplitude,
        decay_scale,
        tail_slope,
    )
    return {
        "model_formula": (
            r"$R^\star-c_-(|\gamma|-\gamma^\star)^2$ "
            r"before the peak; "
            r"$R^\star-A[1-e^{-(|\gamma|-\gamma^\star)/\tau}]"
            r"-m(|\gamma|-\gamma^\star)$ after the peak"
        ),
        "peak_gamma": peak_gamma,
        "peak_value": peak_value,
        "baseline_value": baseline_value,
        "left_curvature": left_curvature,
        "decay_amplitude": decay_amplitude,
        "decay_scale": decay_scale,
        "tail_slope": tail_slope,
        "weighted_sse": float(
            np.sum(residuals(best_fit.x) ** 2)
        ),
        "tail_value": float(dense_ratio[-1]),
        "gamma_dense": dense_gamma,
        "ratio_dense": dense_ratio,
    }


def fit_curve_result(label: str, data_path: Path) -> CurveResult:
    """Load and fit one Figure (b) final-ratio series."""
    gamma, ratio, sem = load_final_ratio_data(data_path)
    fit = fit_asymmetric_rise_decay(gamma, ratio, sem)
    return CurveResult(
        label=label,
        model_name="asymmetric_rise_exp_linear_decay",
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


def add_peak_region_inset(
    axis,
    fixed_result: CurveResult,
    ascending_result: CurveResult,
) -> None:
    """Add a linear 0–4 inset that highlights both fitted peak values."""
    inset = axis.inset_axes([0.37, 0.57, 0.31, 0.30], zorder=6)
    series = [
        (
            fixed_result,
            "#1f4e79",
            "o",
            (0, (4, 3)),
            "Fixed",
        ),
        (
            ascending_result,
            "#EF7C00",
            "s",
            "-",
            "Ascending",
        ),
    ]
    displayed_values = []
    for result, color, marker, linestyle, _ in series:
        dense_mask = result.gamma_dense <= 4.0
        data_mask = result.gamma_data <= 4.0
        inset.plot(
            result.gamma_dense[dense_mask],
            result.ratio_dense[dense_mask],
            color=color,
            linestyle=linestyle,
            linewidth=1.5,
            zorder=2,
        )
        inset.scatter(
            result.gamma_data[data_mask],
            result.ratio_data[data_mask],
            marker=marker,
            color=color,
            s=24,
            edgecolors="white",
            linewidths=0.35,
            zorder=3,
        )
        peak_gamma = float(result.params["peak_gamma"])
        peak_value = float(result.params["peak_value"])
        inset.scatter(
            [peak_gamma],
            [peak_value],
            marker="*",
            color=color,
            s=70,
            edgecolors="white",
            linewidths=0.45,
            zorder=4,
        )
        displayed_values.extend(
            [
                *result.ratio_dense[dense_mask],
                *result.ratio_data[data_mask],
            ]
        )

    displayed_values = np.asarray(displayed_values, dtype=float)
    value_span = float(np.ptp(displayed_values))
    vertical_padding = max(0.18 * value_span, 0.004)
    inset.set(
        xlim=(0.0, 4.0),
        ylim=(
            float(np.min(displayed_values) - vertical_padding),
            float(np.max(displayed_values) + vertical_padding),
        ),
        xticks=[0, 1, 2, 3, 4],
    )
    inset.tick_params(axis="both", labelsize=7, width=0.8, length=3)
    inset.grid(
        color="#d9d9d9",
        linestyle=(0, (2, 2)),
        linewidth=0.55,
    )
    for spine in inset.spines.values():
        spine.set_linewidth(0.8)

    fixed_peak = (
        float(fixed_result.params["peak_gamma"]),
        float(fixed_result.params["peak_value"]),
    )
    ascending_peak = (
        float(ascending_result.params["peak_gamma"]),
        float(ascending_result.params["peak_value"]),
    )
    inset.text(
        0.98,
        0.97,
        (
            rf"Fixed: $\gamma^*={fixed_peak[0]:g}$, "
            rf"$R^*={fixed_peak[1]:.3f}$"
        ),
        transform=inset.transAxes,
        ha="right",
        va="top",
        fontsize=6.5,
        color="#1f4e79",
    )
    inset.text(
        0.98,
        0.84,
        (
            rf"Ascending: $\gamma^*={ascending_peak[0]:g}$, "
            rf"$R^*={ascending_peak[1]:.3f}$"
        ),
        transform=inset.transAxes,
        ha="right",
        va="top",
        fontsize=6.5,
        color="#EF7C00",
    )


def make_plot(
    fixed_result: CurveResult,
    ascending_result: CurveResult,
    output_png: Path,
    output_pdf: Path,
    *,
    x_scale: str = "linear",
) -> None:
    """Render the fitted curves with the established Figure (b) style."""
    if x_scale not in {"linear", "symlog"}:
        raise ValueError(f"Unsupported Figure (b) x scale: {x_scale}")
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
        clip_on=False,
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
        clip_on=False,
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
        xlabel=r"Average tilt parameter $|\gamma|$",
        ylabel="Mean Final Ratio",
        xlim=(
            0.0,
            float(
                min(
                    PLOT_GAMMA_MAX,
                    max(
                        np.max(fixed_result.gamma_data),
                        np.max(ascending_result.gamma_data),
                    ),
                )
            ),
        ),
        ylim=(0.65, 0.865),
    )
    if x_scale == "symlog":
        axis.set_xscale(
            "symlog",
            base=2,
            linthresh=1.0,
            linscale=1.0,
        )
        log_ticks = np.array(
            [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 36.0]
        )
        log_ticks = log_ticks[
            log_ticks <= axis.get_xlim()[1]
        ]
        axis.set_xticks(
            log_ticks,
            labels=[f"{value:g}" for value in log_ticks],
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
    add_peak_region_inset(
        axis,
        fixed_result,
        ascending_result,
    )
    save_figure(figure, output_png, output_pdf, dpi=600)


def make_log_plot(
    fixed_result: CurveResult,
    ascending_result: CurveResult,
    output_png: Path,
    output_pdf: Path,
) -> None:
    """Render Figure (b) with a zero-preserving logarithmic x-axis."""
    make_plot(
        fixed_result,
        ascending_result,
        output_png,
        output_pdf,
        x_scale="symlog",
    )


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


def plot_ascending_tilt_log(
    fixed_csv: Path,
    ascending_csv: Path,
    output_png: Path,
    output_pdf: Path,
    fit_summary_csv: Path | None = None,
) -> None:
    """Fit both series and render the log-x version of Figure (b)."""
    fixed_result = fit_curve_result(r"Fixed $\gamma$", fixed_csv)
    ascending_result = fit_curve_result(
        r"Ascending $\gamma$",
        ascending_csv,
    )
    make_log_plot(
        fixed_result,
        ascending_result,
        output_png,
        output_pdf,
    )
    if fit_summary_csv is not None:
        fit_summary_csv = Path(fit_summary_csv)
        fit_summary_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            fit_summary_rows(fixed_result, ascending_result)
        ).to_csv(fit_summary_csv, index=False)


def build_parser() -> argparse.ArgumentParser:
    """Build the Figure (b) command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Fit asymmetric rise-and-decay curves and plot Figure (b)."
        )
    )
    parser.add_argument(
        "--fixed-csv",
        type=Path,
        default=DATA_DIR / "fixed_summary.csv",
    )
    parser.add_argument(
        "--ascending-csv",
        type=Path,
        default=(
            DATA_DIR / "ascending_summary.csv"
        ),
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=DATA_DIR / "large_gamma_figure_b.png",
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=DATA_DIR / "large_gamma_figure_b.pdf",
    )
    parser.add_argument(
        "--fit-summary-csv",
        type=Path,
        default=DATA_DIR / "large_gamma_fit_summary.csv",
    )
    parser.add_argument(
        "--log-x",
        action="store_true",
        help=(
            "Use a zero-preserving symmetric-log x-axis with a linear "
            "segment from gamma 0 to 1."
        ),
    )
    return parser


def main() -> None:
    """Load command-line paths and regenerate Figure (b)."""
    arguments = build_parser().parse_args()
    plot_function = (
        plot_ascending_tilt_log
        if arguments.log_x
        else plot_ascending_tilt
    )
    plot_function(
        arguments.fixed_csv,
        arguments.ascending_csv,
        arguments.output_png,
        arguments.output_pdf,
        arguments.fit_summary_csv,
    )


if __name__ == "__main__":
    main()
