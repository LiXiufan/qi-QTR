from __future__ import annotations

import argparse
import os
import tempfile
from dataclasses import dataclass
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


@dataclass
class CurveResult:
    label: str
    model_name: str
    gamma_data: np.ndarray
    ratio_data: np.ndarray
    sem_data: np.ndarray
    gamma_dense: np.ndarray
    ratio_dense: np.ndarray
    params: dict[str, float]


def lorentzian(x, g, h):
    x = np.asarray(x, dtype=float)
    return (h * h) / ((x - g) ** 2 + h * h)


def poly_tail_model(x, a, b, c, d, g, h):
    x = np.asarray(x, dtype=float)
    return a + b * x + c / (x + 1.0) + d * lorentzian(x, g, h)


def exp_quad_model(x, a, b, c, d):
    x = np.asarray(x, dtype=float)
    return d + (a + b * x) * np.exp(-c * x * x)


def exp_quad_first_derivative(x, a, b, c):
    x = np.asarray(x, dtype=float)
    return np.exp(-c * x * x) * (b - 2.0 * c * x * (a + b * x))


def exp_quad_second_derivative(x, a, b, c):
    x = np.asarray(x, dtype=float)
    return np.exp(-c * x * x) * (-2.0 * c * a - 6.0 * b * c * x + 4.0 * c * c * x * x * (a + b * x))


def resolve_existing_path(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    joined = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"None of these files exists:\n{joined}")


def load_curve_frame(path: Path, ratio_column: str, sem_column: str):
    frame = pd.read_csv(path).sort_values("gamma_plot")
    return (
        frame["gamma_plot"].to_numpy(dtype=float),
        frame[ratio_column].to_numpy(dtype=float),
        np.maximum(frame[sem_column].to_numpy(dtype=float), 1e-3),
    )


def load_fixed_curve_result(fixed_path: Path, summary_path: Path) -> CurveResult:
    gamma, ratio, sem = load_curve_frame(fixed_path, "mean_final_ratio", "sem_final_ratio")
    summary = pd.read_csv(summary_path)
    row = summary.loc[summary["dataset"] == "shot=5000"]
    if row.empty:
        raise ValueError("Missing shot=5000 row in curve_fitting_poly_tail_summary.csv")

    row = row.iloc[0]
    params = {key: float(row[key]) for key in ["a", "b", "c", "d", "g", "h"]}
    gamma_dense = np.linspace(float(gamma.min()), float(gamma.max()), DENSE_POINTS)
    ratio_dense = poly_tail_model(gamma_dense, **params)
    return CurveResult(
        label=r"Fixed $\gamma$",
        model_name="poly_tail_from_curve_fitting_data_shot5000",
        gamma_data=gamma,
        ratio_data=ratio,
        sem_data=sem,
        gamma_dense=gamma_dense,
        ratio_dense=ratio_dense,
        params=params,
    )


def fit_schedule_exp_quad(gamma, ratio, sem):
    gamma = np.asarray(gamma, dtype=float)
    ratio = np.asarray(ratio, dtype=float)
    sem = np.asarray(sem, dtype=float)

    y0 = float(ratio[0])
    tail = float(ratio[-1])
    peak_idx = int(np.argmax(ratio))
    x_peak = float(gamma[peak_idx])
    y_peak = float(ratio[peak_idx])

    d_guess = float(np.clip(tail - 0.0015, 0.775, 0.798))
    a_guess = float(np.clip(y0 - d_guess, 0.002, 0.03))
    b_guess = float(np.clip((y_peak - y0) / max(x_peak, 0.25) + 0.008, 0.001, 0.08))
    c_guess = 0.12

    starts = [
        np.array([a_guess, b_guess, c_guess, d_guess], dtype=float),
        np.array([min(0.05, a_guess * 1.2), min(0.10, b_guess * 1.2), 0.08, min(0.80, d_guess + 0.001)], dtype=float),
        np.array([max(0.001, a_guess * 0.8), min(0.10, b_guess * 1.5), 0.18, max(0.775, d_guess - 0.001)], dtype=float),
        np.array([min(0.05, a_guess + 0.004), max(0.002, b_guess * 0.9), 0.25, d_guess], dtype=float),
    ]
    bounds = ([0.0, 0.0, 0.01, 0.775], [0.06, 0.12, 1.20, 0.80])

    min_lift = 0.004
    anchor_weight = 1000.0

    def residuals(params):
        a, b, c, d = params
        data_res = (exp_quad_model(gamma, a, b, c, d) - ratio) / sem

        gamma_dense = np.linspace(0.0, FIT_GAMMA_MAX, 2001)
        values_dense = exp_quad_model(gamma_dense, a, b, c, d)
        peak_idx_dense = int(np.argmax(values_dense))
        peak_gamma = float(gamma_dense[peak_idx_dense])
        peak_value = float(values_dense[peak_idx_dense])
        anchor_value = float(exp_quad_model(x_peak, a, b, c, d))
        lift = peak_value - float(exp_quad_model(0.0, a, b, c, d))

        penalty_terms = [
            30.0 * max(0.0, 0.70 - peak_gamma),
            30.0 * max(0.0, peak_gamma - 1.20),
            10.0 * max(0.0, -float(exp_quad_first_derivative(0.25, a, b, c))),
            10.0 * max(0.0, -float(exp_quad_first_derivative(0.60, a, b, c))),
            10.0 * max(0.0, float(exp_quad_first_derivative(1.40, a, b, c))),
            10.0 * max(0.0, float(exp_quad_first_derivative(2.20, a, b, c))),
            8.0 * max(0.0, -float(exp_quad_second_derivative(0.90, a, b, c))),
            8.0 * max(0.0, -float(exp_quad_second_derivative(1.10, a, b, c))),
            14.0 * max(0.0, min_lift - lift),
            anchor_weight * (anchor_value - y_peak),
        ]
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
        "model_formula": r"$d + (a+b|\gamma|)e^{-c|\gamma|^2}$",
        "a": a,
        "b": b,
        "c": c,
        "d": d,
        "weighted_sse": best_score,
        "peak_gamma": peak_gamma,
        "peak_value": peak_value,
        "tail_value": float(exp_quad_model(4.0, a, b, c, d)),
        "gamma_dense": gamma_dense,
        "ratio_dense": ratio_dense,
    }


def load_schedule_curve_result(schedule_path: Path) -> CurveResult:
    gamma, ratio, sem = load_curve_frame(schedule_path, "mean_peak_ratio", "sem_peak_ratio")
    params = fit_schedule_exp_quad(gamma, ratio, sem)
    return CurveResult(
        label=r"Ascending $\gamma$",
        model_name="exp_quad_shifted_schedule",
        gamma_data=gamma,
        ratio_data=ratio,
        sem_data=sem,
        gamma_dense=params["gamma_dense"],
        ratio_dense=params["ratio_dense"],
        params=params,
    )


def calculate_sem(result: CurveResult) -> float:
    return float(np.sqrt(np.mean(np.square(result.sem_data))))


def summary_rows(result_fixed: CurveResult, result_schedule: CurveResult):
    rows = []
    fixed_row = {"dataset": "fixed", "model": result_fixed.model_name}
    fixed_row.update(result_fixed.params)
    rows.append(fixed_row)

    schedule_row = {"dataset": "ascending", "model": result_schedule.model_name}
    schedule_row.update(
        {
            key: value
            for key, value in result_schedule.params.items()
            if key not in {"gamma_dense", "ratio_dense"}
        }
    )
    rows.append(schedule_row)
    return rows


def make_plot(result_fixed: CurveResult, result_schedule: CurveResult, output_png: Path, output_pdf: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    fixed_sem = calculate_sem(result_fixed)
    schedule_sem = calculate_sem(result_schedule)

    ax.fill_between(
        result_fixed.gamma_dense,
        result_fixed.ratio_dense - fixed_sem,
        result_fixed.ratio_dense + fixed_sem,
        color="#1f4e79",
        alpha=0.08,
        zorder=1,
    )
    ax.fill_between(
        result_schedule.gamma_dense,
        result_schedule.ratio_dense - schedule_sem,
        result_schedule.ratio_dense + schedule_sem,
        color="#EF7C00",
        alpha=0.08,
        zorder=1,
    )

    ax.plot(
        result_fixed.gamma_dense,
        result_fixed.ratio_dense,
        color="#1f4e79",
        linestyle=(0, (4, 3)),
        linewidth=2.2,
        zorder=3,
    )
    ax.plot(
        result_schedule.gamma_dense,
        result_schedule.ratio_dense,
        color="#EF7C00",
        linestyle="-",
        linewidth=2.2,
        zorder=3,
    )

    ax.scatter(result_fixed.gamma_data, result_fixed.ratio_data, marker="o", color="#1f4e79", s=70, edgecolors="white", linewidths=0.5, zorder=4)
    ax.scatter(result_schedule.gamma_data, result_schedule.ratio_data, marker="s", color="#EF7C00", s=70, edgecolors="white", linewidths=0.5, zorder=4)

    legend_handles = [
        Line2D([0], [0], color="#1f4e79", linestyle=(0, (4, 3)), linewidth=2.2, marker="o", markersize=8, markerfacecolor="#1f4e79", markeredgewidth=0.0, label=r"Fixed-$\gamma$"),
        Line2D([0], [0], color="#EF7C00", linestyle="-", linewidth=2.2, marker="s", markersize=8, markerfacecolor="#EF7C00", markeredgewidth=0.0, label=r"Ascending-$\gamma$"),
    ]

    ax.set_xlabel(r"Tilt parameter $|\gamma|$")
    ax.set_ylabel("Mean Final Ratio")
    ax.set_xlim(0, 4)
    ax.set_ylim(0.65, 0.865)
    ax.set_yticks(np.arange(0.65, 0.851, 0.05))
    ax.grid(axis="y", color="#cfcfcf", linestyle=(0, (3, 3)), linewidth=1.0)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["top"].set_linewidth(1.2)
    ax.spines["right"].set_linewidth(1.2)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.legend(handles=legend_handles, loc="upper right")

    fig.savefig(output_png, dpi=600, format="png")
    fig.savefig(output_pdf, format="pdf")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot the fixed-vs-ascending tilt comparison curve.")
    parser.add_argument(
        "--fixed-path",
        type=Path,
        default=SCRIPT_DIR / "data_and_figures" / "fixed_gamma_shot_5000.csv",
    )
    parser.add_argument(
        "--schedule-path",
        type=Path,
        default=SCRIPT_DIR / "data_and_figures" / "schedule_gamma_restart_group(shots5000)all.csv",
    )
    parser.add_argument(
        "--curve-fit-summary-path",
        type=Path,
        default=SCRIPT_DIR / "data_and_figures" / "curve_fitting_poly_tail_summary.csv",
    )
    parser.add_argument("--output-png", type=Path, default=SCRIPT_DIR / "schedule_gamma_expquad_fit.png")
    parser.add_argument("--output-pdf", type=Path, default=SCRIPT_DIR / "schedule_gamma_expquad_fit.pdf")
    parser.add_argument("--output-summary", type=Path, default=SCRIPT_DIR / "schedule_gamma_expquad_fit_summary.csv")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    schedule_path = resolve_existing_path(
        [
            args.schedule_path.resolve(),
            SCRIPT_DIR / "schedule_gamma_restart_group(shots5000)new.csv",
            SCRIPT_DIR / "schedule_gamma_restart_group(shots5000)all.csv",
            SCRIPT_DIR / "submission" / "schedule_gamma_restart_group(shots5000)all.csv",
        ]
    )
    fixed_path = args.fixed_path.resolve()
    summary_path = args.curve_fit_summary_path.resolve()

    result_fixed = load_fixed_curve_result(fixed_path, summary_path)
    result_schedule = load_schedule_curve_result(schedule_path)
    make_plot(result_fixed, result_schedule, args.output_png.resolve(), args.output_pdf.resolve())
    pd.DataFrame(summary_rows(result_fixed, result_schedule)).to_csv(args.output_summary.resolve(), index=False)


if __name__ == "__main__":
    main()
