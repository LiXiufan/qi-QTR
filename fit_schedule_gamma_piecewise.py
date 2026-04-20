from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mplconfig_codex"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


INPUT_PATH = Path(__file__).with_name("schedule_gamma_df_group_2.csv")
OUTPUT_PNG = Path(__file__).with_name("schedule_gamma_df_group_2_piecewise_fit.png")
OUTPUT_PDF = Path(__file__).with_name("schedule_gamma_df_group_2_piecewise_fit.pdf")
OUTPUT_SUMMARY = Path(__file__).with_name("schedule_gamma_df_group_2_piecewise_fit_summary.csv")

BREAKPOINT = 1.0
GAMMA_COLUMN = "gamma_plot"
RATIO_COLUMN = "mean_peak_ratio"
SEM_COLUMN = "sem_peak_ratio"


@dataclass
class FitResult:
    params: np.ndarray
    gamma_data: np.ndarray
    ratio_data: np.ndarray
    sem_data: np.ndarray
    gamma_dense: np.ndarray
    ratio_dense: np.ndarray
    r2: float
    rmse: float

    @property
    def value_at_break(self) -> float:
        return float(self.params[1])

    @property
    def breakpoint(self) -> float:
        return float(self.params[0])

    @property
    def slope(self) -> float:
        return float(self.params[2])

    @property
    def tail_floor(self) -> float:
        return float(self.params[3])

    @property
    def tail_scale(self) -> float:
        return float(self.params[4])

    @property
    def tail_power(self) -> float:
        return float(self.params[5])

    @property
    def transition_width(self) -> float:
        return float(self.params[6])


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def smooth_single_peak_model(gamma: np.ndarray, params: np.ndarray) -> np.ndarray:
    breakpoint, value_at_break, slope, tail_floor, tail_scale, tail_power, transition_width = params
    gamma = np.asarray(gamma, dtype=float)
    left_branch = value_at_break + slope * (gamma - breakpoint)
    right_distance = np.maximum(gamma - breakpoint, 0.0)
    right_branch = tail_floor + (value_at_break - tail_floor) / (
        1.0 + tail_scale * right_distance**tail_power
    )
    blend_weight = sigmoid((gamma - breakpoint) / transition_width)
    return (1.0 - blend_weight) * left_branch + blend_weight * right_branch


def fit_schedule_piecewise(frame: pd.DataFrame) -> FitResult:
    ordered = frame.sort_values(GAMMA_COLUMN).copy()
    gamma = ordered[GAMMA_COLUMN].to_numpy(dtype=float)
    ratio = ordered[RATIO_COLUMN].to_numpy(dtype=float)
    sem = ordered[SEM_COLUMN].to_numpy(dtype=float)
    sigma = np.maximum(sem, 1e-3)

    lower = np.array([0.70, 0.75, 0.0, 0.58, 1e-5, 0.5, 0.03], dtype=float)
    upper = np.array([1.40, 0.85, 0.20, 0.80, 5.0, 6.0, 0.80], dtype=float)

    value_guess = float(np.interp(BREAKPOINT, gamma, ratio))
    slope_guess = max((value_guess - ratio[0]) / max(BREAKPOINT - gamma[0], 1e-6), 0.01)
    floor_guess = max(min(ratio[-1], ratio.min()) - 0.02, 0.60)

    initial_guesses = [
        np.array([1.00, value_guess, slope_guess, floor_guess, 0.023, 1.00, 0.08], dtype=float),
        np.array([1.10, value_guess + 0.002, slope_guess * 0.95, floor_guess + 0.02, 0.015, 1.20, 0.12], dtype=float),
        np.array([1.20, value_guess + 0.003, slope_guess * 0.90, floor_guess, 0.030, 1.00, 0.06], dtype=float),
        np.array([0.95, value_guess, slope_guess * 1.10, floor_guess + 0.04, 0.020, 1.50, 0.15], dtype=float),
    ]

    best_params = None
    best_cost = np.inf

    def residuals(params: np.ndarray) -> np.ndarray:
        return (smooth_single_peak_model(gamma, params) - ratio) / sigma

    for guess in initial_guesses:
        clipped_guess = np.clip(guess, lower + 1e-6, upper - 1e-6)
        result = least_squares(
            residuals,
            clipped_guess,
            bounds=(lower, upper),
            loss="soft_l1",
            max_nfev=50000,
        )
        if result.cost < best_cost:
            best_cost = result.cost
            best_params = result.x

    if best_params is None:
        raise RuntimeError("Smooth single-peak fit failed for schedule_gamma_df_group_2.csv")

    gamma_dense = np.linspace(float(gamma.min()), float(gamma.max()), 1600)
    ratio_dense = smooth_single_peak_model(gamma_dense, best_params)
    fitted = smooth_single_peak_model(gamma, best_params)

    ss_res = float(np.sum((ratio - fitted) ** 2))
    ss_tot = float(np.sum((ratio - np.mean(ratio)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean((ratio - fitted) ** 2)))

    return FitResult(
        params=best_params,
        gamma_data=gamma,
        ratio_data=ratio,
        sem_data=sem,
        gamma_dense=gamma_dense,
        ratio_dense=ratio_dense,
        r2=r2,
        rmse=rmse,
    )


def save_summary(result: FitResult, output_path: Path) -> None:
    summary = pd.DataFrame(
        [
            {
                "dataset": INPUT_PATH.name,
                "gamma_column": GAMMA_COLUMN,
                "ratio_column": RATIO_COLUMN,
                "sem_column": SEM_COLUMN,
                "model_type": "smooth_single_peak_transition",
                "breakpoint": result.breakpoint,
                "left_model": "linear rise branch",
                "right_model": "power-law decay branch",
                "transition_model": "logistic blend between branches",
                "value_at_break": result.value_at_break,
                "slope": result.slope,
                "tail_floor": result.tail_floor,
                "tail_scale": result.tail_scale,
                "tail_power": result.tail_power,
                "transition_width": result.transition_width,
                "r2": result.r2,
                "rmse": result.rmse,
            }
        ]
    )
    summary.to_csv(output_path, index=False)


def make_plot(result: FitResult, output_png: Path, output_pdf: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, ax = plt.subplots(figsize=(8.8, 5.4), constrained_layout=True)

    point_color = "#214d72"
    curve_color = "#b23a48"
    accent_color = "#7a7a7a"

    ax.errorbar(
        result.gamma_data,
        result.ratio_data,
        yerr=result.sem_data,
        fmt="none",
        lw=0.8,
        capsize=1.8,
        color=point_color,
        ecolor=point_color,
        alpha=0.22,
        zorder=1,
    )
    ax.scatter(
        result.gamma_data,
        result.ratio_data,
        s=28,
        facecolor="white",
        edgecolor=point_color,
        linewidth=1.1,
        alpha=0.95,
        label="Schedule data",
        zorder=3,
    )
    ax.plot(
        result.gamma_dense,
        result.ratio_dense,
        color=curve_color,
        lw=2.2,
        label="Smooth one-peak fit",
        zorder=2,
    )
    ax.axvline(
        result.breakpoint,
        color=accent_color,
        lw=1.0,
        ls="--",
        alpha=0.8,
        label=rf"Peak transition $\gamma = {result.breakpoint:.2f}$",
    )

    ax.set_xlabel(r"Schedule parameter $\gamma$")
    ax.set_ylabel("Mean peak ratio")
    ax.set_xlim(-0.5, 66.0)
    ax.set_ylim(0.66, 0.835)
    ax.grid(axis="y", color="#d9d9d9", lw=0.8, alpha=0.7)

    annotation = (
        "Smooth one-peak model\n"
        "Linear rise + power-law decay\n"
        rf"Transition width = {result.transition_width:.3f}" "\n"
        rf"$R^2 = {result.r2:.3f}$, RMSE = {result.rmse:.4f}"
    )
    ax.text(
        0.98,
        0.06,
        annotation,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cfcfcf", "alpha": 0.95},
    )

    ax.legend(loc="upper right", frameon=False)

    inset = inset_axes(ax, width="38%", height="43%", loc="lower left", borderpad=2.2)
    inset.errorbar(
        result.gamma_data,
        result.ratio_data,
        yerr=result.sem_data,
        fmt="none",
        lw=0.7,
        capsize=1.6,
        color=point_color,
        ecolor=point_color,
        alpha=0.22,
        zorder=1,
    )
    inset.scatter(
        result.gamma_data,
        result.ratio_data,
        s=18,
        facecolor="white",
        edgecolor=point_color,
        linewidth=0.95,
        alpha=0.95,
        zorder=3,
    )
    inset.plot(result.gamma_dense, result.ratio_dense, color=curve_color, lw=1.8, zorder=2)
    inset.axvline(result.breakpoint, color=accent_color, lw=0.9, ls="--", alpha=0.8)
    inset.set_xlim(-0.05, 3.2)
    inset.set_ylim(0.735, 0.815)
    inset.set_xticks([0, 1, 2, 3])
    inset.set_yticks([0.74, 0.77, 0.80])
    inset.grid(axis="y", color="#e4e4e4", lw=0.7, alpha=0.8)
    inset.set_title("Near-breakpoint view", fontsize=11)

    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    frame = pd.read_csv(INPUT_PATH)
    result = fit_schedule_piecewise(frame)
    save_summary(result, OUTPUT_SUMMARY)
    make_plot(result, OUTPUT_PNG, OUTPUT_PDF)

    print(f"Saved summary: {OUTPUT_SUMMARY}")
    print(f"Saved plot: {OUTPUT_PNG}")
    print(f"Saved plot: {OUTPUT_PDF}")
    print(
        "Fit parameters:",
        {
            "breakpoint": round(result.breakpoint, 6),
            "value_at_break": round(result.value_at_break, 6),
            "slope": round(result.slope, 6),
            "tail_floor": round(result.tail_floor, 6),
            "tail_scale": round(result.tail_scale, 6),
            "tail_power": round(result.tail_power, 6),
            "transition_width": round(result.transition_width, 6),
            "r2": round(result.r2, 6),
            "rmse": round(result.rmse, 6),
        },
    )


if __name__ == "__main__":
    main()
