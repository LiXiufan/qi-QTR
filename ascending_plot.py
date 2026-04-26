from __future__ import annotations

from scipy.optimize import least_squares
import math

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mplconfig_codex"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


INPUT_PATH_FIXED = Path(__file__).with_name("fixed_gamma_shot_1024.csv")
INPUT_PATH_SCEDULE = Path(__file__).with_name("schedule_gamma_df_group_2.csv")

OUTPUT_PNG = Path(__file__).with_name("schedule_gamma_df_group_2_piecewise_fit.png")
OUTPUT_PDF = Path(__file__).with_name("schedule_gamma_df_group_2_piecewise_fit.pdf")
OUTPUT_SUMMARY = Path(__file__).with_name("schedule_gamma_df_group_2_piecewise_fit_summary.csv")

BREAKPOINT = 1.0
GAMMA_COLUMN = "gamma_plot"
RATIO_COLUMN = "mean_peak_ratio"
SEM_COLUMN = "sem_peak_ratio"

# Configure matplotlib for an elegant, academic publication style
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "font.serif": ["Times New Roman", "DejaVu Serif"], 
    "font.size": 20,
    "axes.labelsize": 20,
    "axes.titlesize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,
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
    "figure.autolayout": True
})


@dataclass
class FitResult:
    params: np.ndarray
    gamma_data: np.ndarray
    ratio_data: np.ndarray
    sem_data: np.ndarray
    gamma_dense: np.ndarray
    ratio_dense: np.ndarray

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

    # gamma_dense = np.linspace(float(gamma.min()), float(gamma.max()), 1600)
    gamma_dense = np.geomspace(1e-4, gamma.max() + 1.0, 800) - 1e-4
    gamma_dense[0] = 0.0
    ratio_dense = smooth_single_peak_model(gamma_dense, best_params)

    return FitResult(
        params=best_params,
        gamma_data=gamma,
        ratio_data=ratio,
        sem_data=sem,
        gamma_dense=gamma_dense,
        ratio_dense=ratio_dense,
    )


def calculate_sem(result:FitResult):
    sem_ave = np.sqrt(np.sum(result.sem_data ** 2) / len(result.sem_data))
    return sem_ave

def make_plot(result_fixed:FitResult, result_schedule:FitResult, output_png: Path, output_pdf: Path) -> None:

    fig, ax = plt.subplots(figsize=(8, 6))

    
    # ax.errorbar(
    #     result.gamma_data,
    #     result.ratio_data,
    #     yerr=result.sem_data,
    #     fmt="none",
    #     lw=0.8,
    #     capsize=1.8,
    #     color=point_color,
    #     ecolor=point_color,
    #     alpha=0.22,
    #     zorder=1,
    # )

    ax.fill_between(result_fixed.gamma_dense, result_fixed.ratio_dense - calculate_sem(result_fixed), result_fixed.ratio_dense + calculate_sem(result_fixed), color='#1f4e79', alpha=0.1, zorder=1)
    ax.fill_between(result_schedule.gamma_dense, result_schedule.ratio_dense - calculate_sem(result_schedule), result_schedule.ratio_dense + calculate_sem(result_schedule), color='#EF7C00', alpha=0.1, zorder=1)
    

    # Plot continuous fitting curve cleanly
    ax.plot(result_fixed.gamma_dense, result_fixed.ratio_dense, label=r"Fixed $\gamma$", color='#1f4e79', alpha=0.8, linewidth=2, zorder=3)
    ax.plot(result_schedule.gamma_dense, result_schedule.ratio_dense, label=r"Linear ascending $\gamma$", color='#EF7C00', alpha=0.8, linewidth=2, zorder=3)

        
    # Plot scattered original points without borders and slightly smaller
    ax.scatter(result_fixed.gamma_data, result_fixed.ratio_data, marker= 'o', color='#1f4e79', s=20, 
            edgecolors='none', zorder=4)

    ax.scatter(
        result_schedule.gamma_data,
        result_schedule.ratio_data,
        marker= 'o',
        color='#EF7C00',
        s=20,
        # label="Schedule data",
        zorder=3,
    )

    # ax.scatter(result_fixed.gamma_data, result_fixed.ratio_data, facecolors='none', edgecolors='#1f4e79', marker= 'o', s=25, zorder=4)

    # ax.scatter(
    #     result_schedule.gamma_data,
    #     result_schedule.ratio_data, facecolors='none', edgecolors='#EF7C00',
    #     marker= 'o',
    #     s=25,
    #     # label="Schedule data",
    #     zorder=4,
    # )
    # ax.axvline(
    #     result.breakpoint,
    #     color=accent_color,
    #     lw=1.0,
    #     ls="--",
    #     alpha=0.8,
    #     label=rf"Peak transition $\gamma = {result.breakpoint:.2f}$",
    # )

    ax.set_xlabel(r"Tilt parameter $|\gamma|$")
    ax.set_ylabel("Mean Final Ratio")
    ax.set_xlim(0, 30)
    ax.set_ylim(0.55, 0.9)
    # ax.grid(axis="y", color="#d9d9d9", lw=0.8, alpha=0.7)
    ax.grid(False)

    # annotation = (
    #     "Smooth one-peak model\n"
    #     "Linear rise + power-law decay\n"
    #     rf"Transition width = {result_schedule.transition_width:.3f}" "\n"
    # )
    # ax.text(
    #     0.98,
    #     0.06,
    #     annotation,
    #     transform=ax.transAxes,
    #     ha="right",
    #     va="bottom",
    #     fontsize=11,
    #     bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cfcfcf", "alpha": 0.95},
    # )

    ax.legend(loc="best")

    fig.savefig(output_png, dpi=600, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def double_peak_decay_model(gamma, params):
    # Model: baseline_floor + baseline_amp * exp(-baseline_decay * gamma) + peak1 + peak2
    baseline_floor, baseline_amp, baseline_decay, amp1, center1, width1, amp2, delta2, width2 = params
    center2 = center1 + delta2
    baseline = baseline_floor + baseline_amp * np.exp(-baseline_decay * gamma)
    peak1 = amp1 * np.exp(-0.5 * ((gamma - center1) / width1) ** 2)
    peak2 = amp2 * np.exp(-0.5 * ((gamma - center2) / width2) ** 2)
    return baseline + peak1 + peak2

def fit_single_dataset(gamma, ratio, sem):
    gamma_max = float(gamma.max())
    # sigma = sem
    sigma = np.maximum(sem, 1e-3) if sem is not None else np.ones_like(ratio)

    # first_peak_guesses = [0.4, 0.5, 0.6]
    # second_peak_guesses = [2.5, 3.5, 4.5, 6.0]
    # decay_guesses = [1e-4, 5e-4, 1e-3, 5e-3, 2e-2]

    first_peak_guesses = [0.45, 0.7, 1.0]
    second_peak_guesses = [2.0, 3.0, 4.5, 6.0]
    # second_peak_guesses = [3.8, 4.0, 5.0]
    decay_guesses = [1e-4, 5e-4, 1e-3, 5e-3, 2e-2]
    
    floor_guess = max(min(ratio.min(), ratio[-1]) - 0.02, 0.0)
    amp_guess = max(ratio[0] - floor_guess, 0.05)

    lower = np.array([0.0, 0.0, 1e-8, 0.0, 0.0, 0.03, 0.0, 0.05, 0.05], dtype=float)
    upper = np.array([1.0, 1.0, 1.0, 1.0, min(2.0, gamma_max), max(4.0, gamma_max / 3.0), 1.0, gamma_max + 5.0, max(20.0, gamma_max / 2.0)], dtype=float)

    def residuals(params):
        return (double_peak_decay_model(gamma, params) - ratio) / sigma

    best = None
    best_cost = math.inf

    for first_gamma in first_peak_guesses:
        center1 = min(first_gamma, upper[4] - 1e-3)
        for second_gamma in second_peak_guesses:
            center2 = min(second_gamma, gamma_max)
            delta2 = max(center2 - center1, lower[7] + 1e-3)
            delta2 = min(delta2, upper[7] - 1e-3)
            for baseline_decay in decay_guesses:
                start = np.array([
                    floor_guess,
                    amp_guess,
                    baseline_decay,
                    0.05,
                    center1,
                    0.18,
                    0.08,
                    delta2,
                    1.10,
                ], dtype=float)
                try:
                    result = least_squares(residuals, start, bounds=(lower, upper), loss="soft_l1", max_nfev=10000)
                except ValueError:
                    continue
                if result.cost < best_cost:
                    best_cost = result.cost
                    best = result.x
    return best


def main() -> None:
    filename = 'fixed_gamma_shot_1024.csv'
    
    df = pd.read_csv(filename)
    df = df.dropna(subset=['gamma_plot', 'mean_final_ratio'])
    df = df.sort_values(by='gamma_plot')
            
    gamma = df['gamma_plot'].values
    ratio = df['mean_final_ratio'].values
    sem = df['sem_final_ratio'].values

    best_params = fit_single_dataset(gamma, ratio, sem)

    if best_params is not None:
        # Generate points up to max 18 for a smooth curve presentation
        # gamma_dense = np.linspace(0, 18, 1000) 
        gamma_dense = np.geomspace(1e-4, gamma.max() + 1.0, 800) - 1e-4
        gamma_dense[0] = 0.0
        ratio_dense = double_peak_decay_model(gamma_dense, best_params)

    result_fixed = FitResult(
        params=best_params,
        gamma_data=gamma,
        ratio_data=ratio,
        sem_data=sem,
        gamma_dense=gamma_dense,
        ratio_dense=ratio_dense,
    )
            
    frame_schedule = pd.read_csv(INPUT_PATH_SCEDULE)

    result_schedule = fit_schedule_piecewise(frame_schedule)
    # save_summary(result, OUTPUT_SUMMARY)
    make_plot(result_fixed, result_schedule, OUTPUT_PNG, OUTPUT_PDF)

   


if __name__ == "__main__":
    main()
