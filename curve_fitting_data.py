import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import math
import os

# Configure matplotlib for an elegant, academic publication style
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "font.serif": ["Times New Roman", "DejaVu Serif"], 
    "font.size": 20,
    "axes.labelsize": 20,
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

def main():
    datasets = [
        ('shot=512', 'fixed_gamma_shot_512.csv', 'o', '#c23b22'),
        ('shot=1024', 'fixed_gamma_shot_1024.csv', 'o', '#1f4e79'),
        ('shot=5000', 'fixed_gamma_shot_5000.csv', 'o', '#2c6e49'),
        ('shot=10000', 'fixed_gamma_shot_10000.csv', 'o', '#7a3e9d')
    ]

    fig1, ax1 = plt.subplots(figsize=(8, 6))
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    fig3, ax3 = plt.subplots(figsize=(8, 6))

    for label, filename, marker, color in datasets:
                
        if not os.path.exists(filename):
            print(f"File {filename} not found, skipping...")
            continue
            
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
            
            # Plot continuous fitting curve cleanly
            ax1.plot(gamma_dense, ratio_dense, label=label, color=color, linewidth=1.5, zorder=3)
            ax2.plot(gamma_dense, ratio_dense, label=label, color=color, linewidth=1.5, zorder=3)
            ax3.plot(gamma_dense, ratio_dense, label=label, color=color, linewidth=1.5, zorder=3)
            
            # # Use damped (alpha=0.08) colors to indicate the narrowed range of peaked gammas
            # center1 = best_params[4]
            # width1 = best_params[5] * 0.2   # Reduced width
            # center2 = center1 + best_params[7]
            # width2 = best_params[8] * 0.2   # Reduced width
            # light_alpha = 0.05              # Lightened color
            
            # # Full View (covers both peaks)
            # ax1.axvspan(center1 - width1, center1 + width1, color=color, alpha=light_alpha, zorder=1)
            # ax1.axvspan(center2 - width2, center2 + width2, color=color, alpha=light_alpha, zorder=1)
            
            # # Enlarged View 1 (covers first peak, around gamma 0-2)
            # ax2.axvspan(center1 - width1, center1 + width1, color=color, alpha=light_alpha, zorder=1)
            
            # # Enlarged View 2 (covers second peak, around gamma 2-7)
            # # Both peaks may optionally be visible if they overlap with the boundaries
            # ax3.axvspan(center1 - width1, center1 + width1, color=color, alpha=light_alpha, zorder=1)
            # ax3.axvspan(center2 - width2, center2 + width2, color=color, alpha=light_alpha, zorder=1)
            
            print(f"[{label}] fit success with params: {best_params}")
        else:
            print(f"[{label}] fit failed")
            
        # Plot scattered original points without borders and slightly smaller
        ax1.scatter(gamma, ratio, marker=marker, color=color, s=20, 
                   edgecolors='none', zorder=4)
        ax2.scatter(gamma, ratio, marker=marker, color=color, s=20, 
                   edgecolors='none', zorder=4)
        ax3.scatter(gamma, ratio, marker=marker, color=color, s=20, 
                   edgecolors='none', zorder=4)

    # First Plot - Full Range View
    ax1.set_xlabel(r'Tilt parameter $|\gamma|$')
    ax1.set_ylabel('Mean Final Ratio')
    ax1.set_xlim(left=0, right=10)
    ax1.set_ylim(bottom=0.68, top=0.83)
    ax1.grid(False)
    ax1.legend(loc='best')

    fig1.savefig('curve_fitting_results.pdf', format='pdf')
    fig1.savefig('curve_fitting_results.png', dpi=600, format='png')
    
    # Second Plot - Enlarged View (0-2)
    ax2.set_xlabel(r'Tilt parameter $|\gamma|$')
    ax2.set_ylabel('Mean Final Ratio')
    ax2.set_xlim(left=0.25, right=1.25)
    ax2.set_ylim(bottom=0.71, top=0.81)
    ax2.grid(False)
    ax2.legend(loc='best')

    fig2.savefig('curve_fitting_results_enlarged.pdf', format='pdf')
    fig2.savefig('curve_fitting_results_enlarged.png', dpi=600, format='png')
    
    # Third Plot - Second Enlarged View (2-7)
    # ax3.set_xlabel(r'Fixed $\gamma$')
    # ax3.set_ylabel('Mean Final Ratio')
    ax3.set_xlim(left=2, right=6)
    ax3.set_ylim(bottom=0.73, top=0.83)
    ax3.grid(False)
    # ax3.legend(loc='best')

    fig3.savefig('curve_fitting_results_enlarged_2to7.pdf', format='pdf')
    fig3.savefig('curve_fitting_results_enlarged_2to7.png', dpi=600, format='png')
    
    print("Saved normal and enlarged plots.")

if __name__ == '__main__':
    main()
