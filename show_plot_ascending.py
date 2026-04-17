import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.interpolate import make_interp_spline
import warnings
from scipy.optimize import curve_fit

# Configure matplotlib for an elegant, academic publication style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"], # Academic standard fonts
    "font.size": 14,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "legend.frameon": False,
    "axes.linewidth": 1.2,
    "lines.linewidth": 2,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "figure.autolayout": True
})

def plot_gamma_comparison(files, scale='linear', base=10, fit_curve=True):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Read reference gammas to filter other datasets
    ref_gammas = None
    if os.path.exists('fixed_gamma.csv'):
        ref_gammas = pd.read_csv('fixed_gamma.csv')['gamma_plot'].unique()

    for label, (file_name, marker, color, linestyle) in files.items():
        if os.path.exists(file_name):
            df = pd.read_csv(file_name)
            
            # Filter to keep only the gammas present in the reference file
            if ref_gammas is not None:
                df = df[df['gamma_plot'].isin(ref_gammas)]
            
            x = df['gamma_plot']
            y = df['mean_final_ratio']
            yerr = df['sem_final_ratio']
            
            # If log scale, x=0 won't be plotted since log(0) is undefined. 
            # We filter it out explicitly to avoid matplotlib warnings.
            if scale == 'log':
                mask = x > 0
                x = x[mask]
                y = y[mask]
                yerr = yerr[mask]
            
            # Sort data for interpolation
            sort_idx = np.argsort(x.values)
            x_sorted = x.values[sort_idx]
            y_sorted = y.values[sort_idx]
            yerr_sorted = yerr.values[sort_idx]
            
            # Keep unique x values to avoid spline interpolation errors
            x_u, unique_idx = np.unique(x_sorted, return_index=True)
            y_u = y_sorted[unique_idx]
            yerr_u = yerr_sorted[unique_idx]
            
            if not fit_curve:
                # Simply plot the data directly with its connecting lines
                ax.plot(x_u, y_u, label=label, marker=marker, color=color, 
                        linestyle=linestyle, markersize=6, 
                        markerfacecolor='white', markeredgewidth=1.5)
                # Use shaded area for errors matching the dataset color
                # ax.fill_between(x_u, y_u - yerr_u, y_u + yerr_u, color=color, alpha=0.15, linewidth=0)
                continue
            
            # Plot the raw data as markers (no lines)
            ax.plot(x_u, y_u, label=label, marker=marker, color=color, 
                    linestyle='None', markersize=6, 
                    markerfacecolor='white', markeredgewidth=1.5)
            # Use shaded area for errors matching the dataset color
            # ax.fill_between(x_u, y_u - yerr_u, y_u + yerr_u, color=color, alpha=0.15, linewidth=0)
            
            # Apply an intelligent selection approach: dynamically testing multiple functional forms 
            # (Higher degree polynomial, Logarithmic, Inverse 1/x, and Exponential saturation)
            # and automatically choosing the fit that best captures the data's shape minimizing the MSE.
            if len(x_u) > 3:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    
                    if scale == 'log':
                        x_new = np.logspace(np.log10(x_u.min()), np.log10(x_u.max()), 300)
                    else:
                        x_new = np.linspace(x_u.min(), x_u.max(), 300)

                    best_mse = np.inf
                    y_smooth = None
                    
                    x_fit = np.log10(x_u) if scale == 'log' else x_u
                    x_new_fit = np.log10(x_new) if scale == 'log' else x_new
                    
                    best_mse = np.inf
                    y_smooth = None

                    # 1. Double Gaussian Model (Suitable for two peaks)
                    def double_gaussian(x_val, a1, b1, c1, a2, b2, c2, d):
                        return (a1 * np.exp(-((x_val - b1) / np.clip(c1, 1e-8, None))**2) + 
                                a2 * np.exp(-((x_val - b2) / np.clip(c2, 1e-8, None))**2) + d)
                    
                    x_span = np.max(x_fit) - np.min(x_fit) if len(x_fit) > 1 else 1.0
                    y_span = np.max(y_u) - np.min(y_u) if len(y_u) > 1 else 1.0
                    
                    # Initial guesses assuming peaks at 1/3 and 2/3 of the domain
                    p0_dg = [
                        y_span, np.min(x_fit) + x_span*0.05, x_span*0.1,
                        y_span, np.min(x_fit) + x_span*0.2, x_span*0.2,
                        np.min(y_u)
                    ]
                    
                    # try:
                    #     # Add bounds to prevent the centers and widths from diverging
                    bounds_dg = (
                        [-np.inf, np.min(x_fit) - x_span*0.1, 1e-5, -np.inf, np.min(x_fit) - x_span*0.1, 1e-5, -np.inf],
                        [np.inf, np.max(x_fit) + x_span*0.1, x_span*2, np.inf, np.max(x_fit) + x_span*0.1, x_span*2, np.inf]
                    )
                    popt_dg, _ = curve_fit(double_gaussian, x_fit, y_u, p0=p0_dg, bounds=bounds_dg, maxfev=15000)
                    mse_dg = np.mean((y_u - double_gaussian(x_fit, *popt_dg))**2)
                    best_mse, y_smooth = mse_dg, double_gaussian(x_new_fit, *popt_dg)
                    # except Exception:
                    #     pass

                    # # 2. Double Lorentzian Model (Alternative heavier tails for two peaks)
                    # def double_lorentzian(x_val, a1, b1, c1, a2, b2, c2, d):
                    #     return (a1 / (1 + ((x_val - b1)/np.clip(c1, 1e-8, None))**2) + 
                    #             a2 / (1 + ((x_val - b2)/np.clip(c2, 1e-8, None))**2) + d)
                                
                    # try:
                    #     popt_dl, _ = curve_fit(double_lorentzian, x_fit, y_u, p0=p0_dg, bounds=bounds_dg, maxfev=15000)
                    #     mse_dl = np.mean((y_u - double_lorentzian(x_fit, *popt_dl))**2)
                    #     if mse_dl < best_mse * 0.95:
                    #         best_mse, y_smooth = mse_dl, double_lorentzian(x_new_fit, *popt_dl)
                    # except Exception:
                    #     pass
                        
                    # # 3. Polynomial Model (Degree 6, can represent 2 peaks without getting too unstable)
                    # try:
                    #     deg = min(6, len(x_u) - 1)
                    #     if deg >= 1:
                    #         p_poly = np.polyfit(x_fit, y_u, deg)
                    #         mse_poly = np.mean((y_u - np.polyval(p_poly, x_fit))**2)
                    #         # Slightly penalize polynomial so it is used only if peaks fail
                    #         if mse_poly < best_mse * 0.9:
                    #             best_mse, y_smooth = mse_poly, np.polyval(p_poly, x_new_fit)
                    # except Exception:
                    #     pass

                    # # 4. Smooth Spline Interpolation (Fallback if parametric functions do not fit)
                    # try:
                    #     if len(x_u) > 3:
                    #         spl = make_interp_spline(x_fit, y_u, k=3)
                    #         mse_spline = np.mean((y_u - spl(x_fit))**2)
                    #         if y_smooth is None or mse_spline < best_mse * 0.1:
                    #             best_mse, y_smooth = mse_spline, spl(x_new_fit)
                    # except Exception:
                    #     pass

                ax.plot(x_new, y_smooth, color=color, linestyle=linestyle, linewidth=2)
            else:
                # Fallback if there are too few points for a spline
                ax.plot(x_u, y_u, color=color, linestyle=linestyle, linewidth=2)

        else:
            print(f"File {file_name} not found.")

    # Mathematical formatting for axes labels
    ax.set_xlabel(r'Fixed $\gamma$')
    ax.set_ylabel('Mean Final Ratio')
    
    # Scale setting
    if scale == 'linear':
        ax.set_xlim(left=0, right=18)
        ax.set_ylim(bottom=0.6, top=0.9)
        ax.set_xscale(scale)
    else:
        ax.set_xlim(left=0.2, right=18)
        ax.set_ylim(bottom=0.6, top=0.9)
        ax.set_xscale(scale, base=base)

        
    ax.grid(False) # Turn off grid entirely (common in academic papers)
    ax.legend(loc='best')
    
    # Save the figure to both PDF and high-res PNG for publications
    suffix = f'_{scale}' if fit_curve else f'_{scale}_simple'
    plt.savefig(f'fixed_gamma_comparison{suffix}.pdf')
    plt.savefig(f'fixed_gamma_comparison{suffix}.png', dpi=600)
    print(f"Academic plots ({scale} scale) saved as 'fixed_gamma_comparison{suffix}.pdf' and '.png'.")
    
    plt.show(block=True)

def main():
    # File mapping with distinct symbols, standard colors, and linestyles
    files = {
        'fixed gamma': ('fixed_gamma.csv', '^', '#2ca02c', '-'),
        'ascending gamma': ('schedule_gamma_df_group_2.csv', 's', '#d62728', '-'),

        # '128 Shots': ('fixed_gamma_shot_128.csv', 's', '#d62728', '-'),
        # '256 Shots': ('fixed_gamma_shot_256.csv', '^', '#2ca02c', '-'),
        # '512 Shots': ('fixed_gamma_shot_512.csv', 'v', '#ff7f0e', '-'),
        # '1024 Shots': ('fixed_gamma.csv', 'D', '#9467bd', '-'),
        # '5000 Shots': ('fixed_gamma_shot_5000.csv', 'p', '#8c564b', '-'),
        # '10,000 Shots': ('fixed_gamma_shot_10000.csv', 'o', '#1f77b4', '-'),
        # '20,000 Shots': ('fixed_gamma_shot_20000.csv', 's', '#d62728', '--'),
        # '50,000 Shots': ('fixed_gamma_shot_50000.csv', '^', '#2ca02c', '-.'),
        # '100,000 Shots': ('fixed_gamma_shot_100000.csv', 'v', '#ff7f0e', '-.'),
    }
    
    # Generate the plots with advanced curve fitting
    plot_gamma_comparison(files, scale='linear', fit_curve=True)
    # plot_gamma_comparison(files, scale='log', base=10, fit_curve=True)
    
    # Generate the simple plots directly plotting data (no advanced fit)
    # plot_gamma_comparison(files, scale='linear', fit_curve=False)
    # plot_gamma_comparison(files, scale='log', base=10, fit_curve=False)


if __name__ == "__main__":
    main()
