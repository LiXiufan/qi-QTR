import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.interpolate import make_interp_spline

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
    if os.path.exists('fixed_gamma_shot_10000.csv'):
        ref_gammas = pd.read_csv('fixed_gamma_shot_10000.csv')['gamma_plot'].unique()

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
                import warnings
                from scipy.optimize import curve_fit
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
                    
                    # 1. Higher Degree Polynomial Model (up to degree 7)
                    deg = min(7, len(x_u) - 1)
                    p_poly = np.polyfit(x_fit, y_u, deg)
                    mse_poly = np.mean((y_u - np.polyval(p_poly, x_fit))**2)
                    best_mse, y_smooth = mse_poly, np.polyval(p_poly, x_new_fit)
                    
                    # 2. Logarithmic Model: y = a + b * ln(x)
                    if np.all(x_u > 0) and scale != 'log':
                        deg_log = min(3, len(x_u) - 1)
                        p_log = np.polyfit(np.log(x_u), y_u, deg_log)
                        mse_log = np.mean((y_u - np.polyval(p_log, np.log(x_u)))**2)
                        # Slightly prefer simpler models
                        if mse_log < best_mse * 0.95:
                            best_mse, y_smooth = mse_log, np.polyval(p_log, np.log(x_new))
                            
                    # 3. Inverse Model (1/x)
                    if np.all(x_u > 0):
                        deg_inv = min(3, len(x_u) - 1)
                        p_inv = np.polyfit(1/x_u, y_u, deg_inv)
                        mse_inv = np.mean((y_u - np.polyval(p_inv, 1/x_u))**2)
                        if mse_inv < best_mse * 0.95:
                            best_mse, y_smooth = mse_inv, np.polyval(p_inv, 1/x_new)

                    # 4. Exponential Model: a - b * exp(-c * x)
                    def exp_func(x_val, a, b, c): 
                        # Use np.clip to prevent overflow warnings
                        return a - b * np.exp(-c * np.clip(x_val, -100, 100))
                        
                    try:
                        p0 = [y_u[-1], y_u[-1] - y_u[0], 1.0 / (np.median(x_fit) + 1e-8)]
                        popt, _ = curve_fit(exp_func, x_fit, y_u, p0=p0, maxfev=5000)
                        mse_exp = np.mean((y_u - exp_func(x_fit, *popt))**2)
                        if mse_exp < best_mse * 0.95:
                            best_mse, y_smooth = mse_exp, exp_func(x_new_fit, *popt)
                    except Exception:
                        pass

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
        # '128 Shots': ('fixed_gamma_shot_128.csv', 's', '#d62728', '-'),
        # '256 Shots': ('fixed_gamma_shot_256.csv', '^', '#2ca02c', '-'),
        '512 Shots': ('fixed_gamma_shot_512.csv', 'v', '#ff7f0e', '-'),
        '1024 Shots': ('fixed_gamma.csv', 'D', '#9467bd', '-'),
        '5000 Shots': ('fixed_gamma_shot_5000.csv', 'p', '#8c564b', '-'),
        '10,000 Shots': ('fixed_gamma_shot_10000.csv', 'o', '#1f77b4', '-'),
        # '20,000 Shots': ('fixed_gamma_shot_20000.csv', 's', '#d62728', '--'),
        # '50,000 Shots': ('fixed_gamma_shot_50000.csv', '^', '#2ca02c', '-.'),
        # '100,000 Shots': ('fixed_gamma_shot_100000.csv', 'v', '#ff7f0e', '-.'),
    }
    
    # Generate the plots with advanced curve fitting
    plot_gamma_comparison(files, scale='linear', fit_curve=True)
    # plot_gamma_comparison(files, scale='log', base=10, fit_curve=True)
    
    # Generate the simple plots directly plotting data (no advanced fit)
    plot_gamma_comparison(files, scale='linear', fit_curve=False)
    # plot_gamma_comparison(files, scale='log', base=10, fit_curve=False)


if __name__ == "__main__":
    main()
