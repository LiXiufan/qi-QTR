# QTL MaxCut code submission

This anonymous code package contains the Python scripts, generated summary CSVs, and rendered figures for the QTL MaxCut experiments.

The plotting scripts are configured so that, after extraction, their default inputs point to `data_and_figures/`.

## Quick preflight

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m py_compile run_maxcut_qtl.py plot_fixed_tilt.py plot_ascending_tilt.py plot_comparison.py
```

## Figure-only reproduction from included summaries

These three commands work directly after extraction:

```bash
python plot_fixed_tilt.py
python plot_ascending_tilt.py
python plot_comparison.py
```

To write the regenerated figures into a separate folder:

```bash
mkdir -p results
python plot_fixed_tilt.py --base-dir data_and_figures --reference-shape-path data_and_figures/curve_fitting_poly_tail_summary.csv --output-png results/fixed_plot_results.png --output-pdf results/fixed_plot_results.pdf --summary-path results/fixed_plot_fit_summary.csv
python plot_ascending_tilt.py --fixed-path data_and_figures/fixed_gamma_shot_5000.csv --schedule-path data_and_figures/schedule_gamma_restart_group\(shots5000\)all.csv --curve-fit-summary-path data_and_figures/curve_fitting_poly_tail_summary.csv --output-png results/schedule_gamma_expquad_fit.png --output-pdf results/schedule_gamma_expquad_fit.pdf --output-summary results/schedule_gamma_expquad_fit_summary.csv
python plot_comparison.py --data-path data_and_figures/maxcut_compare_avg_shot5000.csv --output-png results/maxcut_mean_optimal_mass_plot.png --output-pdf results/maxcut_mean_optimal_mass_plot.pdf
```

## Full experiment regeneration

```bash
mkdir -p results
python run_maxcut_qtl.py fixed --shots 1000 --steps 100 --num-init-points 5 --gammas "0,0.25,0.4,0.5,0.6,0.75,1,1.5,2,2.5,3,4" --output-dir results
python run_maxcut_qtl.py fixed --shots 5000 --steps 100 --num-init-points 5 --gammas "0,0.25,0.4,0.5,0.6,0.75,1,1.5,2,2.5,3,4" --output-dir results
python run_maxcut_qtl.py fixed --shots 10000 --steps 100 --num-init-points 5 --gammas "0,0.25,0.4,0.5,0.6,0.75,1,1.5,2,2.5,3,4" --output-dir results
python run_maxcut_qtl.py ascending --shots 5000 --steps 100 --num-init-points 5 --gamma-ends "0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,2,2.5,3,4" --output-dir results
python run_maxcut_qtl.py comparison --shots 5000 --steps 100 --num-init-points 5 --output-dir results
```

The archive already includes `data_and_figures/maxcut_compare_avg_shot5000.csv`, so Figure 2(c) can be reproduced without rerunning the comparison experiment.

See `REPRODUCIBILITY.md` for the asset table, compute accounting, hyperparameters, seeds, and plotting commands.
