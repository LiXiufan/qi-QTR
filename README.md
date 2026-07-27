# Curated QTL MaxCut project

This folder is a self-contained, curated copy of the requested work. The
original parent project was not deleted or modified. Historical smoke tests,
superseded CVaR runs, early Figure (b) tests, IDE files, caches, and the virtual
environment are intentionally excluded.

## Retained results

| Study | Code | Included data and figures |
|---|---|---|
| Figure (a), fixed QTL | `fixed_tilt.py` | `data_and_figures/fixed_*` |
| Figure (b), fixed versus ascending QTL | `ascending_tilt.py`, `run_figure_b_incremental.py` | `figure_b_latest/` |
| Figure (c), scale benchmark | `scale_benchmark.py` | `data_and_figures/maxcut_*` |
| Fixed-alpha CVaR versus matched fixed-gamma QTL | `paired_fixed_cvar_qtl.py` | `paired_fixed_cvar_qtl_5000/` |
| Parameter shift versus finite difference | `parameter_shift_rule_comparison/parameter_shift_comparison.py` | the CSV and four figures in the same directory |

The retained Figure (b) is the newest expanded result. It contains 18 fixed
and 18 ascending points at
`0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 2, 3, 4, 8, 10, 16, 24, 36, 50`.
The published views show the requested range `0 <= gamma <= 36`; the gamma-50
rows remain in the CSV as tail-fit provenance. Both the linear and
zero-preserving logarithmic views are included. The two source runs and their
logs are under `figure_b_latest/raw_runs/`.

The retained CVaR/QTL study is the newest strictly paired 5,000-shot result:
350 tasks formed from two objectives, seven matched control values, five
graphs, and five identical initializations. Its shared coordinate is

```text
r = -ln(alpha) / ln(1 / alpha_min) = |gamma| / gamma_max
```

This mapping is a comparison coordinate, not a claim that CVaR and QTL are
physically equivalent. `performance_matching_function.*` contains the two
independently fitted response functions.

## Setup

Python 3.11 or newer is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Regenerate all retained figures from included CSVs

This command performs plotting and curve fitting only; it does not rerun the
expensive simulations:

```powershell
python regenerate_all_figures.py
python validate_package.py
```

Individual plot-only commands are:

```powershell
python fixed_tilt.py
python ascending_tilt.py
python ascending_tilt.py --log-x `
  --output-png figure_b_latest/large_gamma_figure_b_log.png `
  --output-pdf figure_b_latest/large_gamma_figure_b_log.pdf
python scale_benchmark.py
python paired_fixed_cvar_qtl.py --plot-only
python parameter_shift_rule_comparison/parameter_shift_comparison.py --plot-only
```

## Regenerate simulation data

Figure (a) uses its original 12-point grid. Figure (b) has a separate expanded
grid, preventing one panel's defaults from silently changing the other.

```powershell
# Figure (a); the 1,000/10,000-shot files retain gamma=5,6 tail anchors
python run_experiments.py fixed --shots 1000 `
  --gammas "0,0.25,0.4,0.5,0.6,0.75,1,1.5,2,2.5,3,4,5,6" `
  --output-dir data_and_figures
python run_experiments.py fixed --shots 5000 --output-dir data_and_figures
python run_experiments.py fixed --shots 10000 `
  --gammas "0,0.25,0.4,0.5,0.6,0.75,1,1.5,2,2.5,3,4,5,6" `
  --output-dir data_and_figures

# Figure (b), fixed and ascending with a shared grid and initializations
python run_figure_b_incremental.py `
  --gammas "0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,2,3,4,8,10,16,24,36,50"

# Figure (c)
python run_experiments.py scale --shots 5000 --output-dir data_and_figures

# Strictly paired fixed-alpha CVaR versus fixed-gamma QTL
python paired_fixed_cvar_qtl.py

# Parameter-shift versus finite-difference comparison
python parameter_shift_rule_comparison/parameter_shift_comparison.py
```

These production simulations are computationally expensive. Use the CLI
options for fewer shots, steps, initializations, or gamma values for smoke
tests. `run_full_pipeline.py` runs the QTL data jobs concurrently and keeps the
Figure (a) and Figure (b) fixed-gamma outputs in separate locations.

See `REPRODUCIBILITY.md` for the numerical specifications and `CONTENTS.md` for
the retained-file policy.
