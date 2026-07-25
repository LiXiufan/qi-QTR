# QTL MaxCut code submission

This package contains the experiment code, aggregated CSV data, and rendered
figures for the QTL MaxCut study. The implementation is separated by scientific
responsibility: MaxCut, QAOA, QTL optimization, experiment sweeps, and plotting.

## Code structure

| File | Responsibility |
|---|---|
| `max_cut.py` | Graph generation, exact MaxCut enumeration, and distribution metrics. |
| `qaoa.py` | QAOA circuit ansatz, finite-shot QNode construction, and parameter initialization. |
| `qtl.py` | QTL objectives, optimizer settings, one complete optimization run, and optional per-run CSV output. |
| `experiments.py` | Fixed-tilt, ascending-tilt, and scale sweeps plus cross-run aggregation. |
| `run_experiments.py` | Command-line interface for the experiment sweeps. |
| `tune_ascending_optimizer.py` | Reproducible ascending-gamma optimizer screening and ranking. |
| `plotting.py` | Shared CSV validation, interpolation, style, and figure-saving helpers. |
| `fixed_tilt.py` | CSV-to-Figure (a) plotting function. |
| `ascending_tilt.py` | CSV-to-Figure (b) plotting function. |
| `scale_benchmark.py` | CSV-to-Figure (c) plotting function. |
| `data_and_figures/` | Included aggregate data and PNG/PDF figure outputs. |

## Shared optimizer

The fixed- and ascending-tilt experiments use exactly the same optimizer:

```text
learning rate                    0.22
learning-rate decay power        0.30
learning-rate decay offset       4.0
Polyak momentum                  0.75
tilt learning-rate penalty       0.005
gradient clipping threshold      3.0
```

At optimization step `t`, the effective learning rate is

```text
0.22 / (t + 1 + 4)^0.30 / (1 + 0.005 |gamma_t|).
```

Both experiment functions construct their settings through
`experiments.tilt_optimization_settings`; there are no separate fixed and
ascending optimizer dictionaries.

The fixed- and ascending-tilt sweeps also call the same
`experiments.build_tilt_initial_points` helper. Consequently, every paired
graph/initialization uses the identical seed and initial QAOA parameter vector
in both sweeps. The shared base seed is `20260429`, and parameters are sampled
from `[0, 2π)`.

This profile was the highest-ranked candidate in the included reproducible
ascending-gamma screen. The comparison is a bounded empirical selection, not a
claim of a global optimum. See `REPRODUCIBILITY.md` for the benchmark and CSV
artifacts.

## Environment setup

Python 3.11 or newer is recommended.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Compile all source files:

```bash
python -m py_compile max_cut.py qaoa.py qtl.py experiments.py run_experiments.py tune_ascending_optimizer.py plotting.py fixed_tilt.py ascending_tilt.py scale_benchmark.py
```

Re-run the optimizer screen:

```bash
python tune_ascending_optimizer.py
```

## Reproduce figures from included CSVs

The plotting functions are deliberately data-only and do not run an optimizer:

```bash
python fixed_tilt.py
python ascending_tilt.py
python scale_benchmark.py
```

By default they read and update the corresponding files in
`data_and_figures/`. Custom paths are available through `--help`.

## Regenerate experiment data

Create a separate result directory:

```bash
mkdir -p results
```

Fixed tilt, Figure (a):

```bash
python run_experiments.py fixed --shots 1000 --output-dir results
python run_experiments.py fixed --shots 5000 --output-dir results
python run_experiments.py fixed --shots 10000 --output-dir results
```

Ascending tilt, Figure (b):

```bash
python run_experiments.py ascending --shots 5000 --output-dir results
```

Use the installed compiled simulator and independent process workers for a
faster finite-shot run:

```bash
python run_experiments.py ascending \
  --shots 5000 \
  --workers 2 \
  --simulator lightning.qubit \
  --output-dir results
```

`default.qubit` remains the command-line default for exact backend provenance.
On the packaged environment, a representative 5,000-shot benchmark was about
3.1 times faster with `lightning.qubit`. Backend random-number implementations
differ, so individual finite-shot samples are not byte-identical, although the
circuit, shot budget, gradient method, optimizer, and seeds are unchanged.

Generate isolated Figure (b) test artifacts with two workers per sweep:

```bash
python run_figure_b_test.py
```

This writes `test.csv`, `test.png`, `test.pdf`, and `test_fit_summary.csv`
without replacing the reported artifacts. For a substantially smaller
screening workload, use `python run_figure_b_test.py --fast`; that preset uses
1,000 shots, 40 steps, and three initializations and therefore is not the full
experiment.

Scale benchmark, Figure (c):

```bash
python run_experiments.py scale --shots 5000 --output-dir results
```

Run the complete pipeline:

```bash
python run_full_pipeline.py --output-dir results
```

This launches the five independent experiment families concurrently and
regenerates all three figures only if every data job succeeds. Per-experiment
logs and a final status manifest are written to `full_run_logs/`.

## Plot regenerated CSVs

```bash
python fixed_tilt.py \
  --shot-1000 results/fixed_gamma_shot_1000.csv \
  --shot-5000 results/fixed_gamma_shot_5000.csv \
  --shot-10000 results/fixed_gamma_shot_10000.csv \
  --output-png results/fixed_plot_results.png \
  --output-pdf results/fixed_plot_results.pdf

python ascending_tilt.py \
  --fixed-csv results/fixed_gamma_shot_5000.csv \
  --ascending-csv "results/schedule_gamma_restart_group(shots5000)all.csv" \
  --output-png results/schedule_gamma_expquad_fit.png \
  --output-pdf results/schedule_gamma_expquad_fit.pdf

python scale_benchmark.py \
  --data-csv results/maxcut_compare_avg_shot5000.csv \
  --output-png results/maxcut_mean_optimal_mass_plot.png \
  --output-pdf results/maxcut_mean_optimal_mass_plot.pdf
```

See `REPRODUCIBILITY.md` for seeds, aggregation rules, objectives, and compute
accounting.
