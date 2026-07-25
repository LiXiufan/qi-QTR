# QTL MaxCut reproducibility information

## Architecture and data flow

```text
max_cut.py ─┐
            ├─> qtl.py ─> experiments.py ─> run_experiments.py
qaoa.py ────┘                         │
                                     ├─> fixed summary CSVs
                                     ├─> ascending summary CSV
                                     └─> scale benchmark CSV

fixed summary CSVs ─────────────> fixed_tilt.py ─────> Figure (a)
fixed + ascending summary CSVs ─> ascending_tilt.py ─> Figure (b)
scale benchmark CSV ────────────> scale_benchmark.py ─> Figure (c)
```

`qtl.execute_run` is the unit of execution. It accepts a `RunSpec`, initial QAOA
parameters, and `OptimizationSettings`; it returns the complete iteration
history and final summary. Optional `history_csv` and `summary_csv` arguments
write those single-run outputs directly.

`experiments.py` is responsible only for repeating that unit across graph seeds,
tilt values, depths, and initializations, then aggregating the returned results.

## Compute resources

The code uses PennyLane `default.qubit`. No quantum hardware, GPU, TPU, or paid
cloud accelerator is required. All reported shot counts are finite simulator
shot budgets.

`run_experiments.py` also accepts `--simulator lightning.qubit`. This compiled
backend preserves the circuit, shot count, parameter-shift method, optimizer,
and explicit device seed while reducing simulator time. It is reproducible
across repeated runs with the same backend, but its finite-shot samples need not
be byte-identical to `default.qubit` because the backends use different random
number implementations. The isolated `run_figure_b_test.py` runner selects
`lightning.qubit` by default and records the simulator name in `test.csv`.

Parameter-shift differentiation of `2p` QAOA parameters, followed by objective
and metric evaluation, is recorded as

\[
\text{circuit evaluations per step}=4p+2.
\]

Consequently,

\[
\text{shots per run}=T(4p+2)N_{\mathrm{shot}}.
\]

With `T=100` and five graph seeds/five initializations:

| Experiment | Runs | Circuit evaluations | Simulator shots |
|---|---:|---:|---:|
| Fixed tilt, 1,000 shots, 12 gamma values | 300 | 300,000 | 300,000,000 |
| Fixed tilt, 5,000 shots, 12 gamma values | 300 | 300,000 | 1,500,000,000 |
| Fixed tilt, 10,000 shots, 12 gamma values | 300 | 300,000 | 3,000,000,000 |
| Ascending tilt, 5,000 shots, 13 final gamma values | 325 | 325,000 | 1,625,000,000 |
| Scale benchmark, 5,000 shots, `p=1` | 225 | 135,000 | 675,000,000 |
| Scale benchmark, 5,000 shots, `p=2` | 225 | 225,000 | 1,125,000,000 |
| Scale benchmark, 5,000 shots, `p=3` | 225 | 315,000 | 1,575,000,000 |
| Full pipeline | 2,000 | 1,900,000 | 9,800,000,000 |

Small smoke tests can be run on a laptop by reducing `--steps`,
`--num-init-points`, and the gamma list. Full regeneration is intended for a
multi-core workstation or cluster.

## Common problem construction

- Graph family: connected Erdős–Rényi.
- Edge probability: `min(0.45, 3/(n-1))`.
- Graph seeds: `0,1,2,3,4`.
- Exact maximum cuts: obtained by enumerating all `2^n` bitstrings.
- QAOA circuit: Hadamard initialization followed by `p` Ising-ZZ cost/RX mixer
  layers.
- Gradient method: PennyLane parameter shift.
- Device seed: `2024 + 100*graph_seed + 10*p + n`.

## Fixed- and ascending-tilt optimizer

Both tilt sweeps call the same `tilt_optimization_settings` function and use:

| Setting | Value |
|---|---:|
| Steps | 100 |
| Base learning rate | 0.22 |
| Decay power | 0.30 |
| Decay offset | 4.0 |
| Polyak momentum | 0.75 |
| Tilt learning-rate penalty | 0.005 |
| Gradient clip | 3.0 |
| Tail window | 10 |

The effective learning rate is

\[
\eta_t =
\frac{0.22}
{(t+1+4)^{0.30}(1+0.005|\gamma_t|)}.
\]

This profile was selected by `tune_ascending_optimizer.py` from seven candidates
centered on the old ascending settings. Every candidate used the same `n=8`,
`p=2` problems, initialization, graph seeds `0,1,2`, linear schedules ending at
`gamma=0.8,2,4`, 1,000 shots, and 25 optimization steps. Ranking used mean final
approximation ratio, with mean tail ratio and worst final ratio as tie-breakers.
The selected `balanced` profile scored `0.822176`; the old ascending profile
scored `0.820329`. This is an auditable screening result over the stated
candidate set, not a global-optimality claim.

The aggregate ranking is stored in
`data_and_figures/ascending_optimizer_screening.csv`; all 63 individual
benchmark cases are in
`data_and_figures/ascending_optimizer_screening_runs.csv`.

## Experiment specifications

### Figure (a): fixed tilt

- `n=8`, `p=2`.
- Gamma values:
  `0,0.25,0.4,0.5,0.6,0.75,1,1.5,2,2.5,3,4`.
- Shot budgets: `1000,5000,10000`.
- Shared tilt initialization base seed: `20260429`.
- Five initial parameters per graph and gamma, drawn uniformly from `[0,2pi)`.
- Gamma zero is evaluated as the expectation objective.

### Figure (b): ascending tilt

- `n=8`, `p=2`, 5,000 shots.
- Linear schedules from zero to:
  `0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,2,2.5,3,4`.
- Shared tilt initialization base seed: `20260429`.
- Five initial parameters per graph and endpoint, drawn uniformly from
  `[0,2pi)`.

Fixed and ascending sweeps obtain these points from the same
`build_tilt_initial_points` helper. Thus initialization ID `i` uses the same
seed and exact parameter vector in both datasets, enabling paired comparison.

### Figure (c): scale benchmark

- Sizes: `n=8,10,12`.
- QAOA depths: `p=1,2,3`.
- Shot budget: 5,000.
- Five graph seeds and five initializations.
- Objectives:
  - expectation;
  - fixed QTL with `gamma=0.4`;
  - ascending QTL with a linear `0 -> 0.8` schedule.
- Base learning rate: `0.12`; other decay, momentum, penalty, and clipping
  settings match the common tilt optimizer.

## Aggregation

For each graph/objective setting:

1. Average the five initialization results.
2. Aggregate those graph-level averages across five graph seeds.
3. Calculate SEM as the sample standard deviation divided by the square root of
   the number of graphs.

Fixed and ascending summaries contain:

```text
gamma_plot
mean_peak_ratio, sem_peak_ratio
mean_tail_ratio, sem_tail_ratio
mean_final_ratio, sem_final_ratio
mean_optimal_mass
```

The scale plotting input has one graph-level row per
`n, p, seed, objective`. `scale_benchmark.py` performs the final mean and SEM
aggregation.

## Plotting policy

The figure files contain no optimization code.

- Figure (a) uses the original multi-start robust least-squares matching
  procedure with the shifted exponential-quadratic model
  `d + (a + b*x) exp(-c*x^2)`. Its peak, derivative, curvature, and tail
  constraints are matched to `curve_fitting_poly_tail_summary.csv`.
- Figure (b) fits both series directly from the regenerated data with the same
  robust SEM-weighted cubic model
  `a0 + a1*x + a2*x^2 + a3*x^3`.
- Figure (b) uses `mean_final_ratio` and `sem_final_ratio` for both series, in
  agreement with its y-axis label.
- Figure (a) uses the original solid/dashed/dash-dot line styles, markers,
  colors, framed legend, and fixed axis range.
- Figure (b) uses the original dashed-blue/solid-orange styles and an RMS SEM
  band of constant width around each matched curve.
- Figure (c) uses the original layered difference bars and colored value
  markers.
- Figures (a) and (b) are written as 600-DPI PNGs; Figure (c) is written at
  300 DPI. Every panel also has a vector PDF.

`data_and_figures/curve_fitting_poly_tail_summary.csv` supplies Figure (a)'s
reference matching targets. Regeneration also writes
`data_and_figures/fixed_plot_fit_summary.csv` with the resulting display-model
parameters and `data_and_figures/schedule_gamma_expquad_fit_summary.csv` with
the Figure (b) model parameters.

## Exact commands

From the package root:

```bash
python -m pip install -r requirements.txt
python -m py_compile max_cut.py qaoa.py qtl.py experiments.py run_experiments.py tune_ascending_optimizer.py plotting.py fixed_tilt.py ascending_tilt.py scale_benchmark.py
mkdir -p results
```

Re-run optimizer selection:

```bash
python tune_ascending_optimizer.py
```

Regenerate data:

```bash
python run_experiments.py fixed --shots 1000 --output-dir results
python run_experiments.py fixed --shots 5000 --output-dir results
python run_experiments.py fixed --shots 10000 --output-dir results
python run_experiments.py ascending --shots 5000 --output-dir results
python run_experiments.py scale --shots 5000 --output-dir results
```

Regenerate the included figures without rerunning experiments:

```bash
python fixed_tilt.py
python ascending_tilt.py
python scale_benchmark.py
```
