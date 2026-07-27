# Reproducibility notes

## Shared implementation

`max_cut.py` constructs seeded MaxCut instances and exact reference cuts.
`qaoa.py` defines the QAOA ansatz, probability QNodes, and initial parameters.
`qtl.py` implements a complete optimization run. `experiments.py` performs
repeated fixed-QTL, ascending-QTL, and scale sweeps. `plotting.py` contains the
shared validation and plotting utilities.

All finite-shot QTL sweeps use PennyLane parameter-shift differentiation.
Fixed and ascending QTL use identical initial parameter vectors and this
optimizer:

| Setting | Value |
|---|---:|
| learning rate | 0.18 |
| decay power | 0.35 |
| decay offset | 6.0 |
| Polyak momentum | 0.70 |
| tilt learning-rate penalty | 0.01 |
| gradient clipping | 2.5 |

At step `t`, the learning rate is

```text
0.18 / (t + 1 + 6)^0.35 / (1 + 0.01 * |gamma_t|)
```

The shared initialization base seed is `20260429`.

## Figures (a)-(c)

Figure (a) uses `n=8`, QAOA depth `p=2`, five graph seeds, five
initializations, 100 steps, shot budgets 1,000/5,000/10,000, and the core
fixed-gamma values `0, 0.25, 0.4, 0.5, 0.6, 0.75, 1, 1.5, 2, 2.5, 3, 4`.
The included 1,000- and 10,000-shot summaries also retain gamma 5 and 6 as
tail anchors; the 5,000-shot summary ends at gamma 4. Its shifted
exponential-quadratic matching targets are retained in
`data_and_figures/curve_fitting_poly_tail_summary.csv`.

Figure (b) uses `n=8`, `p=2`, 5,000 shots, five graph seeds, five
initializations, and 100 steps. For a requested average gamma `g`, the
ascending schedule is

```text
gamma_t = 2 * g * t / T,  t = 0, ..., T - 1
```

The CSV records both requested and realized averages. The latest combined
summary is `figure_b_latest/large_gamma.csv`; the separately loadable series
are `fixed_summary.csv` and `ascending_summary.csv`. The matching model is a
peak-anchored asymmetric rise followed by exponential-plus-linear decay.

Figure (c) uses sizes `n=8,10,12`, depths `p=1,2,3`, 5,000 shots, five graph
seeds, and five initializations. It compares expectation, fixed QTL at
`gamma=0.4`, and ascending QTL with a `0 -> 0.8` linear schedule.

For the QTL panels, aggregation first averages initializations within each
graph and then averages graph-level values. SEM is the sample standard
deviation across graphs divided by the square root of the graph count.

## Strictly paired fixed CVaR versus fixed QTL

The retained production run is in `paired_fixed_cvar_qtl_5000/`.

- `n=8`, `p=2`, 5,000 shots, 100 steps.
- Five graph seeds and five initializations.
- Alpha values: `1, 0.8, 0.65, 0.5, 0.35, 0.2, 0.1`.
- Matched gamma values follow the coordinate recorded in
  `parameter_pairs.csv`, with `gamma_max=4`.
- Each CVaR/QTL pair has the same graph, initialization, device seed,
  optimizer implementation, and shot budget.
- `alpha=1` and `gamma=0` are an exact expectation-objective control.
- The production backend recorded in `experiment_metadata.json` is
  `lightning.qubit`.

The runner path and runner SHA-256 inside `experiment_metadata.json` record the
exact production runner in the parent archive. The curated runner adds only a
plot-only entry point, so its current source hash is intentionally different.

`paired_restart_results.csv` and `paired_iteration_history.csv` retain every
restart and iteration. `paired_graph_averages.csv`, `paired_summary.csv`, and
`paired_differences.csv` retain the aggregation and paired contrasts.

## Parameter-shift versus finite difference

The self-describing
`parameter_shift_rule_comparison/parameter_shift_comparison.csv` records:

- one seeded 6-vertex 3-regular graph (`graph_seed=17`);
- QAOA depth 2 and QTL gamma 2;
- 2,000 shots per circuit evaluation;
- 60 Adam updates with learning rate 0.08;
- central finite-difference step 0.05;
- the same initial vector and optimizer for both methods;
- an analytic parameter-shift L-BFGS-B reference optimum.

Because each QAOA angle is shared by multiple gates, the parameter-shift code
differentiates the gate-level probability vector and then applies the analytic
QTL chain rule. The parameter-error figure uses distance to the closest
symmetry-equivalent reference vector.

## Artifact integrity

`SHA256SUMS.csv` lists SHA-256 hashes and sizes for the curated artifacts. It is
generated after validation and excludes itself.
