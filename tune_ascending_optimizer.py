"""Screen optimizer profiles on a reproducible ascending-tilt benchmark.

This is a deliberately smaller selection experiment than the production sweep.
Every profile sees the same graphs, initial parameters, finite-shot device
seeds, gamma schedules, and optimization budget.  The output records both the
individual benchmark runs and an aggregate ranking so the selected production
profile is auditable.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from experiments import TILT_INITIALIZATION_BASE_SEED
from qaoa import build_initial_points
from qtl import ObjectiveSpec, OptimizationSettings, RunSpec, execute_run


DEFAULT_OUTPUT_CSV = Path(
    "data_and_figures/ascending_optimizer_screening.csv"
)
DEFAULT_RUNS_CSV = Path(
    "data_and_figures/ascending_optimizer_screening_runs.csv"
)
DEFAULT_GRAPH_SEEDS = (0, 1, 2)
DEFAULT_GAMMA_ENDPOINTS = (0.8, 2.0, 4.0)

# The original ascending profile is the reference.  The alternatives vary all
# six requested controls while remaining close enough to be plausible for the
# 100-step production run.
CANDIDATE_PROFILES: dict[str, dict[str, float]] = {
    "old_ascending": {
        "learning_rate": 0.18,
        "learning_rate_decay_power": 0.35,
        "learning_rate_decay_offset": 6.0,
        "momentum": 0.70,
        "tilt_learning_rate_penalty": 0.01,
        "gradient_clip": 2.5,
    },
    "old_submission_default": {
        "learning_rate": 0.18,
        "learning_rate_decay_power": 0.55,
        "learning_rate_decay_offset": 6.0,
        "momentum": 0.70,
        "tilt_learning_rate_penalty": 0.02,
        "gradient_clip": 2.5,
    },
    "slow_decay": {
        "learning_rate": 0.18,
        "learning_rate_decay_power": 0.25,
        "learning_rate_decay_offset": 6.0,
        "momentum": 0.70,
        "tilt_learning_rate_penalty": 0.005,
        "gradient_clip": 2.5,
    },
    "balanced": {
        "learning_rate": 0.22,
        "learning_rate_decay_power": 0.30,
        "learning_rate_decay_offset": 4.0,
        "momentum": 0.75,
        "tilt_learning_rate_penalty": 0.005,
        "gradient_clip": 3.0,
    },
    "high_momentum": {
        "learning_rate": 0.20,
        "learning_rate_decay_power": 0.25,
        "learning_rate_decay_offset": 6.0,
        "momentum": 0.80,
        "tilt_learning_rate_penalty": 0.005,
        "gradient_clip": 3.0,
    },
    "conservative": {
        "learning_rate": 0.15,
        "learning_rate_decay_power": 0.30,
        "learning_rate_decay_offset": 4.0,
        "momentum": 0.70,
        "tilt_learning_rate_penalty": 0.005,
        "gradient_clip": 2.0,
    },
    "aggressive": {
        "learning_rate": 0.25,
        "learning_rate_decay_power": 0.30,
        "learning_rate_decay_offset": 4.0,
        "momentum": 0.80,
        "tilt_learning_rate_penalty": 0.005,
        "gradient_clip": 3.0,
    },
}


def _evaluate_profile(
    profile_name: str,
    profile: dict[str, float],
    *,
    shots: int,
    steps: int,
    graph_seeds: tuple[int, ...],
    gamma_endpoints: tuple[float, ...],
) -> list[dict[str, float | int | str]]:
    """Evaluate one profile on all common benchmark cases."""
    initialization_seed, initial_parameters = build_initial_points(
        [2],
        base_seed=TILT_INITIALIZATION_BASE_SEED,
        number_of_points=1,
    )[2][0]
    settings = OptimizationSettings(
        shots=shots,
        steps=steps,
        tail_window=min(10, steps),
        **profile,
    )

    rows: list[dict[str, float | int | str]] = []
    for graph_seed in graph_seeds:
        for gamma_end in gamma_endpoints:
            objective = ObjectiveSpec(
                name=f"ascending_qtl_0_to_{gamma_end:g}",
                kind="qtl_schedule",
                gamma_start=0.0,
                gamma_end=gamma_end,
                schedule="linear",
            )
            result = execute_run(
                RunSpec(
                    graph_family="erdos_renyi",
                    n=8,
                    depth=2,
                    graph_seed=graph_seed,
                    objective=objective,
                    initialization_id=0,
                    initialization_seed=initialization_seed,
                ),
                initial_parameters,
                settings,
            )
            rows.append(
                {
                    "profile": profile_name,
                    "graph_seed": graph_seed,
                    "gamma_end": gamma_end,
                    "final_mean_ratio": result.summary["final_mean_ratio"],
                    "tail_mean_ratio": result.summary["tail_mean_ratio"],
                    "final_optimal_mass": result.summary[
                        "final_optimal_mass"
                    ],
                }
            )
    return rows


def _standard_error(values: pd.Series) -> float:
    if len(values) <= 1:
        return 0.0
    return float(values.std(ddof=1) / np.sqrt(len(values)))


def screen_profiles(
    *,
    shots: int,
    steps: int,
    workers: int,
    output_csv: Path,
    runs_csv: Path,
    graph_seeds: tuple[int, ...] = DEFAULT_GRAPH_SEEDS,
    gamma_endpoints: tuple[float, ...] = DEFAULT_GAMMA_ENDPOINTS,
) -> pd.DataFrame:
    """Run and rank the candidate profiles, then write both CSV artifacts."""
    all_rows: list[dict[str, float | int | str]] = []
    worker_count = max(1, min(workers, len(CANDIDATE_PROFILES)))
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                _evaluate_profile,
                name,
                profile,
                shots=shots,
                steps=steps,
                graph_seeds=graph_seeds,
                gamma_endpoints=gamma_endpoints,
            ): name
            for name, profile in CANDIDATE_PROFILES.items()
        }
        for future in as_completed(futures):
            name = futures[future]
            rows = future.result()
            all_rows.extend(rows)
            mean_ratio = np.mean(
                [float(row["final_mean_ratio"]) for row in rows]
            )
            print(
                f"completed {name}: mean final ratio={mean_ratio:.8f}",
                flush=True,
            )

    runs = pd.DataFrame(all_rows).sort_values(
        ["profile", "graph_seed", "gamma_end"]
    )
    aggregate = (
        runs.groupby("profile", sort=False)
        .agg(
            mean_final_ratio=("final_mean_ratio", "mean"),
            sem_final_ratio=("final_mean_ratio", _standard_error),
            worst_final_ratio=("final_mean_ratio", "min"),
            mean_tail_ratio=("tail_mean_ratio", "mean"),
            mean_optimal_mass=("final_optimal_mass", "mean"),
        )
        .reset_index()
    )

    for key in next(iter(CANDIDATE_PROFILES.values())):
        aggregate[key] = aggregate["profile"].map(
            lambda name, parameter=key: CANDIDATE_PROFILES[name][parameter]
        )
    aggregate["shots"] = shots
    aggregate["steps"] = steps
    aggregate["graph_seeds"] = ",".join(map(str, graph_seeds))
    aggregate["gamma_endpoints"] = ",".join(map(str, gamma_endpoints))
    aggregate = aggregate.sort_values(
        ["mean_final_ratio", "mean_tail_ratio", "worst_final_ratio"],
        ascending=False,
    ).reset_index(drop=True)
    aggregate.insert(0, "rank", np.arange(1, len(aggregate) + 1))

    output_csv = Path(output_csv)
    runs_csv = Path(runs_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    runs_csv.parent.mkdir(parents=True, exist_ok=True)
    aggregate.to_csv(output_csv, index=False)
    runs.to_csv(runs_csv, index=False)
    return aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--runs-csv", type=Path, default=DEFAULT_RUNS_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ranking = screen_profiles(
        shots=args.shots,
        steps=args.steps,
        workers=args.workers,
        output_csv=args.output_csv,
        runs_csv=args.runs_csv,
    )
    print("\nOptimizer ranking:")
    print(
        ranking[
            [
                "rank",
                "profile",
                "mean_final_ratio",
                "mean_tail_ratio",
                "worst_final_ratio",
            ]
        ].to_string(index=False)
    )
    print(f"\nSelected profile: {ranking.iloc[0]['profile']}")
    print(f"Aggregate CSV: {args.output_csv}")
    print(f"Run-level CSV: {args.runs_csv}")


if __name__ == "__main__":
    main()
