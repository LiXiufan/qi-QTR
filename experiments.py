"""Experiment sweeps and CSV aggregation for the three reported panels."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from pathlib import Path

import numpy as np
import pandas as pd

from qaoa import build_initial_points
from qtl import (
    ObjectiveSpec,
    OptimizationSettings,
    RunResult,
    RunSpec,
    execute_run,
)


DEFAULT_GRAPH_FAMILIES = ["erdos_renyi"]
DEFAULT_GRAPH_SEEDS = [0, 1, 2, 3, 4]
# DEFAULT_FIXED_GAMMAS = [
#     0.0,
#     0.25,
#     0.4,
#     0.5,
#     0.6,
#     0.75,
#     1.0,
#     1.5,
#     2.0,
#     2.5,
#     3.0,
#     4.0,
# ]
# DEFAULT_ASCENDING_FINAL_GAMMAS = [
#     0.0,
#     0.2,
#     0.4,
#     0.6,
#     0.8,
#     1.0,
#     1.2,
#     1.4,
#     1.6,
#     2.0,
#     2.5,
#     3.0,
#     4.0,
# ]
DEFAULT_FIXED_GAMMAS = [
    0.0,
    1.0,
    2.0,
    4.0,
]
DEFAULT_ASCENDING_FINAL_GAMMAS = [
    0.0,
    1.0,
    2.0,
    4.0,
]
DEFAULT_FIXED_SHOTS = [1000, 5000, 10000]
DEFAULT_NUMBER_OF_INITIALIZATIONS = 5

TILT_INITIALIZATION_BASE_SEED = 20260429
# Backward-compatible names for external imports. Both aliases intentionally
# resolve to the single shared Figure (b) initialization sequence.
FIXED_INITIALIZATION_BASE_SEED = TILT_INITIALIZATION_BASE_SEED
SCHEDULE_INITIALIZATION_BASE_SEED = TILT_INITIALIZATION_BASE_SEED

# Best profile in the reproducible ascending-gamma screen.  Fixed and
# ascending sweeps intentionally use this exact same optimizer.
SHARED_TILT_OPTIMIZER = {
    "learning_rate": 0.22,
    "learning_rate_decay_power": 0.30,
    "learning_rate_decay_offset": 4.0,
    "momentum": 0.75,
    "tilt_learning_rate_penalty": 0.005,
    "gradient_clip": 3.0,
}


def build_tilt_initial_points(
    number_of_initializations: int,
) -> dict[int, list[tuple[int, np.ndarray]]]:
    """Return the initial QAOA parameters shared by both tilt sweeps."""
    return build_initial_points(
        [2],
        base_seed=TILT_INITIALIZATION_BASE_SEED,
        number_of_points=number_of_initializations,
        upper_bound=2.0 * np.pi,
    )


def standard_error(values: pd.Series | np.ndarray) -> float:
    """Return the graph-level sample standard error of the mean."""
    array = np.asarray(values, dtype=float)
    if len(array) <= 1:
        return 0.0
    return float(array.std(ddof=1) / np.sqrt(len(array)))


def tilt_optimization_settings(
    shots: int,
    steps: int,
    tail_window: int,
    simulator: str = "default.qubit",
) -> OptimizationSettings:
    """Build the optimizer shared by fixed- and ascending-tilt sweeps."""
    return OptimizationSettings(
        shots=shots,
        steps=steps,
        tail_window=tail_window,
        simulator=simulator,
        **SHARED_TILT_OPTIMIZER,
    )


def scale_optimization_settings(
    shots: int,
    steps: int,
    simulator: str = "default.qubit",
) -> OptimizationSettings:
    """Build the scale-benchmark optimizer."""
    return OptimizationSettings(
        shots=shots,
        steps=steps,
        simulator=simulator,
        learning_rate=0.12,
        learning_rate_decay_power=0.35,
        learning_rate_decay_offset=6.0,
        momentum=0.70,
        tilt_learning_rate_penalty=0.01,
        gradient_clip=2.5,
    )


def _restart_record(
    result: RunResult,
    **experiment_fields: object,
) -> dict[str, object]:
    """Reduce a single-run result to the fields used in sweep CSVs."""
    summary = result.summary
    return {
        "graph_family": summary["graph_family"],
        "n": summary["n"],
        "p": summary["p"],
        "seed": summary["seed"],
        "objective": summary["objective"],
        "init_id": summary["init_id"],
        "init_seed": summary["init_seed"],
        "init_params": summary["initial_parameters"],
        "final_mean_ratio": summary["final_mean_ratio"],
        "tail_mean_ratio": summary["tail_mean_ratio"],
        "peak_mean_ratio": summary["peak_mean_ratio"],
        "final_optimal_mass": summary["final_optimal_mass"],
        **experiment_fields,
    }


def _run_one(
    *,
    graph_family: str,
    n: int,
    depth: int,
    graph_seed: int,
    objective: ObjectiveSpec,
    initialization_id: int,
    initialization_seed: int,
    initial_parameters: np.ndarray,
    settings: OptimizationSettings,
) -> RunResult:
    run_spec = RunSpec(
        graph_family=graph_family,
        n=n,
        depth=depth,
        graph_seed=graph_seed,
        objective=objective,
        initialization_id=initialization_id,
        initialization_seed=initialization_seed,
    )
    return execute_run(run_spec, initial_parameters, settings)


def _execute_restart_task(
    task: dict[str, object],
) -> tuple[int, dict[str, object]]:
    """Execute one independently reproducible sweep task."""
    result = _run_one(
        graph_family=str(task["graph_family"]),
        n=int(task["n"]),
        depth=int(task["depth"]),
        graph_seed=int(task["graph_seed"]),
        objective=task["objective"],
        initialization_id=int(task["initialization_id"]),
        initialization_seed=int(task["initialization_seed"]),
        initial_parameters=np.asarray(
            task["initial_parameters"],
            dtype=float,
        ),
        settings=task["settings"],
    )
    return (
        int(task["task_index"]),
        _restart_record(
            result,
            **dict(task["experiment_fields"]),
        ),
    )


def _collect_restart_records(
    tasks: list[dict[str, object]],
    *,
    workers: int,
) -> list[dict[str, object]]:
    """Execute restart tasks serially or with deterministic process workers."""
    if workers < 1:
        raise ValueError("workers must be at least one.")
    records: dict[int, dict[str, object]] = {}
    if workers == 1:
        for completed, task in enumerate(tasks, start=1):
            print(task["progress_message"], flush=True)
            task_index, record = _execute_restart_task(task)
            records[task_index] = record
            print(
                f"[completed] {completed}/{len(tasks)}",
                flush=True,
            )
    else:
        worker_count = min(workers, len(tasks))
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(_execute_restart_task, task): task
                for task in tasks
            }
            for completed, future in enumerate(
                as_completed(futures),
                start=1,
            ):
                task = futures[future]
                task_index, record = future.result()
                records[task_index] = record
                print(
                    f"[completed] {completed}/{len(tasks)} "
                    f"{task['progress_message']}",
                    flush=True,
                )
    return [records[index] for index in range(len(tasks))]


def _aggregate_tilt_rows(
    restart_frame: pd.DataFrame,
    group_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average restarts per graph, then aggregate graph-level statistics."""
    graph_average = (
        restart_frame.groupby(group_columns)
        .agg(
            num_init_points=("init_id", "nunique"),
            final_mean_ratio=("final_mean_ratio", "mean"),
            tail_mean_ratio=("tail_mean_ratio", "mean"),
            peak_mean_ratio=("peak_mean_ratio", "mean"),
            final_optimal_mass=("final_optimal_mass", "mean"),
        )
        .reset_index()
    )
    grouped_summary = (
        graph_average.groupby("gamma_plot")
        .agg(
            mean_peak_ratio=("peak_mean_ratio", "mean"),
            sem_peak_ratio=("peak_mean_ratio", standard_error),
            mean_tail_ratio=("tail_mean_ratio", "mean"),
            sem_tail_ratio=("tail_mean_ratio", standard_error),
            mean_final_ratio=("final_mean_ratio", "mean"),
            sem_final_ratio=("final_mean_ratio", standard_error),
            mean_optimal_mass=("final_optimal_mass", "mean"),
        )
        .reset_index()
        .sort_values("gamma_plot")
    )
    return graph_average, grouped_summary


def run_fixed_tilt_experiment(
    *,
    shots: int = 5000,
    steps: int = 100,
    gamma_values: list[float] | None = None,
    output_dir: Path = Path("."),
    number_of_initializations: int = DEFAULT_NUMBER_OF_INITIALIZATIONS,
    tail_window: int = 10,
    workers: int = 1,
    simulator: str = "default.qubit",
) -> tuple[Path, Path, Path]:
    """Run and save the fixed-tilt sweep used for Figure (a)."""
    gamma_values = gamma_values or DEFAULT_FIXED_GAMMAS
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = tilt_optimization_settings(
        shots,
        steps,
        tail_window,
        simulator,
    )
    initial_points = build_tilt_initial_points(number_of_initializations)
    tasks: list[dict[str, object]] = []
    started_at = time.time()

    for graph_family in DEFAULT_GRAPH_FAMILIES:
        for graph_seed in DEFAULT_GRAPH_SEEDS:
            for gamma in gamma_values:
                objective = (
                    ObjectiveSpec("Expectation", "expectation")
                    if abs(gamma) < 1e-12
                    else ObjectiveSpec(
                        f"QTL gamma={gamma:.2f}",
                        "qtl",
                        gamma=float(gamma),
                    )
                )
                for initialization_id, (
                    initialization_seed,
                    parameters,
                ) in enumerate(initial_points[2]):
                    tasks.append(
                        {
                            "task_index": len(tasks),
                            "graph_family": graph_family,
                            "n": 8,
                            "depth": 2,
                            "graph_seed": graph_seed,
                            "objective": objective,
                            "initialization_id": initialization_id,
                            "initialization_seed": initialization_seed,
                            "initial_parameters": parameters,
                            "settings": settings,
                            "experiment_fields": {
                                "gamma_plot": float(gamma),
                                "gamma_raw": float(gamma),
                            },
                            "progress_message": (
                                "[fixed] "
                                f"shots={shots} seed={graph_seed} "
                                f"gamma={gamma:.2f} "
                                f"init={initialization_id}"
                            ),
                        }
                    )

    restart_rows = _collect_restart_records(tasks, workers=workers)
    restart_frame = pd.DataFrame(restart_rows)
    graph_average, grouped_summary = _aggregate_tilt_rows(
        restart_frame,
        [
            "graph_family",
            "n",
            "p",
            "seed",
            "gamma_plot",
            "gamma_raw",
        ],
    )
    paths = (
        output_dir / f"fixed_gamma_all_restarts(shot{shots}).csv",
        output_dir / f"fixed_gamma_restart_avg(shot{shots}).csv",
        output_dir / f"fixed_gamma_shot_{shots}.csv",
    )
    restart_frame.to_csv(paths[0], index=False)
    graph_average.to_csv(paths[1], index=False)
    grouped_summary.to_csv(paths[2], index=False)
    print(
        f"Saved {', '.join(path.name for path in paths)} "
        f"in {time.time() - started_at:.2f}s"
    )
    return paths


def run_ascending_tilt_experiment(
    *,
    shots: int = 5000,
    steps: int = 100,
    final_gamma_values: list[float] | None = None,
    output_dir: Path = Path("."),
    number_of_initializations: int = DEFAULT_NUMBER_OF_INITIALIZATIONS,
    tail_window: int = 10,
    workers: int = 1,
    simulator: str = "default.qubit",
) -> tuple[Path, Path, Path]:
    """Run and save the ascending-tilt sweep used for Figure (b)."""
    final_gamma_values = (
        final_gamma_values or DEFAULT_ASCENDING_FINAL_GAMMAS
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = tilt_optimization_settings(
        shots,
        steps,
        tail_window,
        simulator,
    )
    initial_points = build_tilt_initial_points(number_of_initializations)
    tasks: list[dict[str, object]] = []
    started_at = time.time()

    for graph_family in DEFAULT_GRAPH_FAMILIES:
        for graph_seed in DEFAULT_GRAPH_SEEDS:
            for final_gamma in final_gamma_values:
                objective = ObjectiveSpec(
                    f"QTL linear schedule to {final_gamma:.2f}",
                    "qtl_schedule",
                    gamma_start=0.0,
                    gamma_end=float(final_gamma),
                    schedule="linear",
                )
                for initialization_id, (
                    initialization_seed,
                    parameters,
                ) in enumerate(initial_points[2]):
                    tasks.append(
                        {
                            "task_index": len(tasks),
                            "graph_family": graph_family,
                            "n": 8,
                            "depth": 2,
                            "graph_seed": graph_seed,
                            "objective": objective,
                            "initialization_id": initialization_id,
                            "initialization_seed": initialization_seed,
                            "initial_parameters": parameters,
                            "settings": settings,
                            "experiment_fields": {
                                "gamma_plot": float(final_gamma),
                                "gamma_start": 0.0,
                                "gamma_end": float(final_gamma),
                            },
                            "progress_message": (
                                "[ascending] "
                                f"shots={shots} seed={graph_seed} "
                                f"gamma_end={final_gamma:.2f} "
                                f"init={initialization_id}"
                            ),
                        }
                    )

    restart_rows = _collect_restart_records(tasks, workers=workers)
    restart_frame = pd.DataFrame(restart_rows)
    graph_average, grouped_summary = _aggregate_tilt_rows(
        restart_frame,
        [
            "graph_family",
            "n",
            "p",
            "seed",
            "gamma_plot",
            "gamma_start",
            "gamma_end",
        ],
    )
    paths = (
        output_dir / f"schedule_gamma_all_restarts(shots{shots}).csv",
        output_dir / f"schedule_gamma_restart_avg(shots{shots}).csv",
        output_dir / f"schedule_gamma_restart_group(shots{shots})all.csv",
    )
    restart_frame.to_csv(paths[0], index=False)
    graph_average.to_csv(paths[1], index=False)
    grouped_summary.to_csv(paths[2], index=False)
    print(
        f"Saved {', '.join(path.name for path in paths)} "
        f"in {time.time() - started_at:.2f}s"
    )
    return paths


def run_scale_benchmark(
    *,
    shots: int = 5000,
    steps: int = 100,
    output_dir: Path = Path("."),
    number_of_initializations: int = DEFAULT_NUMBER_OF_INITIALIZATIONS,
    simulator: str = "default.qubit",
) -> tuple[Path, Path, Path]:
    """Run and save the size/depth benchmark used for Figure (c)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = scale_optimization_settings(shots, steps, simulator)
    depths = [1, 2, 3]
    initial_points = build_initial_points(
        depths,
        base_seed=SCHEDULE_INITIALIZATION_BASE_SEED,
        number_of_points=number_of_initializations,
    )
    objectives = [
        ObjectiveSpec("Expectation", "expectation"),
        ObjectiveSpec("Fixed QTL", "qtl", gamma=0.4),
        ObjectiveSpec(
            "Ascending QTL",
            "qtl_schedule",
            gamma_start=0.0,
            gamma_end=0.8,
            schedule="linear",
        ),
    ]
    restart_rows: list[dict[str, object]] = []
    started_at = time.time()

    for graph_family in DEFAULT_GRAPH_FAMILIES:
        for n in [8, 10, 12]:
            for depth in depths:
                for graph_seed in DEFAULT_GRAPH_SEEDS:
                    for objective in objectives:
                        for initialization_id, (
                            initialization_seed,
                            parameters,
                        ) in enumerate(initial_points[depth]):
                            print(
                                "[scale] "
                                f"shots={shots} n={n} p={depth} "
                                f"seed={graph_seed} "
                                f"objective={objective.name} "
                                f"init={initialization_id}"
                            )
                            result = _run_one(
                                graph_family=graph_family,
                                n=n,
                                depth=depth,
                                graph_seed=graph_seed,
                                objective=objective,
                                initialization_id=initialization_id,
                                initialization_seed=initialization_seed,
                                initial_parameters=parameters,
                                settings=settings,
                            )
                            restart_rows.append(_restart_record(result))

    restart_frame = pd.DataFrame(restart_rows)
    graph_average = (
        restart_frame.groupby(
            ["graph_family", "n", "p", "seed", "objective"]
        )
        .agg(
            num_init_points=("init_id", "nunique"),
            final_mean_ratio=("final_mean_ratio", "mean"),
            peak_mean_ratio=("peak_mean_ratio", "mean"),
            final_optimal_mass=("final_optimal_mass", "mean"),
        )
        .reset_index()
    )
    grouped_summary = (
        graph_average.groupby(["n", "p", "objective"])
        .agg(
            mean_final_ratio=("final_mean_ratio", "mean"),
            sem_final_ratio=("final_mean_ratio", standard_error),
            mean_peak_ratio=("peak_mean_ratio", "mean"),
            mean_optimal_mass=("final_optimal_mass", "mean"),
            sem_optimal_mass=("final_optimal_mass", standard_error),
        )
        .reset_index()
        .sort_values(["n", "p", "objective"])
    )
    paths = (
        output_dir / f"maxcut_compare_shot{shots}.csv",
        output_dir / f"maxcut_compare_avg_shot{shots}.csv",
        output_dir / f"maxcut_compare_group_shot{shots}.csv",
    )
    restart_frame.to_csv(paths[0], index=False)
    graph_average.to_csv(paths[1], index=False)
    grouped_summary.to_csv(paths[2], index=False)
    print(
        f"Saved {', '.join(path.name for path in paths)} "
        f"in {time.time() - started_at:.2f}s"
    )
    return paths
