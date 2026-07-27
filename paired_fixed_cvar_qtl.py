"""Run a strictly paired fixed-CVaR versus fixed-QTL MaxCut experiment.

Every objective/control-point pair uses the same graph, QAOA initialization,
finite-shot device seed, optimizer update rule, hyperparameters, and number of
circuit evaluations. The default logarithmic parameter correspondence is

    r = -log(alpha) / log(10) = |gamma| / 4,
    alpha = exp[-log(10) |gamma| / 4].

The correspondence is a visualization coordinate, not a physical equivalence
between the CVaR and quantum tilted loss functions.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pennylane as qml
from pennylane import numpy as qnp

from experiments import (
    DEFAULT_GRAPH_FAMILIES,
    DEFAULT_GRAPH_SEEDS,
    SHARED_TILT_OPTIMIZER,
    TILT_INITIALIZATION_BASE_SEED,
    standard_error,
)
from max_cut import build_maxcut_problem, summarize_distribution
from plotting import apply_plot_style, padded_limits, plt, save_figure
from qaoa import build_initial_points, make_probability_qnode


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "paired_fixed_cvar_qtl_5000"
DEFAULT_ALPHAS = (1.00, 0.80, 0.65, 0.50, 0.35, 0.20, 0.10)
DEFAULT_GAMMA_MAX = 4.0
TWO_PI = 2.0 * np.pi
DENSE_POINTS = 800


@dataclass(frozen=True)
class PairedSettings:
    """Numerical settings shared exactly by both fixed objectives."""

    shots: int = 5000
    steps: int = 100
    n: int = 8
    depth: int = 2
    number_of_initializations: int = 5
    tail_window: int = 10
    learning_rate: float = SHARED_TILT_OPTIMIZER["learning_rate"]
    learning_rate_decay_power: float = SHARED_TILT_OPTIMIZER[
        "learning_rate_decay_power"
    ]
    learning_rate_decay_offset: float = SHARED_TILT_OPTIMIZER[
        "learning_rate_decay_offset"
    ]
    momentum: float = SHARED_TILT_OPTIMIZER["momentum"]
    gradient_clip: float = SHARED_TILT_OPTIMIZER["gradient_clip"]
    tilt_learning_rate_penalty: float = 0.0
    simulator: str = "lightning.qubit"

    def __post_init__(self) -> None:
        if self.shots < 1 or self.steps < 1:
            raise ValueError("shots and steps must be positive.")
        if self.n < 2 or self.depth < 1:
            raise ValueError("n must be at least two and depth must be positive.")
        if self.number_of_initializations < 1:
            raise ValueError("number_of_initializations must be positive.")
        if self.tail_window < 1:
            raise ValueError("tail_window must be positive.")
        if abs(self.tilt_learning_rate_penalty) > 1e-15:
            raise ValueError(
                "The paired optimizer requires zero objective-specific "
                "learning-rate penalty."
            )


def cvar_loss_from_distribution(probabilities, losses, alpha: float):
    """Evaluate lower-tail CVaR for a discrete finite-shot distribution."""
    if not 0.0 < float(alpha) <= 1.0:
        raise ValueError("CVaR alpha must lie in (0, 1].")
    if np.isclose(float(alpha), 1.0, rtol=0.0, atol=1e-12):
        return qml.math.dot(
            probabilities,
            qml.math.asarray(np.asarray(losses, dtype=float)),
        )
    order = np.argsort(np.asarray(losses, dtype=float), kind="stable")
    ordered_losses = qml.math.asarray(np.asarray(losses, dtype=float)[order])
    ordered_probabilities = probabilities[order]
    cumulative_mass = qml.math.cumsum(ordered_probabilities)
    preceding_mass = qml.math.concatenate(
        (
            qml.math.zeros_like(ordered_probabilities[:1]),
            cumulative_mass[:-1],
        )
    )
    selected_mass = qml.math.minimum(
        ordered_probabilities,
        qml.math.maximum(float(alpha) - preceding_mass, 0.0),
    )
    return qml.math.dot(selected_mass, ordered_losses) / float(alpha)


def qtl_loss_from_distribution(probabilities, losses, gamma: float):
    """Evaluate tilted loss, using expectation exactly when gamma is zero."""
    probabilities = qml.math.asarray(probabilities)
    losses = qml.math.asarray(losses)
    if abs(float(gamma)) < 1e-12:
        return qml.math.dot(probabilities, losses)
    minimum_loss = qml.math.min(losses)
    shifted_losses = losses - minimum_loss
    tilted_mass = qml.math.sum(
        probabilities * qml.math.exp(-float(gamma) * shifted_losses)
    )
    return minimum_loss - qml.math.log(tilted_mass) / float(gamma)


def objective_loss(
    probabilities,
    losses: np.ndarray,
    method: str,
    alpha: float,
    gamma: float,
):
    """Evaluate one of the two paired fixed loss functions."""
    if method == "CVaR":
        return cvar_loss_from_distribution(probabilities, losses, alpha)
    if method == "QTL":
        return qtl_loss_from_distribution(probabilities, losses, gamma)
    raise ValueError(f"Unknown method: {method}")


def control_pairs(
    alphas: tuple[float, ...],
    gamma_max: float,
) -> pd.DataFrame:
    """Build one-to-one alpha/gamma pairs on the shared log coordinate."""
    values = np.asarray(sorted(set(alphas), reverse=True), dtype=float)
    if len(values) < 4 or np.any(values <= 0.0) or values[0] > 1.0:
        raise ValueError("Provide at least four unique alpha values in (0, 1].")
    if not np.isclose(values[0], 1.0):
        raise ValueError("The paired grid must include alpha=1.")
    alpha_min = float(values[-1])
    denominator = np.log(1.0 / alpha_min)
    coordinate = -np.log(values) / denominator
    gamma = gamma_max * coordinate
    return pd.DataFrame(
        {
            "control_index": np.arange(len(values), dtype=int),
            "control_r": coordinate,
            "alpha": values,
            "one_minus_alpha": 1.0 - values,
            "gamma": gamma,
        }
    )


def device_seed(
    graph_seed: int,
    control_index: int,
    initialization_id: int,
) -> int:
    """Return a shot seed shared across the two paired objectives."""
    return (
        910_000
        + 10_000 * int(graph_seed)
        + 100 * int(control_index)
        + int(initialization_id)
    )


def optimize_paired_objective(
    circuit,
    problem,
    initial_parameters: np.ndarray,
    *,
    method: str,
    alpha: float,
    gamma: float,
    settings: PairedSettings,
) -> tuple[dict[str, object], list[dict[str, float | int]]]:
    """Optimize either objective with exactly the same update implementation."""
    parameters = qnp.array(initial_parameters, requires_grad=True)
    momentum_vector = np.zeros_like(initial_parameters, dtype=float)
    losses = problem.maximum_cut - problem.cut_values
    history: list[dict[str, float | int]] = []
    best_ratio_so_far = 0.0

    for step in range(settings.steps):

        def loss_function(theta):
            probabilities = circuit(qnp.mod(theta, TWO_PI))
            return objective_loss(
                probabilities,
                losses,
                method,
                alpha,
                gamma,
            )

        gradient = np.asarray(qml.grad(loss_function)(parameters), dtype=float)
        raw_gradient_norm = float(np.linalg.norm(gradient))
        if raw_gradient_norm > settings.gradient_clip:
            gradient *= settings.gradient_clip / raw_gradient_norm

        momentum_vector = (
            settings.momentum * momentum_vector
            + (1.0 - settings.momentum) * gradient
        )
        learning_rate = settings.learning_rate / (
            (step + 1 + settings.learning_rate_decay_offset)
            ** settings.learning_rate_decay_power
        )
        updated_parameters = np.mod(
            np.asarray(parameters, dtype=float)
            - learning_rate * momentum_vector,
            TWO_PI,
        )
        parameters = qnp.array(updated_parameters, requires_grad=True)

        probabilities = np.asarray(circuit(parameters), dtype=float)
        metrics = summarize_distribution(
            probabilities,
            problem.cut_values,
            problem.maximum_cut,
        )
        numeric_loss = float(
            objective_loss(probabilities, losses, method, alpha, gamma)
        )
        best_ratio_so_far = max(
            best_ratio_so_far,
            float(metrics["best_ratio"]),
        )
        history.append(
            {
                "iteration": step + 1,
                "objective_loss": numeric_loss,
                "mean_ratio": float(metrics["mean_ratio"]),
                "best_ratio_step": float(metrics["best_ratio"]),
                "best_ratio_so_far": best_ratio_so_far,
                "optimal_mass": float(metrics["optimal_mass"]),
                "learning_rate": float(learning_rate),
                "gradient_norm": float(np.linalg.norm(gradient)),
            }
        )

    history_frame = pd.DataFrame(history)
    tail_count = min(settings.tail_window, len(history_frame))
    summary = {
        "initial_parameters": json.dumps(
            np.asarray(initial_parameters, dtype=float).tolist()
        ),
        "final_parameters": json.dumps(
            np.asarray(parameters, dtype=float).tolist()
        ),
        "final_objective_loss": float(
            history_frame["objective_loss"].iloc[-1]
        ),
        "final_mean_ratio": float(history_frame["mean_ratio"].iloc[-1]),
        "tail_mean_ratio": float(
            history_frame["mean_ratio"].tail(tail_count).mean()
        ),
        "peak_mean_ratio": float(history_frame["mean_ratio"].max()),
        "best_ratio_so_far": float(
            history_frame["best_ratio_so_far"].iloc[-1]
        ),
        "final_optimal_mass": float(
            history_frame["optimal_mass"].iloc[-1]
        ),
    }
    return summary, history


def execute_task(
    task: dict[str, object],
) -> tuple[int, dict[str, object], list[dict[str, object]]]:
    """Worker entry point for one fixed-objective optimization."""
    settings = PairedSettings(**dict(task["settings"]))
    problem = build_maxcut_problem(
        settings.n,
        str(task["graph_family"]),
        int(task["graph_seed"]),
    )
    circuit = make_probability_qnode(
        problem.n,
        problem.edges,
        shots=settings.shots,
        device_seed=int(task["device_seed"]),
        simulator=settings.simulator,
    )
    summary, history = optimize_paired_objective(
        circuit,
        problem,
        np.asarray(task["initial_parameters"], dtype=float),
        method=str(task["method"]),
        alpha=float(task["alpha"]),
        gamma=float(task["gamma"]),
        settings=settings,
    )
    common = {
        "task_id": str(task["task_id"]),
        "method": str(task["method"]),
        "control_index": int(task["control_index"]),
        "control_r": float(task["control_r"]),
        "alpha": float(task["alpha"]),
        "one_minus_alpha": 1.0 - float(task["alpha"]),
        "gamma": float(task["gamma"]),
        "graph_family": str(task["graph_family"]),
        "n": settings.n,
        "p": settings.depth,
        "graph_seed": int(task["graph_seed"]),
        "init_id": int(task["initialization_id"]),
        "init_seed": int(task["initialization_seed"]),
        "device_seed": int(task["device_seed"]),
    }
    history_rows = [{**common, **row} for row in history]
    restart_row = {
        **common,
        **summary,
        "shots": settings.shots,
        "steps": settings.steps,
        "tail_window": settings.tail_window,
        "optimizer_learning_rate": settings.learning_rate,
        "optimizer_learning_rate_decay_power": (
            settings.learning_rate_decay_power
        ),
        "optimizer_learning_rate_decay_offset": (
            settings.learning_rate_decay_offset
        ),
        "optimizer_momentum": settings.momentum,
        "optimizer_gradient_clip": settings.gradient_clip,
        "optimizer_tilt_learning_rate_penalty": (
            settings.tilt_learning_rate_penalty
        ),
        "simulator": settings.simulator,
    }
    return int(task["task_index"]), restart_row, history_rows


def build_tasks(
    settings: PairedSettings,
    pairs: pd.DataFrame,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    """Create the complete paired task list and initialization table."""
    initial_points = build_initial_points(
        [settings.depth],
        base_seed=TILT_INITIALIZATION_BASE_SEED,
        number_of_points=settings.number_of_initializations,
        upper_bound=TWO_PI,
    )[settings.depth]
    initialization_rows = [
        {
            "init_id": initialization_id,
            "init_seed": int(seed),
            "initial_parameters": json.dumps(
                np.asarray(parameters, dtype=float).tolist()
            ),
        }
        for initialization_id, (seed, parameters) in enumerate(initial_points)
    ]
    tasks: list[dict[str, object]] = []
    settings_dict = asdict(settings)
    for graph_family in DEFAULT_GRAPH_FAMILIES:
        for graph_seed in DEFAULT_GRAPH_SEEDS:
            for pair in pairs.itertuples(index=False):
                for initialization_id, (
                    initialization_seed,
                    parameters,
                ) in enumerate(initial_points):
                    shared_seed = device_seed(
                        graph_seed,
                        pair.control_index,
                        initialization_id,
                    )
                    for method in ("CVaR", "QTL"):
                        task_id = (
                            f"{method}|r={pair.control_r:.12g}|"
                            f"{graph_family}|g={graph_seed}|"
                            f"init={initialization_id}"
                        )
                        tasks.append(
                            {
                                "task_index": len(tasks),
                                "task_id": task_id,
                                "method": method,
                                "control_index": pair.control_index,
                                "control_r": pair.control_r,
                                "alpha": pair.alpha,
                                "gamma": pair.gamma,
                                "graph_family": graph_family,
                                "graph_seed": graph_seed,
                                "initialization_id": initialization_id,
                                "initialization_seed": initialization_seed,
                                "initial_parameters": np.asarray(
                                    parameters,
                                    dtype=float,
                                ),
                                "device_seed": shared_seed,
                                "settings": settings_dict,
                            }
                        )
    validate_task_pairing(tasks)
    return tasks, pd.DataFrame(initialization_rows)


def validate_task_pairing(tasks: list[dict[str, object]]) -> None:
    """Prove that every CVaR/QTL task pair differs only by its objective."""
    frame = pd.DataFrame(tasks)
    pair_columns = [
        "control_index",
        "graph_family",
        "graph_seed",
        "initialization_id",
    ]
    for key, pair in frame.groupby(pair_columns, sort=False):
        if set(pair["method"]) != {"CVaR", "QTL"} or len(pair) != 2:
            raise ValueError(f"Incomplete objective pair: {key}")
        for column in (
            "control_r",
            "alpha",
            "gamma",
            "initialization_seed",
            "device_seed",
            "settings",
        ):
            values = pair[column].tolist()
            if values[0] != values[1]:
                raise ValueError(f"Pair {key} differs in {column}.")
        parameters = [
            np.asarray(value, dtype=float)
            for value in pair["initial_parameters"]
        ]
        if not np.array_equal(parameters[0], parameters[1]):
            raise ValueError(f"Pair {key} has different initial parameters.")


def execute_tasks(
    tasks: list[dict[str, object]],
    *,
    workers: int,
    restart_partial: Path,
    history_partial: Path,
    resume: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute tasks with deterministic ordering and resumable CSV checkpoints."""
    if workers < 1:
        raise ValueError("workers must be at least one.")
    restart_records: dict[int, dict[str, object]] = {}
    history_records: dict[int, list[dict[str, object]]] = {}
    if resume and restart_partial.exists() and history_partial.exists():
        saved_restart = pd.read_csv(restart_partial)
        saved_history = pd.read_csv(history_partial)
        index_by_id = {
            str(task["task_id"]): int(task["task_index"]) for task in tasks
        }
        for row in saved_restart.to_dict("records"):
            task_id = str(row["task_id"])
            if task_id in index_by_id:
                restart_records[index_by_id[task_id]] = row
        for task_id, group in saved_history.groupby("task_id", sort=False):
            if str(task_id) in index_by_id:
                history_records[index_by_id[str(task_id)]] = (
                    group.to_dict("records")
                )
        common_indices = set(restart_records).intersection(history_records)
        restart_records = {
            index: restart_records[index] for index in common_indices
        }
        history_records = {
            index: history_records[index] for index in common_indices
        }
        print(f"Resuming after {len(common_indices)} completed tasks.", flush=True)

    pending = [
        task
        for task in tasks
        if int(task["task_index"]) not in restart_records
    ]
    started = time.time()

    def checkpoint() -> None:
        ordered_indices = sorted(restart_records)
        pd.DataFrame(
            [restart_records[index] for index in ordered_indices]
        ).to_csv(restart_partial, index=False)
        pd.DataFrame(
            [
                row
                for index in ordered_indices
                for row in history_records[index]
            ]
        ).to_csv(history_partial, index=False)

    if workers == 1:
        iterator = (
            execute_task(task)
            for task in pending
        )
        for completed, result in enumerate(iterator, start=1):
            task_index, restart_row, history_rows = result
            restart_records[task_index] = restart_row
            history_records[task_index] = history_rows
            if completed % 5 == 0 or completed == len(pending):
                checkpoint()
                print(
                    f"[completed] {len(restart_records)}/{len(tasks)} "
                    f"elapsed={time.time()-started:.1f}s",
                    flush=True,
                )
    else:
        with ProcessPoolExecutor(
            max_workers=min(workers, len(pending))
        ) as executor:
            futures = {
                executor.submit(execute_task, task): task for task in pending
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                task_index, restart_row, history_rows = future.result()
                restart_records[task_index] = restart_row
                history_records[task_index] = history_rows
                if completed % 5 == 0 or completed == len(pending):
                    checkpoint()
                    print(
                        f"[completed] {len(restart_records)}/{len(tasks)} "
                        f"elapsed={time.time()-started:.1f}s",
                        flush=True,
                    )

    if len(restart_records) != len(tasks):
        raise RuntimeError("Not all paired tasks completed.")
    ordered_indices = range(len(tasks))
    return (
        pd.DataFrame([restart_records[index] for index in ordered_indices]),
        pd.DataFrame(
            [
                row
                for index in ordered_indices
                for row in history_records[index]
            ]
        ),
    )


def aggregate_results(
    restart: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Aggregate restarts by graph and then calculate graph-level SEM."""
    control_columns = [
        "method",
        "control_index",
        "control_r",
        "alpha",
        "one_minus_alpha",
        "gamma",
    ]
    graph_average = (
        restart.groupby(
            control_columns
            + ["graph_family", "n", "p", "graph_seed"],
            as_index=False,
        )
        .agg(
            num_initializations=("init_id", "nunique"),
            final_mean_ratio=("final_mean_ratio", "mean"),
            tail_mean_ratio=("tail_mean_ratio", "mean"),
            peak_mean_ratio=("peak_mean_ratio", "mean"),
            final_optimal_mass=("final_optimal_mass", "mean"),
            final_objective_loss=("final_objective_loss", "mean"),
        )
    )
    summary = (
        graph_average.groupby(control_columns, as_index=False)
        .agg(
            graph_count=("graph_seed", "nunique"),
            mean_final_ratio=("final_mean_ratio", "mean"),
            sem_final_ratio=("final_mean_ratio", standard_error),
            mean_tail_ratio=("tail_mean_ratio", "mean"),
            sem_tail_ratio=("tail_mean_ratio", standard_error),
            mean_peak_ratio=("peak_mean_ratio", "mean"),
            sem_peak_ratio=("peak_mean_ratio", standard_error),
            mean_optimal_mass=("final_optimal_mass", "mean"),
        )
        .sort_values(["method", "control_index"])
    )
    pair_keys = [
        "control_index",
        "control_r",
        "alpha",
        "one_minus_alpha",
        "gamma",
        "graph_family",
        "graph_seed",
        "init_id",
        "init_seed",
        "device_seed",
    ]
    paired = restart.pivot(
        index=pair_keys,
        columns="method",
        values=[
            "final_mean_ratio",
            "tail_mean_ratio",
            "peak_mean_ratio",
            "final_optimal_mass",
        ],
    )
    paired.columns = [
        f"{metric}_{method.lower()}" for metric, method in paired.columns
    ]
    paired = paired.reset_index()
    for metric in (
        "final_mean_ratio",
        "tail_mean_ratio",
        "peak_mean_ratio",
        "final_optimal_mass",
    ):
        paired[f"{metric}_cvar_minus_qtl"] = (
            paired[f"{metric}_cvar"] - paired[f"{metric}_qtl"]
        )
    return graph_average, summary, paired


def validate_completed_pairing(
    restart: pd.DataFrame,
    history: pd.DataFrame,
    settings: PairedSettings,
) -> None:
    """Validate pairing and exact equality of the expectation-loss control."""
    pair_keys = ["control_index", "graph_seed", "init_id"]
    if not (restart.groupby(pair_keys).size() == 2).all():
        raise ValueError("Completed results contain incomplete method pairs.")
    for _, pair in restart.groupby(pair_keys, sort=False):
        if pair["initial_parameters"].nunique() != 1:
            raise ValueError("Completed pair has different initial parameters.")
        if pair["device_seed"].nunique() != 1:
            raise ValueError("Completed pair has different device seeds.")
    baseline = restart.loc[np.isclose(restart["control_r"], 0.0)]
    pivot = baseline.pivot(
        index=["graph_seed", "init_id"],
        columns="method",
        values=["final_mean_ratio", "tail_mean_ratio", "peak_mean_ratio"],
    )
    maximum_difference = float(
        np.max(np.abs(pivot.xs("CVaR", level=1, axis=1).to_numpy()
                      - pivot.xs("QTL", level=1, axis=1).to_numpy()))
    )
    if maximum_difference > 1e-12:
        raise ValueError(
            "alpha=1 and gamma=0 controls are not identical; "
            f"maximum difference={maximum_difference:.3e}."
        )
    expected_history_rows = len(restart) * settings.steps
    if len(history) != expected_history_rows:
        raise ValueError(
            f"Expected {expected_history_rows} history rows; found {len(history)}."
        )


def asymmetric_peak_model(
    coordinate: np.ndarray,
    peak_value: float,
    peak_coordinate: float,
    left_curvature: float,
    right_curvature: float,
) -> np.ndarray:
    """Evaluate a baseline-peak-tail anchored asymmetric quadratic."""
    coordinate = np.asarray(coordinate, dtype=float)
    displacement = coordinate - peak_coordinate
    curvature = np.where(
        displacement <= 0.0,
        left_curvature,
        right_curvature,
    )
    return peak_value - curvature * displacement**2


def fit_landscape_response(data: pd.DataFrame) -> dict[str, float | str]:
    """Anchor a phenomenological response at baseline, best, and tail."""
    data = data.sort_values("control_r")
    coordinate = data["control_r"].to_numpy(dtype=float)
    ratio = data["mean_final_ratio"].to_numpy(dtype=float)
    sem = np.maximum(data["sem_final_ratio"].to_numpy(dtype=float), 1e-3)
    peak_index = int(np.argmax(ratio))
    peak_coordinate = float(coordinate[peak_index])
    peak_value = float(ratio[peak_index])
    baseline_value = float(ratio[0])
    tail_value = float(ratio[-1])
    left_span = max(peak_coordinate, np.finfo(float).eps)
    right_span = max(1.0 - peak_coordinate, np.finfo(float).eps)
    left_curvature = max(
        0.0,
        (peak_value - baseline_value) / left_span**2,
    )
    right_curvature = max(
        0.0,
        (peak_value - tail_value) / right_span**2,
    )
    fitted = asymmetric_peak_model(
        coordinate,
        peak_value,
        peak_coordinate,
        left_curvature,
        right_curvature,
    )
    return {
        "model": "baseline_peak_tail_anchored_asymmetric_quadratic",
        "formula": (
            "R*=peak; R(r)=R*-c_left(r-r*)^2 for r<=r*; "
            "R(r)=R*-c_right(r-r*)^2 for r>r*"
        ),
        "peak_control_r": peak_coordinate,
        "peak_value": peak_value,
        "baseline_value": baseline_value,
        "tail_value": tail_value,
        "left_curvature": left_curvature,
        "right_curvature": right_curvature,
        "weighted_sse": float(np.sum(((fitted - ratio) / sem) ** 2)),
    }


def publication_style() -> None:
    """Apply restrained journal-style plotting settings."""
    apply_plot_style()
    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.labelsize": 17,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
            "axes.linewidth": 1.15,
            "lines.linewidth": 2.2,
        }
    )


def style_axis(axis) -> None:
    """Apply consistent ticks, spines, and grid to one panel."""
    axis.grid(
        axis="y",
        color="#D0D0D0",
        linestyle=(0, (3, 3)),
        linewidth=0.75,
        alpha=0.75,
    )
    axis.grid(axis="x", visible=False)
    axis.tick_params(which="both", direction="in", top=True, right=True)
    for spine in axis.spines.values():
        spine.set_linewidth(1.15)


def plot_single_method(
    data: pd.DataFrame,
    *,
    method: str,
    x_column: str,
    x_label: str,
    output_jpg: Path,
    output_pdf: Path,
) -> None:
    """Plot one fixed-objective curve with graph-level SEM shading."""
    publication_style()
    data = data.loc[data["method"] == method].sort_values(x_column)
    x = data[x_column].to_numpy(dtype=float)
    mean = data["mean_final_ratio"].to_numpy(dtype=float)
    sem = data["sem_final_ratio"].to_numpy(dtype=float)
    color = "#1F4E79" if method == "CVaR" else "#C4512D"
    figure, axis = plt.subplots(figsize=(6.75, 4.9))
    axis.fill_between(
        x,
        mean - sem,
        mean + sem,
        color=color,
        alpha=0.16,
        linewidth=0.0,
        label="Graph-level SEM",
    )
    axis.plot(
        x,
        mean,
        color=color,
        marker="o" if method == "CVaR" else "s",
        markersize=6.5,
        markerfacecolor="white",
        markeredgewidth=1.3,
        label=f"Fixed {method}",
    )
    y_min, y_max = padded_limits(
        [mean - sem, mean + sem],
        padding_fraction=0.08,
    )
    axis.set(
        xlabel=x_label,
        ylabel="Mean Final Ratio",
        xlim=(float(x.min()), float(x.max())),
        ylim=(y_min, y_max),
    )
    style_axis(axis)
    axis.legend(loc="best")
    save_figure(figure, output_jpg, output_pdf, dpi=600)


def plot_paired_performance(
    summary: pd.DataFrame,
    output_jpg: Path,
    output_pdf: Path,
) -> None:
    """Plot both paired empirical curves on their shared coordinate."""
    publication_style()
    figure, axis = plt.subplots(figsize=(7.1, 5.05))
    specifications = (
        ("CVaR", "#1F4E79", "o", "-"),
        ("QTL", "#C4512D", "s", (0, (5, 3))),
    )
    all_bounds: list[np.ndarray] = []
    for method, color, marker, linestyle in specifications:
        data = summary.loc[summary["method"] == method].sort_values(
            "control_r"
        )
        x = data["control_r"].to_numpy(dtype=float)
        mean = data["mean_final_ratio"].to_numpy(dtype=float)
        sem = data["sem_final_ratio"].to_numpy(dtype=float)
        all_bounds.extend([mean - sem, mean + sem])
        axis.fill_between(
            x,
            mean - sem,
            mean + sem,
            color=color,
            alpha=0.13,
            linewidth=0.0,
        )
        axis.plot(
            x,
            mean,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markersize=6.5,
            markerfacecolor="white",
            markeredgewidth=1.3,
            label=f"Fixed {method}",
        )
    y_min, y_max = padded_limits(all_bounds, padding_fraction=0.08)
    axis.set(
        xlabel=r"Landscape-control coordinate $r=|\gamma|/4$",
        ylabel="Mean Final Ratio",
        xlim=(0.0, 1.0),
        ylim=(y_min, y_max),
    )
    style_axis(axis)
    axis.legend(loc="best")
    save_figure(figure, output_jpg, output_pdf, dpi=600)


def plot_matching_functions(
    summary: pd.DataFrame,
    fits: dict[str, dict[str, float | str]],
    alpha_min: float,
    gamma_max: float,
    output_jpg: Path,
    output_pdf: Path,
) -> None:
    """Plot independently matched response functions under the log/exp map."""
    publication_style()
    coordinate = np.linspace(0.0, 1.0, DENSE_POINTS)
    figure, axis = plt.subplots(figsize=(7.35, 5.45))
    specifications = (
        ("CVaR", "#1F4E79", "o", "-"),
        ("QTL", "#C4512D", "s", (0, (5, 3))),
    )
    all_values: list[np.ndarray] = []
    for method, color, marker, linestyle in specifications:
        fit = fits[method]
        response = asymmetric_peak_model(
            coordinate,
            float(fit["peak_value"]),
            float(fit["peak_control_r"]),
            float(fit["left_curvature"]),
            float(fit["right_curvature"]),
        )
        data = summary.loc[summary["method"] == method].sort_values(
            "control_r"
        )
        all_values.extend(
            [response, data["mean_final_ratio"].to_numpy(dtype=float)]
        )
        axis.plot(
            coordinate,
            response,
            color=color,
            linestyle=linestyle,
            linewidth=2.35,
            label=f"Fixed {method} response",
        )
        axis.scatter(
            data["control_r"],
            data["mean_final_ratio"],
            s=42,
            marker=marker,
            facecolor="white",
            edgecolor=color,
            linewidth=1.25,
            zorder=4,
            label=f"{method} means",
        )
    y_min, y_max = padded_limits(all_values, padding_fraction=0.10)
    axis.set(
        xlabel=rf"Landscape-control coordinate $r=|\gamma|/{gamma_max:g}$",
        ylabel="Mean Final Ratio",
        xlim=(0.0, 1.0),
        ylim=(y_min, y_max),
    )
    style_axis(axis)
    axis.tick_params(top=False)
    axis.legend(
        loc="lower left",
        ncol=2,
        handlelength=2.7,
        columnspacing=1.2,
    )
    secondary_axis = axis.secondary_xaxis(
        "top",
        functions=(
            lambda r: alpha_min**np.asarray(r),
            lambda alpha: (
                -np.log(
                    np.maximum(np.asarray(alpha), np.finfo(float).tiny)
                )
                / np.log(1.0 / alpha_min)
            ),
        ),
    )
    alpha_ticks = np.array([1.0, 0.8, 0.5, 0.2, alpha_min])
    secondary_axis.set_xticks(alpha_ticks)
    secondary_axis.set_xticklabels([f"{value:g}" for value in alpha_ticks])
    secondary_axis.set_xlabel(
        (
            r"Equivalent CVaR tail fraction "
            rf"$\alpha={alpha_min:g}^r="
            rf"\exp[-\ln(1/{alpha_min:g})|\gamma|/{gamma_max:g}]$"
        ),
        labelpad=8,
    )
    secondary_axis.tick_params(direction="in", pad=4)
    save_figure(figure, output_jpg, output_pdf, dpi=600)


def graph_table(settings: PairedSettings) -> pd.DataFrame:
    """Serialize exact graph instances used by the experiment."""
    rows = []
    for family in DEFAULT_GRAPH_FAMILIES:
        for seed in DEFAULT_GRAPH_SEEDS:
            problem = build_maxcut_problem(settings.n, family, seed)
            rows.append(
                {
                    "graph_family": family,
                    "graph_seed": seed,
                    "n": settings.n,
                    "edge_count": len(problem.edges),
                    "edges": json.dumps(problem.edges),
                    "maximum_cut": problem.maximum_cut,
                }
            )
    return pd.DataFrame(rows)


def write_metadata(
    path: Path,
    settings: PairedSettings,
    pairs: pd.DataFrame,
    task_count: int,
) -> None:
    """Write a machine-readable reproducibility manifest."""
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    metadata = {
        "experiment": "strictly_paired_fixed_cvar_vs_fixed_qtl",
        "parameter_correspondence": (
            "r=-ln(alpha)/ln(1/alpha_min)=abs(gamma)/gamma_max"
        ),
        "parameter_correspondence_is_physical_equivalence": False,
        "paired_initial_parameters": True,
        "paired_graphs": True,
        "paired_device_seeds": True,
        "identical_optimizer_implementation": True,
        "objective_specific_learning_rate_penalty": 0.0,
        "expectation_control": "alpha=1 and gamma=0",
        "initialization_base_seed": TILT_INITIALIZATION_BASE_SEED,
        "graph_families": DEFAULT_GRAPH_FAMILIES,
        "graph_seeds": DEFAULT_GRAPH_SEEDS,
        "task_count": task_count,
        "settings": asdict(settings),
        "control_pairs": pairs.to_dict("records"),
        "runner": str(Path(__file__).resolve()),
        "runner_sha256": script_hash,
    }
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def run_experiment(
    *,
    settings: PairedSettings,
    alphas: tuple[float, ...],
    gamma_max: float,
    workers: int,
    output_dir: Path,
    resume: bool,
) -> None:
    """Run, validate, aggregate, serialize, fit, and plot the experiment."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs = control_pairs(alphas, gamma_max)
    tasks, initializations = build_tasks(settings, pairs)
    graphs = graph_table(settings)
    pairs.to_csv(output_dir / "parameter_pairs.csv", index=False)
    initializations.to_csv(output_dir / "initializations.csv", index=False)
    graphs.to_csv(output_dir / "graphs.csv", index=False)
    write_metadata(
        output_dir / "experiment_metadata.json",
        settings,
        pairs,
        len(tasks),
    )

    restart_partial = output_dir / "paired_restart_results.partial.csv"
    history_partial = output_dir / "paired_iteration_history.partial.csv"
    started = time.time()
    restart, history = execute_tasks(
        tasks,
        workers=workers,
        restart_partial=restart_partial,
        history_partial=history_partial,
        resume=resume,
    )
    validate_completed_pairing(restart, history, settings)
    graph_average, summary, paired = aggregate_results(restart)

    fits = {
        method: fit_landscape_response(
            summary.loc[summary["method"] == method]
        )
        for method in ("CVaR", "QTL")
    }
    fit_frame = pd.DataFrame(
        [{"method": method, **fit} for method, fit in fits.items()]
    )

    restart.to_csv(output_dir / "paired_restart_results.csv", index=False)
    history.to_csv(output_dir / "paired_iteration_history.csv", index=False)
    graph_average.to_csv(
        output_dir / "paired_graph_averages.csv",
        index=False,
    )
    summary.to_csv(output_dir / "paired_summary.csv", index=False)
    paired.to_csv(output_dir / "paired_differences.csv", index=False)
    fit_frame.to_csv(output_dir / "fit_parameters.csv", index=False)

    plot_single_method(
        summary,
        method="CVaR",
        x_column="one_minus_alpha",
        x_label=r"CVaR control strength $1-\alpha$",
        output_jpg=output_dir / "fixed_CVaR.jpg",
        output_pdf=output_dir / "fixed_CVaR.pdf",
    )
    plot_single_method(
        summary,
        method="QTL",
        x_column="gamma",
        x_label=r"Tilt parameter $|\gamma|$",
        output_jpg=output_dir / "fixed_QTL.jpg",
        output_pdf=output_dir / "fixed_QTL.pdf",
    )
    plot_paired_performance(
        summary,
        output_dir / "paired_performance.jpg",
        output_dir / "paired_performance.pdf",
    )
    plot_matching_functions(
        summary,
        fits,
        alpha_min=float(pairs["alpha"].min()),
        gamma_max=gamma_max,
        output_jpg=output_dir / "performance_matching_function.jpg",
        output_pdf=output_dir / "performance_matching_function.pdf",
    )

    for partial_path in (restart_partial, history_partial):
        if partial_path.exists():
            partial_path.unlink()
    print(
        f"Saved paired data and eight figures to {output_dir} "
        f"in {time.time()-started:.1f}s.",
        flush=True,
    )


def regenerate_figures_from_results(output_dir: Path) -> None:
    """Regenerate all paired-comparison figures from the saved summary CSV."""
    output_dir = Path(output_dir).resolve()
    summary_path = output_dir / "paired_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing paired summary: {summary_path}")
    summary = pd.read_csv(summary_path)
    required = {
        "method",
        "control_r",
        "alpha",
        "one_minus_alpha",
        "gamma",
        "mean_final_ratio",
        "sem_final_ratio",
    }
    missing = sorted(required.difference(summary.columns))
    if missing:
        raise ValueError(
            f"{summary_path} is missing required columns: {missing}"
        )
    fits = {
        method: fit_landscape_response(
            summary.loc[summary["method"] == method]
        )
        for method in ("CVaR", "QTL")
    }
    pd.DataFrame(
        [{"method": method, **fit} for method, fit in fits.items()]
    ).to_csv(output_dir / "fit_parameters.csv", index=False)
    plot_single_method(
        summary,
        method="CVaR",
        x_column="one_minus_alpha",
        x_label=r"CVaR control strength $1-\alpha$",
        output_jpg=output_dir / "fixed_CVaR.jpg",
        output_pdf=output_dir / "fixed_CVaR.pdf",
    )
    plot_single_method(
        summary,
        method="QTL",
        x_column="gamma",
        x_label=r"Tilt parameter $|\gamma|$",
        output_jpg=output_dir / "fixed_QTL.jpg",
        output_pdf=output_dir / "fixed_QTL.pdf",
    )
    plot_paired_performance(
        summary,
        output_dir / "paired_performance.jpg",
        output_dir / "paired_performance.pdf",
    )
    plot_matching_functions(
        summary,
        fits,
        alpha_min=float(summary["alpha"].min()),
        gamma_max=float(summary["gamma"].abs().max()),
        output_jpg=output_dir / "performance_matching_function.jpg",
        output_pdf=output_dir / "performance_matching_function.pdf",
    )
    print(f"Regenerated paired-comparison figures in {output_dir}", flush=True)


def parse_float_list(value: str) -> tuple[float, ...]:
    """Parse a comma-separated alpha grid."""
    return tuple(
        float(item.strip()) for item in value.split(",") if item.strip()
    )


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, default=PairedSettings.shots)
    parser.add_argument("--steps", type=int, default=PairedSettings.steps)
    parser.add_argument(
        "--num-init-points",
        type=int,
        default=PairedSettings.number_of_initializations,
    )
    parser.add_argument("--tail-window", type=int, default=10)
    parser.add_argument(
        "--alphas",
        type=parse_float_list,
        default=DEFAULT_ALPHAS,
        help="Comma-separated fixed CVaR alpha values.",
    )
    parser.add_argument("--gamma-max", type=float, default=DEFAULT_GAMMA_MAX)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--simulator",
        choices=["default.qubit", "lightning.qubit"],
        default=PairedSettings.simulator,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate figures from paired_summary.csv without simulation.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore compatible partial CSV checkpoints.",
    )
    return parser


def main() -> None:
    """Run the paired experiment from command-line arguments."""
    arguments = build_parser().parse_args()
    if arguments.plot_only:
        regenerate_figures_from_results(arguments.output_dir)
        return
    settings = PairedSettings(
        shots=arguments.shots,
        steps=arguments.steps,
        number_of_initializations=arguments.num_init_points,
        tail_window=arguments.tail_window,
        simulator=arguments.simulator,
    )
    run_experiment(
        settings=settings,
        alphas=tuple(arguments.alphas),
        gamma_max=float(arguments.gamma_max),
        workers=arguments.workers,
        output_dir=arguments.output_dir,
        resume=not arguments.no_resume,
    )


if __name__ == "__main__":
    main()
