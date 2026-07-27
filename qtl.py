"""QTL objectives and one complete QAOA optimization procedure."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pennylane as qml
from pennylane import numpy as qnp

from max_cut import MaxCutProblem, build_maxcut_problem, summarize_distribution
from qaoa import make_probability_qnode


@dataclass(frozen=True)
class ObjectiveSpec:
    """Definition of an expectation, fixed-QTL, or scheduled-QTL objective."""

    name: str
    kind: str
    gamma: float | None = None
    gamma_start: float | None = None
    gamma_end: float | None = None
    schedule: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"expectation", "qtl", "qtl_schedule"}:
            raise ValueError(f"Unsupported objective kind: {self.kind}")
        if self.kind == "qtl" and self.gamma is None:
            raise ValueError("A fixed QTL objective requires gamma.")
        if self.kind == "qtl_schedule":
            if self.gamma_start is None or self.gamma_end is None:
                raise ValueError("A scheduled QTL objective requires both endpoints.")
            if self.schedule not in {"linear", "linear_average"}:
                raise ValueError(
                    "Supported QTL schedules are 'linear' and "
                    "'linear_average'."
                )


@dataclass(frozen=True)
class OptimizationSettings:
    """Numerical settings shared by fixed- and ascending-tilt experiments."""

    shots: int = 5000
    steps: int = 100
    learning_rate: float = 0.18
    learning_rate_decay_power: float = 0.35
    learning_rate_decay_offset: float = 6.0
    momentum: float = 0.70
    tilt_learning_rate_penalty: float = 0.01
    gradient_clip: float = 2.5
    tail_window: int = 10
    simulator: str = "default.qubit"


@dataclass(frozen=True)
class RunSpec:
    """Problem, circuit-depth, objective, and initialization identifiers."""

    graph_family: str
    n: int
    depth: int
    graph_seed: int
    objective: ObjectiveSpec
    initialization_id: int
    initialization_seed: int


@dataclass
class RunResult:
    """Iteration history and final summary for one complete optimization."""

    history: pd.DataFrame
    summary: dict[str, object]


def qtl_loss_from_distribution(probabilities, losses, gamma: float | None):
    """Evaluate the tilted loss, with expectation as the zero-tilt limit."""
    probabilities = qml.math.asarray(probabilities)
    losses = qml.math.asarray(losses)
    if gamma is None or abs(gamma) < 1e-12:
        return qml.math.dot(probabilities, losses)
    minimum_loss = qml.math.min(losses)
    shifted_losses = losses - minimum_loss
    tilted_mass = qml.math.sum(
        probabilities * qml.math.exp(-gamma * shifted_losses)
    )
    return minimum_loss - qml.math.log(tilted_mass) / gamma


def gamma_at_step(
    objective: ObjectiveSpec,
    step: int,
    total_steps: int,
) -> float | None:
    """Return the fixed or scheduled tilt used at one optimization step."""
    if objective.kind != "qtl_schedule":
        return objective.gamma
    if total_steps <= 1:
        if objective.schedule == "linear_average":
            return float(
                0.5 * (objective.gamma_start + objective.gamma_end)
            )
        return float(objective.gamma_end)
    if objective.schedule == "linear_average":
        # Treat gamma_end as the exclusive linear endpoint. For example,
        # average gamma=4 with four steps uses 0, 2, 4, 6.
        fraction = step / total_steps
    else:
        fraction = step / (total_steps - 1)
    return float(
        objective.gamma_start
        + fraction * (objective.gamma_end - objective.gamma_start)
    )


def average_gamma_over_steps(
    objective: ObjectiveSpec,
    total_steps: int,
) -> float | None:
    """Return the arithmetic mean of the scheduled optimization tilts."""
    values = [
        gamma_at_step(objective, step, total_steps)
        for step in range(total_steps)
    ]
    finite_values = [value for value in values if value is not None]
    if not finite_values:
        return None
    return float(np.mean(finite_values))


def objective_loss(
    probabilities,
    problem: MaxCutProblem,
    objective: ObjectiveSpec,
    step: int,
    total_steps: int,
):
    """Evaluate the selected objective on a QAOA probability distribution."""
    losses = problem.maximum_cut - qml.math.asarray(problem.cut_values)
    if objective.kind == "expectation":
        return qml.math.dot(probabilities, losses)
    return qtl_loss_from_distribution(
        probabilities,
        losses,
        gamma_at_step(objective, step, total_steps),
    )


def optimize_qaoa(
    circuit,
    problem: MaxCutProblem,
    initial_parameters: np.ndarray,
    objective: ObjectiveSpec,
    settings: OptimizationSettings,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Optimize one QAOA parameter vector and return its full history."""
    parameters = qnp.array(initial_parameters, requires_grad=True)
    momentum_vector = np.zeros_like(np.asarray(initial_parameters, dtype=float))
    best_cut_so_far = -np.inf
    evaluations_per_step = 2 * len(initial_parameters) + 2
    history_rows: list[dict[str, float | int]] = []

    for step in range(settings.steps):

        def loss_function(theta):
            wrapped_parameters = qnp.mod(theta, 2 * np.pi)
            probabilities = circuit(wrapped_parameters)
            return objective_loss(
                probabilities,
                problem,
                objective,
                step,
                settings.steps,
            )

        gradient = np.asarray(qml.grad(loss_function)(parameters), dtype=float)
        raw_gradient_norm = float(np.linalg.norm(gradient))
        if raw_gradient_norm > settings.gradient_clip:
            gradient *= settings.gradient_clip / raw_gradient_norm

        momentum_vector = (
            settings.momentum * momentum_vector
            + (1.0 - settings.momentum) * gradient
        )
        step_learning_rate = settings.learning_rate / (
            (step + 1 + settings.learning_rate_decay_offset)
            ** settings.learning_rate_decay_power
        )
        current_gamma = gamma_at_step(objective, step, settings.steps)
        if current_gamma is not None:
            step_learning_rate /= (
                1.0
                + settings.tilt_learning_rate_penalty * abs(current_gamma)
            )

        updated_parameters = np.mod(
            np.asarray(parameters, dtype=float)
            - step_learning_rate * momentum_vector,
            2 * np.pi,
        )
        parameters = qnp.array(updated_parameters, requires_grad=True)

        probabilities = np.asarray(circuit(parameters), dtype=float)
        metrics = summarize_distribution(
            probabilities,
            problem.cut_values,
            problem.maximum_cut,
        )
        best_cut_so_far = max(best_cut_so_far, metrics["best_cut"])
        history_rows.append(
            {
                "iteration": step + 1,
                "loss": float(loss_function(parameters)),
                "mean_cut": metrics["mean_cut"],
                "mean_ratio": metrics["mean_ratio"],
                "best_ratio_step": metrics["best_ratio"],
                "best_ratio_so_far": best_cut_so_far / problem.maximum_cut,
                "best_cut_so_far": int(best_cut_so_far),
                "optimal_mass": metrics["optimal_mass"],
                "gamma": (
                    np.nan if current_gamma is None else float(current_gamma)
                ),
                "learning_rate": float(step_learning_rate),
                "gradient_norm": float(np.linalg.norm(gradient)),
                "circuit_evaluations": int(
                    (step + 1) * evaluations_per_step
                ),
                "shots_used": int(
                    (step + 1) * evaluations_per_step * settings.shots
                ),
            }
        )

    return np.asarray(parameters, dtype=float), pd.DataFrame(history_rows)


def execute_run(
    run_spec: RunSpec,
    initial_parameters: np.ndarray,
    settings: OptimizationSettings,
    *,
    history_csv: Path | None = None,
    summary_csv: Path | None = None,
) -> RunResult:
    """Execute one complete problem/QAOA/QTL run and optionally save its CSVs."""
    if len(initial_parameters) != 2 * run_spec.depth:
        raise ValueError(
            f"Depth {run_spec.depth} requires {2 * run_spec.depth} parameters; "
            f"received {len(initial_parameters)}."
        )

    problem = build_maxcut_problem(
        run_spec.n,
        run_spec.graph_family,
        run_spec.graph_seed,
    )
    circuit = make_probability_qnode(
        problem.n,
        problem.edges,
        shots=settings.shots,
        device_seed=(
            2024
            + 100 * run_spec.graph_seed
            + 10 * run_spec.depth
            + run_spec.n
        ),
        simulator=settings.simulator,
    )
    final_parameters, history = optimize_qaoa(
        circuit,
        problem,
        initial_parameters,
        run_spec.objective,
        settings,
    )

    metadata = {
        "graph_family": run_spec.graph_family,
        "n": run_spec.n,
        "p": run_spec.depth,
        "seed": run_spec.graph_seed,
        "objective": run_spec.objective.name,
        "init_id": run_spec.initialization_id,
        "init_seed": run_spec.initialization_seed,
    }
    for column, value in metadata.items():
        history[column] = value

    final_row = history.iloc[-1]
    summary: dict[str, object] = {
        **metadata,
        "initial_parameters": np.asarray(
            initial_parameters,
            dtype=float,
        ).tolist(),
        "final_parameters": final_parameters.tolist(),
        "final_mean_ratio": float(final_row["mean_ratio"]),
        "tail_mean_ratio": float(
            history["mean_ratio"]
            .tail(min(settings.tail_window, len(history)))
            .mean()
        ),
        "peak_mean_ratio": float(history["mean_ratio"].max()),
        "final_best_ratio": float(final_row["best_ratio_so_far"]),
        "final_optimal_mass": float(final_row["optimal_mass"]),
        "final_loss": float(final_row["loss"]),
        **{
            f"optimizer_{key}": value
            for key, value in asdict(settings).items()
        },
    }

    result = RunResult(history=history, summary=summary)
    save_run_result(result, history_csv=history_csv, summary_csv=summary_csv)
    return result


def save_run_result(
    result: RunResult,
    *,
    history_csv: Path | None = None,
    summary_csv: Path | None = None,
) -> None:
    """Persist a single-run history and/or summary."""
    if history_csv is not None:
        history_csv = Path(history_csv)
        history_csv.parent.mkdir(parents=True, exist_ok=True)
        result.history.to_csv(history_csv, index=False)
    if summary_csv is not None:
        summary_csv = Path(summary_csv)
        summary_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([result.summary]).to_csv(summary_csv, index=False)
