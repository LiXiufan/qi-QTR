"""Compare finite-difference and parameter-shift QAOA/QTL optimization.

The default experiment solves one reproducible MaxCut instance twice from the
same QAOA parameters.  Both runs use the same Adam update; only the gradient
estimator changes.  The comparison uses finite-shot probabilities and a
deliberately non-infinitesimal finite-difference step so that estimator noise
and finite-difference bias are visible.  Results are written to one
self-describing CSV and plotted as academic-style PNG/PDF figures.

The QAOA angles in this project are shared by several gates.  Consequently, a
single two-point shift of a *shared* angle is not generally an exact
parameter-shift rule.  ``parameter_shift_gradient`` instead asks PennyLane for
the gate-level probability Jacobian (using ``diff_method="parameter-shift"``)
and then applies the analytic quantum tilted-loss chain rule.  This correctly
accumulates the contributions from every gate that uses an angle.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pennylane as qml
from pennylane import numpy as qnp
from scipy.optimize import minimize


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from max_cut import MaxCutProblem, build_maxcut_problem, summarize_distribution
from plotting import apply_plot_style, plt, save_figure
from qaoa import qaoa_ansatz, random_initial_parameters


DEFAULT_CSV = SCRIPT_DIR / "parameter_shift_comparison.csv"
DEFAULT_PNG = SCRIPT_DIR / "parameter_shift_comparison.png"
DEFAULT_PDF = SCRIPT_DIR / "parameter_shift_comparison.pdf"
DEFAULT_ERROR_PNG = SCRIPT_DIR / "parameter_vector_error.png"
DEFAULT_ERROR_PDF = SCRIPT_DIR / "parameter_vector_error.pdf"
TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class ComparisonSettings:
    """Complete specification of the reproducible comparison experiment."""

    n: int = 6
    graph_family: str = "regular3"
    graph_seed: int = 17
    depth: int = 2
    gamma: float = 2.0
    iterations: int = 60
    learning_rate: float = 0.08
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    finite_difference_step: float = 0.05
    initialization_seed: int = 20260725
    simulator: str = "default.qubit"
    shots: int | None = 2000
    device_seed: int = 314159
    reference_max_iterations: int = 300
    reference_gradient_tolerance: float = 1e-10

    def __post_init__(self) -> None:
        if self.n < 2:
            raise ValueError("The MaxCut graph requires at least two vertices.")
        if self.depth < 1:
            raise ValueError("QAOA depth must be positive.")
        if self.gamma <= 0.0:
            raise ValueError("The tilted-loss parameter gamma must be positive.")
        if self.iterations < 1:
            raise ValueError("The number of iterations must be positive.")
        if self.learning_rate <= 0.0:
            raise ValueError("The learning rate must be positive.")
        if self.finite_difference_step <= 0.0:
            raise ValueError("The finite-difference step must be positive.")
        if self.shots is not None and self.shots < 1:
            raise ValueError("shots must be positive or omitted for exact mode.")
        if self.reference_max_iterations < 1:
            raise ValueError("reference_max_iterations must be positive.")
        if self.reference_gradient_tolerance <= 0.0:
            raise ValueError(
                "reference_gradient_tolerance must be positive."
            )


@dataclass
class AdamState:
    """First- and second-moment state shared by both gradient methods."""

    first_moment: np.ndarray
    second_moment: np.ndarray
    step: int = 0


@dataclass(frozen=True)
class ReferenceOptimum:
    """High-accuracy analytic reference on the initialization's optimum branch."""

    parameters: np.ndarray
    loss: float
    mean_ratio: float
    optimal_mass: float
    iterations: int
    function_evaluations: int
    gradient_norm: float
    converged: bool
    message: str


def make_probability_circuit(
    problem: MaxCutProblem,
    settings: ComparisonSettings,
    *,
    device_seed: int,
):
    """Build a probability QNode with gate-level parameter-shift derivatives."""
    device = qml.device(
        settings.simulator,
        wires=problem.n,
        seed=int(device_seed),
    )

    @qml.qnode(
        device,
        interface="autograd",
        diff_method="parameter-shift",
    )
    def circuit(parameters):
        qaoa_ansatz(parameters, problem.n, problem.edges)
        return qml.probs(wires=range(problem.n))

    if settings.shots is None:
        return circuit
    return qml.set_shots(circuit, shots=settings.shots)


def tilted_loss_and_probability_gradient(
    probabilities: np.ndarray,
    losses: np.ndarray,
    gamma: float,
) -> tuple[float, np.ndarray]:
    r"""Return QTL and its analytic derivative with respect to probabilities.

    For :math:`L_\gamma=-\gamma^{-1}\log\sum_z p_z e^{-\gamma l_z}`,
    shifting all classical losses by their minimum improves numerical
    stability without changing the derivative.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    losses = np.asarray(losses, dtype=float)
    minimum_loss = float(np.min(losses))
    exponential_weights = np.exp(-gamma * (losses - minimum_loss))
    tilted_mass = float(np.dot(probabilities, exponential_weights))
    if tilted_mass <= 0.0:
        raise FloatingPointError("The exponentially tilted mass is not positive.")
    loss = minimum_loss - np.log(tilted_mass) / gamma
    probability_gradient = -exponential_weights / (gamma * tilted_mass)
    return float(loss), probability_gradient


def evaluate_parameters(
    circuit,
    parameters: np.ndarray,
    problem: MaxCutProblem,
    gamma: float,
) -> tuple[float, dict[str, float]]:
    """Evaluate tilted loss and MaxCut metrics for one parameter vector."""
    probabilities = np.asarray(
        circuit(qnp.array(parameters, requires_grad=False)),
        dtype=float,
    )
    classical_losses = problem.maximum_cut - problem.cut_values
    loss, _ = tilted_loss_and_probability_gradient(
        probabilities,
        classical_losses,
        gamma,
    )
    metrics = summarize_distribution(
        probabilities,
        problem.cut_values,
        problem.maximum_cut,
    )
    return loss, metrics


def wrapped_parameter_error(
    parameters: np.ndarray,
    reference_parameters: np.ndarray,
) -> float:
    r"""Return symmetry-adjusted QAOA parameter error in radians.

    With the ansatz used here, cost angles have objective period ``pi``.
    Mixer-angle shifts of ``pi/2`` apply a global bit flip, which leaves every
    MaxCut value and hence QTL invariant.  Wrapping with these componentwise
    periods measures distance to the closest physically equivalent
    representation of the reference vector.
    """
    parameters = np.asarray(parameters, dtype=float)
    reference_parameters = np.asarray(reference_parameters, dtype=float)
    if len(parameters) != len(reference_parameters) or len(parameters) % 2:
        raise ValueError(
            "QAOA parameter and reference vectors must have equal even length."
        )
    depth = len(parameters) // 2
    periods = np.concatenate(
        [
            np.full(depth, np.pi),
            np.full(depth, np.pi / 2.0),
        ]
    )
    angular_difference = (
        parameters - reference_parameters + periods / 2.0
    ) % periods - periods / 2.0
    return float(np.linalg.norm(angular_difference))


def find_reference_optimum(
    problem: MaxCutProblem,
    initial_parameters: np.ndarray,
    settings: ComparisonSettings,
) -> ReferenceOptimum:
    """Find a deterministic reference optimum using analytic probabilities.

    The solve starts from the same parameters as both noisy optimizers.  This
    selects the relevant local branch in a QAOA landscape with symmetry-related
    optima; it is a high-accuracy reference stationary point, not a uniqueness
    claim about the global parameter vector.
    """
    analytic_settings = replace(settings, shots=None)
    circuit = make_probability_circuit(
        problem,
        analytic_settings,
        device_seed=settings.device_seed + 10_000,
    )

    def reference_loss(parameters):
        loss, _ = evaluate_parameters(
            circuit,
            np.mod(parameters, TWO_PI),
            problem,
            settings.gamma,
        )
        return loss

    def reference_gradient(parameters):
        return parameter_shift_gradient(
            circuit,
            np.mod(parameters, TWO_PI),
            problem,
            settings.gamma,
        )

    result = minimize(
        reference_loss,
        x0=np.asarray(initial_parameters, dtype=float),
        jac=reference_gradient,
        method="L-BFGS-B",
        bounds=[(0.0, TWO_PI)] * len(initial_parameters),
        options={
            "maxiter": settings.reference_max_iterations,
            "ftol": 1e-14,
            "gtol": settings.reference_gradient_tolerance,
            "maxls": 40,
        },
    )
    optimal_parameters = np.mod(np.asarray(result.x, dtype=float), TWO_PI)
    optimal_loss, optimal_metrics = evaluate_parameters(
        circuit,
        optimal_parameters,
        problem,
        settings.gamma,
    )
    gradient = reference_gradient(optimal_parameters)
    return ReferenceOptimum(
        parameters=optimal_parameters,
        loss=float(optimal_loss),
        mean_ratio=float(optimal_metrics["mean_ratio"]),
        optimal_mass=float(optimal_metrics["optimal_mass"]),
        iterations=int(result.nit),
        function_evaluations=int(result.nfev),
        gradient_norm=float(np.linalg.norm(gradient)),
        converged=bool(result.success),
        message=str(result.message),
    )


def finite_difference_gradient(
    circuit,
    parameters: np.ndarray,
    problem: MaxCutProblem,
    gamma: float,
    difference_step: float,
) -> np.ndarray:
    """Estimate the QTL gradient with a central finite difference."""
    gradient = np.empty_like(parameters, dtype=float)
    for parameter_index in range(len(parameters)):
        displacement = np.zeros_like(parameters, dtype=float)
        displacement[parameter_index] = difference_step
        loss_plus, _ = evaluate_parameters(
            circuit,
            np.mod(parameters + displacement, TWO_PI),
            problem,
            gamma,
        )
        loss_minus, _ = evaluate_parameters(
            circuit,
            np.mod(parameters - displacement, TWO_PI),
            problem,
            gamma,
        )
        gradient[parameter_index] = (
            loss_plus - loss_minus
        ) / (2.0 * difference_step)
    return gradient


def parameter_shift_gradient(
    circuit,
    parameters: np.ndarray,
    problem: MaxCutProblem,
    gamma: float,
) -> np.ndarray:
    """Calculate the exact gate-level parameter-shift QTL gradient.

    The circuit Jacobian has shape ``(2**n, 2*depth)``.  Multiplication by
    ``d(QTL)/d(probabilities)`` implements the nonlinear classical chain rule.
    PennyLane expands shared QAOA angles to their individual gate occurrences
    before applying the quantum parameter-shift rule.
    """
    differentiable_parameters = qnp.array(parameters, requires_grad=True)
    probabilities = np.asarray(
        circuit(differentiable_parameters),
        dtype=float,
    )
    probability_jacobian = np.asarray(
        qml.jacobian(circuit)(differentiable_parameters),
        dtype=float,
    )
    classical_losses = problem.maximum_cut - problem.cut_values
    _, loss_probability_gradient = tilted_loss_and_probability_gradient(
        probabilities,
        classical_losses,
        gamma,
    )
    return np.asarray(
        loss_probability_gradient @ probability_jacobian,
        dtype=float,
    )


def adam_step(
    parameters: np.ndarray,
    gradient: np.ndarray,
    state: AdamState,
    settings: ComparisonSettings,
) -> tuple[np.ndarray, AdamState]:
    """Apply one wrapped-angle Adam update."""
    next_step = state.step + 1
    first_moment = (
        settings.adam_beta1 * state.first_moment
        + (1.0 - settings.adam_beta1) * gradient
    )
    second_moment = (
        settings.adam_beta2 * state.second_moment
        + (1.0 - settings.adam_beta2) * np.square(gradient)
    )
    corrected_first = first_moment / (
        1.0 - settings.adam_beta1**next_step
    )
    corrected_second = second_moment / (
        1.0 - settings.adam_beta2**next_step
    )
    updated_parameters = np.mod(
        parameters
        - settings.learning_rate
        * corrected_first
        / (np.sqrt(corrected_second) + settings.adam_epsilon),
        TWO_PI,
    )
    return updated_parameters, AdamState(
        first_moment=first_moment,
        second_moment=second_moment,
        step=next_step,
    )


def finite_difference_update(
    circuit,
    parameters: np.ndarray,
    state: AdamState,
    problem: MaxCutProblem,
    settings: ComparisonSettings,
) -> tuple[np.ndarray, AdamState, np.ndarray]:
    """Update QAOA parameters using a central finite-difference gradient."""
    gradient = finite_difference_gradient(
        circuit,
        parameters,
        problem,
        settings.gamma,
        settings.finite_difference_step,
    )
    updated_parameters, updated_state = adam_step(
        parameters,
        gradient,
        state,
        settings,
    )
    return updated_parameters, updated_state, gradient


def parameter_shift_update(
    circuit,
    parameters: np.ndarray,
    state: AdamState,
    problem: MaxCutProblem,
    settings: ComparisonSettings,
) -> tuple[np.ndarray, AdamState, np.ndarray]:
    """Update QAOA parameters using the gate-level parameter-shift rule."""
    gradient = parameter_shift_gradient(
        circuit,
        parameters,
        problem,
        settings.gamma,
    )
    updated_parameters, updated_state = adam_step(
        parameters,
        gradient,
        state,
        settings,
    )
    return updated_parameters, updated_state, gradient


def optimization_history(
    method: str,
    update_function: Callable,
    circuit,
    initial_parameters: np.ndarray,
    reference_parameters: np.ndarray,
    problem: MaxCutProblem,
    settings: ComparisonSettings,
) -> pd.DataFrame:
    """Run one optimizer and return its complete iteration history."""
    parameters = np.asarray(initial_parameters, dtype=float).copy()
    state = AdamState(
        first_moment=np.zeros_like(parameters),
        second_moment=np.zeros_like(parameters),
    )
    rows: list[dict[str, object]] = []

    initial_loss, initial_metrics = evaluate_parameters(
        circuit,
        parameters,
        problem,
        settings.gamma,
    )
    rows.append(
        iteration_row(
            method,
            0,
            initial_loss,
            initial_metrics,
            parameters,
            reference_parameters,
            gradient=None,
        )
    )

    for iteration in range(1, settings.iterations + 1):
        parameters, state, gradient = update_function(
            circuit,
            parameters,
            state,
            problem,
            settings,
        )
        loss, metrics = evaluate_parameters(
            circuit,
            parameters,
            problem,
            settings.gamma,
        )
        rows.append(
            iteration_row(
                method,
                iteration,
                loss,
                metrics,
                parameters,
                reference_parameters,
                gradient,
            )
        )
    return pd.DataFrame(rows)


def iteration_row(
    method: str,
    iteration: int,
    loss: float,
    metrics: dict[str, float],
    parameters: np.ndarray,
    reference_parameters: np.ndarray,
    gradient: np.ndarray | None,
) -> dict[str, object]:
    """Create one trajectory row for the comparison CSV."""
    return {
        "record_type": "iteration",
        "method": method,
        "iteration": iteration,
        "loss": float(loss),
        "mean_final_ratio": float(metrics["mean_ratio"]),
        "mean_cut": float(metrics["mean_cut"]),
        "best_ratio": float(metrics["best_ratio"]),
        "optimal_mass": float(metrics["optimal_mass"]),
        "parameter_error_l2": wrapped_parameter_error(
            parameters,
            reference_parameters,
        ),
        "gradient_norm": (
            np.nan
            if gradient is None
            else float(np.linalg.norm(gradient))
        ),
        "parameters": json.dumps(np.asarray(parameters, dtype=float).tolist()),
    }


def problem_metadata(
    problem: MaxCutProblem,
    settings: ComparisonSettings,
    initial_parameters: np.ndarray,
    reference_optimum: ReferenceOptimum,
) -> dict[str, object]:
    """Build CSV-safe problem, circuit, objective, and optimizer metadata."""
    settings_data = asdict(settings)
    for redundant_key in ("n", "graph_family", "depth"):
        settings_data.pop(redundant_key)
    settings_data["shots"] = (
        "analytic" if settings.shots is None else int(settings.shots)
    )
    return {
        "graph_family": problem.graph_family,
        "n_vertices": problem.n,
        "n_edges": len(problem.edges),
        "edges": json.dumps([list(edge) for edge in problem.edges]),
        "maximum_cut": problem.maximum_cut,
        "qaoa_depth": settings.depth,
        "n_parameters": 2 * settings.depth,
        "objective": "quantum_tilted_loss",
        "optimizer": "Adam",
        "initial_parameters": json.dumps(initial_parameters.tolist()),
        "reference_optimum_method": (
            "analytic_L-BFGS-B_parameter-shift_from_shared_initialization"
        ),
        "optimal_parameters": json.dumps(
            reference_optimum.parameters.tolist()
        ),
        "optimal_loss": reference_optimum.loss,
        "optimal_mean_ratio": reference_optimum.mean_ratio,
        "optimal_mass_at_reference": reference_optimum.optimal_mass,
        "reference_iterations": reference_optimum.iterations,
        "reference_function_evaluations": (
            reference_optimum.function_evaluations
        ),
        "reference_gradient_norm": reference_optimum.gradient_norm,
        "reference_converged": reference_optimum.converged,
        "reference_message": reference_optimum.message,
        "parameter_error_definition": (
            "L2 distance to closest symmetry-equivalent optimal vector"
        ),
        "parameter_error_cost_period": np.pi,
        "parameter_error_mixer_period": np.pi / 2.0,
        **settings_data,
    }


def build_results_frame(
    finite_history: pd.DataFrame,
    shift_history: pd.DataFrame,
    metadata: dict[str, object],
) -> pd.DataFrame:
    """Combine the specification and both histories in one CSV table."""
    specification_row: dict[str, object] = {
        "record_type": "problem_specification",
        "method": "shared",
        **metadata,
    }
    histories = pd.concat(
        [finite_history, shift_history],
        ignore_index=True,
    )
    for column, value in metadata.items():
        histories[column] = value
    combined = pd.concat(
        [pd.DataFrame([specification_row]), histories],
        ignore_index=True,
        sort=False,
    )
    combined["iteration"] = combined["iteration"].astype("Int64")
    return combined


def plot_comparison(
    results: pd.DataFrame,
    output_png: Path,
    output_pdf: Path,
) -> None:
    """Plot mean-ratio trajectories with a tilted-loss inset."""
    apply_plot_style()
    plt.rcParams.update(
        {
            "axes.labelsize": 17,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
            "axes.linewidth": 1.1,
        }
    )
    trajectories = results.loc[results["record_type"] == "iteration"]
    methods = [
        (
            "finite_difference",
            "Finite difference",
            "#2C5F8A",
            "--",
            "o",
            2.2,
            6.4,
        ),
        (
            "parameter_shift",
            "Parameter shift",
            "#C4512D",
            "-",
            "s",
            1.8,
            5.2,
        ),
    ]

    figure, axis = plt.subplots(figsize=(8.2, 5.6))
    marker_interval = max(1, int(trajectories["iteration"].max()) // 10)
    for (
        method,
        label,
        color,
        line_style,
        marker,
        line_width,
        marker_size,
    ) in methods:
        series = trajectories.loc[
            trajectories["method"] == method
        ].sort_values("iteration")
        axis.plot(
            series["iteration"],
            series["mean_final_ratio"],
            label=label,
            color=color,
            linestyle=line_style,
            marker=marker,
            markevery=marker_interval,
            markersize=marker_size,
            markerfacecolor="white",
            markeredgewidth=1.1,
            linewidth=line_width,
            zorder=3,
        )

    axis.set_xlabel("Optimization iteration")
    axis.set_ylabel("Mean final ratio")
    axis.grid(
        axis="y",
        color="#D9D9D9",
        linestyle=(0, (3, 3)),
        linewidth=0.75,
        alpha=0.75,
    )
    axis.minorticks_on()
    axis.tick_params(which="both", direction="in", top=True, right=True)
    axis.tick_params(which="minor", length=3, width=0.8)
    for spine in axis.spines.values():
        spine.set_linewidth(1.1)
    axis.text(
        0.015,
        0.975,
        "(a)",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=14,
        fontweight="bold",
    )
    axis.legend(
        loc="lower right",
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        facecolor="white",
        edgecolor="#A8A8A8",
        borderpad=0.55,
        handlelength=2.6,
    )

    # This lower-left placement occupies the empty region below the converged
    # mean-ratio trajectories and therefore does not obscure the main curves.
    inset = axis.inset_axes([0.14, 0.13, 0.40, 0.33])
    for (
        method,
        _,
        color,
        line_style,
        _,
        line_width,
        _,
    ) in methods:
        series = trajectories.loc[
            trajectories["method"] == method
        ].sort_values("iteration")
        inset.plot(
            series["iteration"],
            series["loss"],
            color=color,
            linestyle=line_style,
            linewidth=max(1.2, 0.78 * line_width),
        )
    inset.set_xlabel("Iteration", fontsize=9)
    inset.set_ylabel(r"QTL loss $\mathcal{L}_{\gamma}$", fontsize=9)
    inset.set_facecolor("white")
    inset.patch.set_alpha(0.97)
    inset.tick_params(
        axis="both",
        which="major",
        labelsize=8,
        direction="in",
        top=True,
        right=True,
    )
    inset.grid(
        axis="y",
        color="#E3E3E3",
        linestyle=(0, (2, 2)),
        linewidth=0.6,
    )
    for spine in inset.spines.values():
        spine.set_linewidth(0.9)
    inset.set_title("Tilted-loss trajectory", fontsize=9.5, pad=3)
    inset.text(
        0.035,
        0.94,
        "(b)",
        transform=inset.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
    )

    save_figure(figure, output_png, output_pdf, dpi=600)


def plot_parameter_error(
    results: pd.DataFrame,
    output_png: Path,
    output_pdf: Path,
) -> None:
    """Plot wrapped distance from each iterate to the reference optimum."""
    apply_plot_style()
    plt.rcParams.update(
        {
            "axes.labelsize": 17,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
            "axes.linewidth": 1.1,
        }
    )
    trajectories = results.loc[results["record_type"] == "iteration"]
    methods = [
        (
            "finite_difference",
            "Finite difference",
            "#2C5F8A",
            "--",
            "o",
            2.3,
            6.5,
        ),
        (
            "parameter_shift",
            "Parameter shift",
            "#C4512D",
            "-",
            "s",
            1.8,
            5.2,
        ),
    ]
    figure, axis = plt.subplots(figsize=(8.2, 5.8))
    marker_interval = max(1, int(trajectories["iteration"].max()) // 10)
    for (
        method,
        label,
        color,
        line_style,
        marker,
        line_width,
        marker_size,
    ) in methods:
        series = trajectories.loc[
            trajectories["method"] == method
        ].sort_values("iteration")
        axis.plot(
            series["iteration"],
            series["parameter_error_l2"],
            label=label,
            color=color,
            linestyle=line_style,
            marker=marker,
            markevery=marker_interval,
            markersize=marker_size,
            markerfacecolor="white",
            markeredgewidth=1.1,
            linewidth=line_width,
            zorder=3,
        )

    axis.axhline(
        0.0,
        color="#555555",
        linestyle=":",
        linewidth=1.0,
        zorder=1,
    )
    axis.set_xlabel("Optimization iteration")
    axis.set_ylabel(
        r"Symmetry-adjusted parameter error "
        r"$\|\boldsymbol{\theta}-\boldsymbol{\theta}^{*}\|_{2}$ (rad)"
    )
    axis.grid(
        True,
        color="#D9D9D9",
        linestyle=(0, (3, 3)),
        linewidth=0.8,
        alpha=0.85,
    )
    axis.tick_params(which="both", direction="in", top=True, right=True)
    for spine in axis.spines.values():
        spine.set_linewidth(1.1)
    axis.legend(
        loc="best",
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#B8B8B8",
    )
    save_figure(figure, output_png, output_pdf, dpi=600)


def run_comparison(
    settings: ComparisonSettings,
    csv_path: Path = DEFAULT_CSV,
    png_path: Path = DEFAULT_PNG,
    pdf_path: Path = DEFAULT_PDF,
    error_png_path: Path = DEFAULT_ERROR_PNG,
    error_pdf_path: Path = DEFAULT_ERROR_PDF,
) -> pd.DataFrame:
    """Solve the shared instance, save the CSV, and render the figure."""
    problem = build_maxcut_problem(
        settings.n,
        settings.graph_family,
        settings.graph_seed,
    )
    initial_parameters = random_initial_parameters(
        settings.depth,
        settings.initialization_seed,
    )
    reference_optimum = find_reference_optimum(
        problem,
        initial_parameters,
        settings,
    )
    finite_circuit = make_probability_circuit(
        problem,
        settings,
        device_seed=settings.device_seed,
    )
    shift_circuit = make_probability_circuit(
        problem,
        settings,
        device_seed=settings.device_seed,
    )

    finite_history = optimization_history(
        "finite_difference",
        finite_difference_update,
        finite_circuit,
        initial_parameters,
        reference_optimum.parameters,
        problem,
        settings,
    )
    shift_history = optimization_history(
        "parameter_shift",
        parameter_shift_update,
        shift_circuit,
        initial_parameters,
        reference_optimum.parameters,
        problem,
        settings,
    )
    metadata = problem_metadata(
        problem,
        settings,
        initial_parameters,
        reference_optimum,
    )
    results = build_results_frame(
        finite_history,
        shift_history,
        metadata,
    )

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(csv_path, index=False)
    plot_comparison(results, png_path, pdf_path)
    plot_parameter_error(results, error_png_path, error_pdf_path)
    return results


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare central finite differences with the gate-level "
            "parameter-shift rule for QAOA quantum tilted loss."
        )
    )
    parser.add_argument("--n", type=int, default=ComparisonSettings.n)
    parser.add_argument(
        "--graph-family",
        choices=["regular3", "erdos_renyi"],
        default=ComparisonSettings.graph_family,
    )
    parser.add_argument(
        "--graph-seed",
        type=int,
        default=ComparisonSettings.graph_seed,
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=ComparisonSettings.depth,
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=ComparisonSettings.gamma,
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=ComparisonSettings.iterations,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=ComparisonSettings.learning_rate,
    )
    parser.add_argument(
        "--finite-difference-step",
        type=float,
        default=ComparisonSettings.finite_difference_step,
    )
    parser.add_argument(
        "--initialization-seed",
        type=int,
        default=ComparisonSettings.initialization_seed,
    )
    parser.add_argument(
        "--simulator",
        default=ComparisonSettings.simulator,
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=ComparisonSettings.shots,
        help=(
            "Finite shots per circuit evaluation "
            f"(default: {ComparisonSettings.shots})."
        ),
    )
    parser.add_argument(
        "--reference-max-iterations",
        type=int,
        default=ComparisonSettings.reference_max_iterations,
        help="Maximum analytic L-BFGS-B iterations for the reference optimum.",
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--png", type=Path, default=DEFAULT_PNG)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument(
        "--error-png",
        type=Path,
        default=DEFAULT_ERROR_PNG,
    )
    parser.add_argument(
        "--error-pdf",
        type=Path,
        default=DEFAULT_ERROR_PDF,
    )
    return parser


def main() -> None:
    """Run the command-line comparison."""
    arguments = build_parser().parse_args()
    settings = ComparisonSettings(
        n=arguments.n,
        graph_family=arguments.graph_family,
        graph_seed=arguments.graph_seed,
        depth=arguments.depth,
        gamma=arguments.gamma,
        iterations=arguments.iterations,
        learning_rate=arguments.learning_rate,
        finite_difference_step=arguments.finite_difference_step,
        initialization_seed=arguments.initialization_seed,
        simulator=arguments.simulator,
        shots=arguments.shots,
        reference_max_iterations=arguments.reference_max_iterations,
    )
    results = run_comparison(
        settings,
        csv_path=arguments.csv,
        png_path=arguments.png,
        pdf_path=arguments.pdf,
        error_png_path=arguments.error_png,
        error_pdf_path=arguments.error_pdf,
    )
    trajectories = results.loc[results["record_type"] == "iteration"]
    final_rows = (
        trajectories.sort_values("iteration")
        .groupby("method", as_index=False)
        .tail(1)
    )
    for row in final_rows.itertuples(index=False):
        print(
            f"{row.method}: loss={row.loss:.8f}, "
            f"mean_ratio={row.mean_final_ratio:.8f}, "
            f"parameter_error={row.parameter_error_l2:.8f}"
        )


if __name__ == "__main__":
    main()
