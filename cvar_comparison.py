"""Compare fixed and ascending CVaR objectives for finite-shot QAOA MaxCut.

The experiment mirrors the design of Figure (b): QAOA depth two, five
independently generated eight-vertex MaxCut instances, repeated initial
parameters, and graph-level aggregation.  The lower-tail CVaR of the MaxCut
loss is minimized. Fixed runs hold ``alpha`` constant, while scheduled runs
linearly decrease alpha from one (expectation loss) to the final alpha. Thus,
the scheduled risk strength ``1 - alpha`` increases during optimization.

The script creates:

* ``CVaR.csv`` -- restart and aggregate simulation results plus curve fits;
* ``CVAR.jpg`` and ``CVAR.pdf`` -- Figure (d);
* ``CVAR_fixed_alpha.jpg`` and ``CVAR_fixed_alpha.pdf`` -- fixed-alpha-only
  curve with a shaded graph-level SEM range;
* ``performance_matching_function.jpg`` and
  ``performance_matching_function.pdf`` -- fixed-CVaR and fixed-QTL
  response functions on a logarithmically matched landscape-control
  coordinate.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pennylane as qml
from pennylane import numpy as qnp
from scipy.optimize import least_squares

from experiments import (
    DEFAULT_GRAPH_FAMILIES,
    DEFAULT_GRAPH_SEEDS,
    SHARED_TILT_OPTIMIZER,
    standard_error,
)
from max_cut import MaxCutProblem, build_maxcut_problem, summarize_distribution
from plotting import apply_plot_style, padded_limits, plt, save_figure
from qaoa import make_probability_qnode, random_initial_parameters


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = SCRIPT_DIR / "CVaR.csv"
DEFAULT_CVAR_JPG = SCRIPT_DIR / "CVAR.jpg"
DEFAULT_CVAR_PDF = SCRIPT_DIR / "CVAR.pdf"
DEFAULT_FIXED_JPG = SCRIPT_DIR / "CVAR_fixed_alpha.jpg"
DEFAULT_FIXED_PDF = SCRIPT_DIR / "CVAR_fixed_alpha.pdf"
DEFAULT_PERFORMANCE_JPG = SCRIPT_DIR / "performance_matching_function.jpg"
DEFAULT_PERFORMANCE_PDF = SCRIPT_DIR / "performance_matching_function.pdf"
DEFAULT_QTL_FIXED_CSV = (
    SCRIPT_DIR
    / "data_and_figures"
    / "fixed_gamma_shot_5000.csv"
)
DEFAULT_ALPHAS = (0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 1.00)
INITIALIZATION_BASE_SEED = 20260726
DENSE_POINTS = 800
FIT_GAMMA_MAX = 4.0
TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class CVaRSettings:
    """Complete numerical specification shared by both CVaR strategies."""

    shots: int = 5000
    steps: int = 100
    n: int = 8
    depth: int = 2
    number_of_initializations: int = 5
    alpha_start: float = 1.00
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
    simulator: str = "lightning.qubit"

    def __post_init__(self) -> None:
        if self.shots < 1:
            raise ValueError("shots must be positive.")
        if self.steps < 1:
            raise ValueError("steps must be positive.")
        if self.n < 2 or self.depth < 1:
            raise ValueError("n must be at least two and depth must be positive.")
        if self.number_of_initializations < 1:
            raise ValueError("number_of_initializations must be positive.")
        if not 0.0 < self.alpha_start <= 1.0:
            raise ValueError("alpha_start must be in (0, 1].")
        if self.tail_window < 1:
            raise ValueError("tail_window must be positive.")


@dataclass(frozen=True)
class MatchingCurveResult:
    """Data and parameters for a robust cubic matching function."""

    parameter_data: np.ndarray
    ratio_data: np.ndarray
    sem_data: np.ndarray
    params: dict[str, float | str]


def cubic_model(x, a0, a1, a2, a3):
    """Evaluate a cubic polynomial in increasing coefficient order."""
    x = np.asarray(x, dtype=float)
    return a0 + a1 * x + a2 * x**2 + a3 * x**3


def asymmetric_peak_model(
    parameter: np.ndarray,
    peak_value: float,
    peak_parameter: float,
    left_curvature: float,
    right_curvature: float,
) -> np.ndarray:
    """Evaluate a continuous asymmetric quadratic around an empirical peak."""
    parameter = np.asarray(parameter, dtype=float)
    displacement = parameter - peak_parameter
    curvature = np.where(
        displacement <= 0.0,
        left_curvature,
        right_curvature,
    )
    return peak_value - curvature * displacement**2


def fit_asymmetric_peak_response(
    parameter: np.ndarray,
    ratio: np.ndarray,
    sem: np.ndarray,
) -> dict[str, float | str]:
    """Match an asymmetric response through baseline, best, and tail means.

    The three empirical anchors determine two nonnegative curvatures. This
    phenomenological response is a visual matching function, not a physical
    identification of the CVaR and QTL parameters.
    """
    parameter = np.asarray(parameter, dtype=float)
    ratio = np.asarray(ratio, dtype=float)
    sem = np.maximum(np.asarray(sem, dtype=float), 1e-3)
    order = np.argsort(parameter)
    parameter = parameter[order]
    ratio = ratio[order]
    sem = sem[order]
    peak_index = int(np.argmax(ratio))
    peak_parameter = float(parameter[peak_index])
    peak_value = float(ratio[peak_index])
    baseline_value = float(ratio[0])
    tail_value = float(ratio[-1])
    left_span = max(peak_parameter, np.finfo(float).eps)
    right_span = max(1.0 - peak_parameter, np.finfo(float).eps)
    left_curvature = max(
        0.0,
        (peak_value - baseline_value) / left_span**2,
    )
    right_curvature = max(
        0.0,
        (peak_value - tail_value) / right_span**2,
    )
    fitted = asymmetric_peak_model(
        parameter,
        peak_value,
        peak_parameter,
        left_curvature,
        right_curvature,
    )
    return {
        "model_formula": (
            r"$R^\star-c_-(r-r^\star)^2$ for $r\leq r^\star$; "
            r"$R^\star-c_+(r-r^\star)^2$ for $r>r^\star$"
        ),
        "peak_parameter": peak_parameter,
        "peak_value": peak_value,
        "left_curvature": left_curvature,
        "right_curvature": right_curvature,
        "baseline_value": baseline_value,
        "tail_value": tail_value,
        "weighted_sse": float(np.sum(((fitted - ratio) / sem) ** 2)),
    }


def cvar_control_coordinate(
    alpha: np.ndarray,
    alpha_min: float,
) -> np.ndarray:
    """Map CVaR alpha to a zero-to-one logarithmic control coordinate."""
    alpha = np.asarray(alpha, dtype=float)
    return -np.log(alpha) / np.log(1.0 / alpha_min)


def fit_weighted_cubic(
    parameter: np.ndarray,
    ratio: np.ndarray,
    sem: np.ndarray,
) -> dict[str, object]:
    """Fit a robust cubic using inverse-SEM normalized residuals."""
    parameter = np.asarray(parameter, dtype=float)
    ratio = np.asarray(ratio, dtype=float)
    sem = np.maximum(np.asarray(sem, dtype=float), 1e-3)
    if len(parameter) < 4:
        raise ValueError("At least four points are required for a cubic fit.")
    initial_coefficients = np.polynomial.polynomial.polyfit(
        parameter,
        ratio,
        deg=3,
        w=1.0 / sem,
    )

    def residuals(coefficients):
        return (
            cubic_model(parameter, *coefficients) - ratio
        ) / sem

    fit = least_squares(
        residuals,
        x0=initial_coefficients,
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=10000,
    )
    if not fit.success:
        raise RuntimeError(f"Cubic fitting failed: {fit.message}")
    a0, a1, a2, a3 = map(float, fit.x)
    dense_parameter = np.linspace(
        float(parameter.min()),
        float(parameter.max()),
        DENSE_POINTS,
    )
    dense_ratio = cubic_model(
        dense_parameter,
        a0,
        a1,
        a2,
        a3,
    )
    peak_index = int(np.argmax(dense_ratio))
    return {
        "model_formula": r"$a_0+a_1x+a_2x^2+a_3x^3$",
        "a0": a0,
        "a1": a1,
        "a2": a2,
        "a3": a3,
        "weighted_sse": float(np.sum(residuals(fit.x) ** 2)),
        "peak_parameter": float(dense_parameter[peak_index]),
        "peak_value": float(dense_ratio[peak_index]),
        "tail_value": float(dense_ratio[-1]),
        "gamma_dense": dense_parameter,
        "ratio_dense": dense_ratio,
    }


def fit_fixed_qtl_curve(path: Path) -> MatchingCurveResult:
    """Fit the 5,000-shot fixed-gamma QTL results with the shared cubic."""
    path = Path(path)
    frame = pd.read_csv(path)
    required = {"gamma_plot", "mean_final_ratio", "sem_final_ratio"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame = (
        frame.dropna(subset=list(required))
        .loc[lambda data: data["gamma_plot"] <= FIT_GAMMA_MAX]
        .sort_values("gamma_plot")
        .drop_duplicates("gamma_plot")
    )
    gamma = frame["gamma_plot"].to_numpy(dtype=float)
    ratio = frame["mean_final_ratio"].to_numpy(dtype=float)
    sem = np.maximum(
        frame["sem_final_ratio"].to_numpy(dtype=float),
        1e-3,
    )
    fit = fit_weighted_cubic(gamma, ratio, sem)
    fit["model_formula"] = (
        r"$a_0+a_1|\gamma|+a_2|\gamma|^2+a_3|\gamma|^3$"
    )
    return MatchingCurveResult(
        parameter_data=gamma,
        ratio_data=ratio,
        sem_data=sem,
        params={
            key: value
            for key, value in fit.items()
            if key not in {"gamma_dense", "ratio_dense"}
        },
    )


def validate_alphas(
    alphas: list[float] | tuple[float, ...],
    alpha_start: float,
) -> tuple[float, ...]:
    """Return sorted unique alpha values valid for both experiment arms."""
    values = tuple(sorted({float(alpha) for alpha in alphas}))
    if len(values) < 4:
        raise ValueError("At least four alpha values are needed for curve fitting.")
    if values[0] <= 0.0 or values[-1] > alpha_start + 1e-12:
        raise ValueError(
            "Every final alpha must lie in (0, alpha_start]."
        )
    return values


def cvar_loss_from_distribution(
    probabilities,
    losses: np.ndarray,
    alpha: float,
):
    r"""Evaluate lower-tail CVaR for a discrete loss distribution.

    For ordered losses :math:`l_{(i)}`, the function takes all probability
    mass below the alpha quantile and the necessary fraction at the quantile:

    .. math::
       \operatorname{CVaR}_{\alpha}(L)
       = \alpha^{-1}\sum_i w_i l_{(i)},\quad \sum_i w_i=\alpha.

    This definition handles quantile ties and finite-shot probability masses
    without expanding a 5,000-element sample vector.
    """
    if not 0.0 < float(alpha) <= 1.0:
        raise ValueError("CVaR alpha must be in (0, 1].")
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
    remaining_tail_mass = qml.math.maximum(
        float(alpha) - preceding_mass,
        0.0,
    )
    selected_mass = qml.math.minimum(
        ordered_probabilities,
        remaining_tail_mass,
    )
    return qml.math.dot(selected_mass, ordered_losses) / float(alpha)


def alpha_at_step(
    strategy: str,
    final_alpha: float,
    alpha_start: float,
    step: int,
    total_steps: int,
) -> float:
    """Return fixed alpha or the endpoint-inclusive 1-to-alpha schedule."""
    if strategy == "fixed":
        return float(final_alpha)
    if strategy != "ascending":
        raise ValueError(f"Unknown CVaR strategy: {strategy}")
    if total_steps <= 1:
        return float(final_alpha)
    fraction = step / (total_steps - 1)
    return float(alpha_start + fraction * (final_alpha - alpha_start))


def _numeric_cvar_loss(
    probabilities: np.ndarray,
    losses: np.ndarray,
    alpha: float,
) -> float:
    """Evaluate CVaR as a plain float for logging after an optimizer step."""
    return float(cvar_loss_from_distribution(probabilities, losses, alpha))


def _device_seed(
    graph_seed: int,
    alpha_index: int,
    initialization_id: int,
) -> int:
    """Create one finite-shot seed shared by a paired strategy comparison."""
    return (
        730_000
        + 10_000 * int(graph_seed)
        + 100 * int(alpha_index)
        + int(initialization_id)
    )


def optimize_cvar_qaoa(
    circuit,
    problem: MaxCutProblem,
    initial_parameters: np.ndarray,
    *,
    strategy: str,
    final_alpha: float,
    settings: CVaRSettings,
) -> dict[str, float | list[float]]:
    """Optimize one QAOA restart using fixed or ascending CVaR."""
    parameters = qnp.array(initial_parameters, requires_grad=True)
    momentum_vector = np.zeros_like(initial_parameters, dtype=float)
    classical_losses = problem.maximum_cut - problem.cut_values
    ratio_history: list[float] = []
    loss_history: list[float] = []
    best_ratio_so_far = 0.0

    for step in range(settings.steps):
        current_alpha = alpha_at_step(
            strategy,
            final_alpha,
            settings.alpha_start,
            step,
            settings.steps,
        )

        def loss_function(theta):
            probabilities = circuit(qnp.mod(theta, TWO_PI))
            return cvar_loss_from_distribution(
                probabilities,
                classical_losses,
                current_alpha,
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
        parameters = qnp.array(
            np.mod(
                np.asarray(parameters, dtype=float)
                - learning_rate * momentum_vector,
                TWO_PI,
            ),
            requires_grad=True,
        )

        probabilities = np.asarray(circuit(parameters), dtype=float)
        metrics = summarize_distribution(
            probabilities,
            problem.cut_values,
            problem.maximum_cut,
        )
        ratio_history.append(float(metrics["mean_ratio"]))
        loss_history.append(
            _numeric_cvar_loss(
                probabilities,
                classical_losses,
                current_alpha,
            )
        )
        best_ratio_so_far = max(
            best_ratio_so_far,
            float(metrics["best_ratio"]),
        )

    tail_count = min(settings.tail_window, len(ratio_history))
    return {
        "final_parameters": np.asarray(parameters, dtype=float).tolist(),
        "final_cvar_loss": loss_history[-1],
        "final_mean_ratio": ratio_history[-1],
        "tail_mean_ratio": float(np.mean(ratio_history[-tail_count:])),
        "peak_mean_ratio": float(np.max(ratio_history)),
        "best_ratio_so_far": best_ratio_so_far,
        "final_optimal_mass": float(metrics["optimal_mass"]),
    }


def _run_restart(task: dict[str, object]) -> dict[str, object]:
    """Worker entry point for one graph/alpha/strategy/initialization."""
    settings = CVaRSettings(**dict(task["settings"]))
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
    initial_parameters = np.asarray(task["initial_parameters"], dtype=float)
    result = optimize_cvar_qaoa(
        circuit,
        problem,
        initial_parameters,
        strategy=str(task["strategy"]),
        final_alpha=float(task["alpha"]),
        settings=settings,
    )
    return {
        "task_id": str(task["task_id"]),
        "record_type": "restart",
        "strategy": str(task["strategy"]),
        "alpha": float(task["alpha"]),
        "one_minus_alpha": 1.0 - float(task["alpha"]),
        "alpha_start": (
            float(task["alpha"])
            if str(task["strategy"]) == "fixed"
            else settings.alpha_start
        ),
        "alpha_end": float(task["alpha"]),
        "schedule": (
            "constant"
            if str(task["strategy"]) == "fixed"
            else "linear_descending_alpha"
        ),
        "graph_family": str(task["graph_family"]),
        "n": settings.n,
        "p": settings.depth,
        "graph_seed": int(task["graph_seed"]),
        "init_id": int(task["initialization_id"]),
        "init_seed": int(task["initialization_seed"]),
        "device_seed": int(task["device_seed"]),
        "initial_parameters": json.dumps(initial_parameters.tolist()),
        "final_parameters": json.dumps(result.pop("final_parameters")),
        **result,
    }


def build_tasks(
    settings: CVaRSettings,
    alphas: tuple[float, ...],
) -> list[dict[str, object]]:
    """Build the deterministic full factorial CVaR task list."""
    tasks: list[dict[str, object]] = []
    settings_dict = asdict(settings)
    initial_points = [
        (
            INITIALIZATION_BASE_SEED + initialization_id,
            random_initial_parameters(
                settings.depth,
                INITIALIZATION_BASE_SEED + initialization_id,
                upper_bound=np.pi,
            ),
        )
        for initialization_id in range(settings.number_of_initializations)
    ]
    for strategy in ("fixed", "ascending"):
        for graph_family in DEFAULT_GRAPH_FAMILIES:
            for graph_seed in DEFAULT_GRAPH_SEEDS:
                for alpha_index, alpha in enumerate(alphas):
                    for initialization_id, (
                        initialization_seed,
                        initial_parameters,
                    ) in enumerate(initial_points):
                        task_id = (
                            f"shots={settings.shots}|steps={settings.steps}|"
                            f"sim={settings.simulator}|"
                            "seed_pairing=common_v2|"
                            f"alpha_start={settings.alpha_start:.12g}|"
                            f"{strategy}|{graph_family}|{graph_seed}|"
                            f"{alpha:.12g}|{initialization_id}"
                        )
                        tasks.append(
                            {
                                "task_id": task_id,
                                "strategy": strategy,
                                "graph_family": graph_family,
                                "graph_seed": graph_seed,
                                "alpha": alpha,
                                "initialization_id": initialization_id,
                                "initialization_seed": initialization_seed,
                                "initial_parameters": initial_parameters,
                                "device_seed": _device_seed(
                                    graph_seed,
                                    alpha_index,
                                    initialization_id,
                                ),
                                "settings": settings_dict,
                            }
                        )
    return tasks


def validate_paired_tasks(tasks: list[dict[str, object]]) -> None:
    """Require identical initial parameters and shot seeds across strategies."""
    paired_fields = [
        "graph_family",
        "graph_seed",
        "alpha",
        "initialization_id",
    ]
    task_frame = pd.DataFrame(tasks)
    for _, pair in task_frame.groupby(paired_fields, sort=False):
        if set(pair["strategy"]) != {"fixed", "ascending"} or len(pair) != 2:
            raise ValueError("Every restart must have one task per strategy.")
        for field in (
            "initialization_seed",
            "device_seed",
        ):
            if pair[field].nunique() != 1:
                raise ValueError(
                    f"Paired strategies have different {field} values."
                )
        parameters = [
            np.asarray(value, dtype=float)
            for value in pair["initial_parameters"]
        ]
        if not np.array_equal(parameters[0], parameters[1]):
            raise ValueError(
                "Paired strategies have different initial parameter vectors."
            )
        settings = pair["settings"].tolist()
        if settings[0] != settings[1]:
            raise ValueError(
                "Paired strategies have different optimizer settings."
            )


def _load_checkpoint(
    checkpoint_path: Path,
    expected_task_ids: set[str],
) -> dict[str, dict[str, object]]:
    """Load compatible completed restart rows from a partial result file."""
    if not checkpoint_path.exists():
        return {}
    checkpoint = pd.read_csv(checkpoint_path)
    if "task_id" not in checkpoint.columns:
        return {}
    checkpoint = checkpoint.loc[
        checkpoint["task_id"].isin(expected_task_ids)
    ].drop_duplicates("task_id", keep="last")
    return {
        str(row["task_id"]): row.to_dict()
        for _, row in checkpoint.iterrows()
    }


def execute_tasks(
    tasks: list[dict[str, object]],
    *,
    workers: int,
    checkpoint_path: Path,
    resume: bool,
) -> pd.DataFrame:
    """Execute restart tasks, checkpointing after every completed result."""
    if workers < 1:
        raise ValueError("workers must be at least one.")
    expected_ids = {str(task["task_id"]) for task in tasks}
    completed = (
        _load_checkpoint(checkpoint_path, expected_ids)
        if resume
        else {}
    )
    pending = [
        task for task in tasks if str(task["task_id"]) not in completed
    ]
    if completed:
        print(
            f"Resuming from {len(completed)}/{len(tasks)} completed restarts.",
            flush=True,
        )

    def save_checkpoint() -> None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(completed.values()).sort_values("task_id").to_csv(
            checkpoint_path,
            index=False,
        )

    if workers == 1:
        for task in pending:
            row = _run_restart(task)
            completed[str(row["task_id"])] = row
            save_checkpoint()
            print(
                f"[completed] {len(completed)}/{len(tasks)} "
                f"{row['strategy']} alpha={row['alpha']:.2f} "
                f"graph={row['graph_seed']} init={row['init_id']}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(
            max_workers=min(workers, max(1, len(pending)))
        ) as executor:
            futures = {
                executor.submit(_run_restart, task): task for task in pending
            }
            for future in as_completed(futures):
                row = future.result()
                completed[str(row["task_id"])] = row
                save_checkpoint()
                print(
                    f"[completed] {len(completed)}/{len(tasks)} "
                    f"{row['strategy']} alpha={row['alpha']:.2f} "
                    f"graph={row['graph_seed']} init={row['init_id']}",
                    flush=True,
                )

    if len(completed) != len(tasks):
        raise RuntimeError(
            f"Only {len(completed)} of {len(tasks)} restarts completed."
        )
    return (
        pd.DataFrame(completed.values())
        .sort_values(
            ["strategy", "graph_seed", "alpha", "init_id"]
        )
        .reset_index(drop=True)
    )


def aggregate_restart_results(
    restart_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average initializations per graph and then compute graph-level SEM."""
    graph_average = (
        restart_frame.groupby(
            [
                "strategy",
                "graph_family",
                "n",
                "p",
                "graph_seed",
                "alpha",
                "one_minus_alpha",
                "alpha_start",
                "alpha_end",
                "schedule",
            ],
            as_index=False,
        )
        .agg(
            num_init_points=("init_id", "nunique"),
            final_mean_ratio=("final_mean_ratio", "mean"),
            tail_mean_ratio=("tail_mean_ratio", "mean"),
            peak_mean_ratio=("peak_mean_ratio", "mean"),
            final_cvar_loss=("final_cvar_loss", "mean"),
            final_optimal_mass=("final_optimal_mass", "mean"),
        )
    )
    summary = (
        graph_average.groupby(
            [
                "strategy",
                "alpha",
                "one_minus_alpha",
                "alpha_start",
                "alpha_end",
                "schedule",
            ],
            as_index=False,
        )
        .agg(
            graph_count=("graph_seed", "nunique"),
            mean_peak_ratio=("peak_mean_ratio", "mean"),
            sem_peak_ratio=("peak_mean_ratio", standard_error),
            mean_tail_ratio=("tail_mean_ratio", "mean"),
            sem_tail_ratio=("tail_mean_ratio", standard_error),
            mean_final_ratio=("final_mean_ratio", "mean"),
            sem_final_ratio=("final_mean_ratio", standard_error),
            mean_final_cvar_loss=("final_cvar_loss", "mean"),
            sem_final_cvar_loss=("final_cvar_loss", standard_error),
            mean_optimal_mass=("final_optimal_mass", "mean"),
        )
        .sort_values(["strategy", "alpha"])
        .reset_index(drop=True)
    )
    return graph_average, summary


def fit_cvar_summary(
    summary: pd.DataFrame,
) -> dict[str, dict[str, object]]:
    """Fit the same robust inverse-SEM cubic used in Figure (b)."""
    fits: dict[str, dict[str, object]] = {}
    for strategy in ("fixed", "ascending"):
        series = summary.loc[summary["strategy"] == strategy].sort_values(
            "one_minus_alpha"
        )
        fit = fit_weighted_cubic(
            series["one_minus_alpha"].to_numpy(dtype=float),
            series["mean_final_ratio"].to_numpy(dtype=float),
            np.maximum(
                series["sem_final_ratio"].to_numpy(dtype=float),
                1e-3,
            ),
        )
        dense_risk = np.linspace(
            float(series["one_minus_alpha"].min()),
            float(series["one_minus_alpha"].max()),
            DENSE_POINTS,
        )
        coefficients = [
            float(fit[name]) for name in ("a0", "a1", "a2", "a3")
        ]
        dense_ratio = cubic_model(dense_risk, *coefficients)
        peak_index = int(np.argmax(dense_ratio))
        fits[strategy] = {
            **{
                key: value
                for key, value in fit.items()
                if key
                not in {
                    "gamma_dense",
                    "ratio_dense",
                    "model_formula",
                    "peak_gamma",
                    "peak_value",
                    "tail_value",
                }
            },
            "model_formula": (
                r"$a_0+a_1(1-\alpha)+a_2(1-\alpha)^2"
                r"+a_3(1-\alpha)^3$"
            ),
            "peak_one_minus_alpha": float(dense_risk[peak_index]),
            "peak_value": float(dense_ratio[peak_index]),
            "tail_value": float(dense_ratio[-1]),
            "one_minus_alpha_dense": dense_risk,
            "ratio_dense": dense_ratio,
        }
    return fits


def _fit_record(
    method: str,
    strategy: str,
    parameter_name: str,
    fit: dict[str, object],
) -> dict[str, object]:
    """Create a CSV-safe fit record."""
    return {
        "record_type": "matching_function",
        "method": method,
        "strategy": strategy,
        "parameter_name": parameter_name,
        "fit_model": "robust_sem_weighted_cubic",
        "fit_formula": fit["model_formula"],
        **{
            key: fit[key]
            for key in (
                "a0",
                "a1",
                "a2",
                "a3",
                "weighted_sse",
                "peak_value",
                "tail_value",
            )
            if key in fit
        },
    }


def fixed_comparison_fit_rows(
    summary: pd.DataFrame,
    qtl_fit: MatchingCurveResult,
    alpha_min: float,
) -> list[dict[str, object]]:
    """Fit and serialize fixed-objective landscape-response functions."""
    cvar_data = summary.loc[summary["strategy"] == "fixed"].sort_values(
        "alpha",
        ascending=False,
    )
    log_denominator = np.log(1.0 / alpha_min)
    cvar_fit = fit_asymmetric_peak_response(
        cvar_control_coordinate(
            cvar_data["alpha"].to_numpy(dtype=float),
            alpha_min,
        ),
        cvar_data["mean_final_ratio"].to_numpy(dtype=float),
        cvar_data["sem_final_ratio"].to_numpy(dtype=float),
    )
    qtl_response_fit = fit_asymmetric_peak_response(
        qtl_fit.parameter_data / FIT_GAMMA_MAX,
        qtl_fit.ratio_data,
        qtl_fit.sem_data,
    )
    rows: list[dict[str, object]] = []
    for method, mapping, fit in (
        (
            "CVaR",
            (
                rf"$r=-\ln(\alpha)/{log_denominator:.6g}$; "
                rf"$\alpha_{{\min}}={alpha_min:g}$"
            ),
            cvar_fit,
        ),
        (
            "QTL",
            rf"$r=|\gamma|/{FIT_GAMMA_MAX:g}$",
            qtl_response_fit,
        ),
    ):
        rows.append(
            {
                "record_type": "comparison_matching_function",
                "method": method,
                "strategy": "fixed",
                "parameter_name": "landscape_control_coordinate",
                "parameter_mapping": mapping,
                "alpha_gamma_relation": (
                    rf"$\alpha=\exp[-\ln(1/{alpha_min:g})"
                    rf"|\gamma|/{FIT_GAMMA_MAX:g}]$"
                ),
                "fit_model": (
                    "baseline_peak_tail_anchored_"
                    "asymmetric_quadratic"
                ),
                "fit_formula": fit["model_formula"],
                "peak_parameter": fit["peak_parameter"],
                "peak_value": fit["peak_value"],
                "left_curvature": fit["left_curvature"],
                "right_curvature": fit["right_curvature"],
                "baseline_value": fit["baseline_value"],
                "tail_value": fit["tail_value"],
                "weighted_sse": fit["weighted_sse"],
            }
        )
    return rows


def build_output_frame(
    restart_frame: pd.DataFrame,
    graph_average: pd.DataFrame,
    summary: pd.DataFrame,
    cvar_fits: dict[str, dict[str, object]],
    qtl_fit,
    settings: CVaRSettings,
    alphas: tuple[float, ...],
) -> pd.DataFrame:
    """Combine all simulation levels, metadata, and fit coefficients."""
    restart_rows = restart_frame.copy()
    graph_rows = graph_average.copy()
    summary_rows = summary.copy()
    graph_rows.insert(0, "record_type", "graph_average")
    summary_rows.insert(0, "record_type", "summary")

    fit_rows = [
        _fit_record(
            "CVaR",
            strategy,
            "one_minus_alpha",
            cvar_fits[strategy],
        )
        for strategy in ("fixed", "ascending")
    ]
    fit_rows.append(
        _fit_record(
            "QTL",
            "fixed",
            "absolute_gamma",
            qtl_fit.params,
        )
    )
    comparison_rows = fixed_comparison_fit_rows(
        summary,
        qtl_fit,
        alpha_min=min(alphas),
    )
    frames = [
        restart_rows,
        graph_rows,
        summary_rows,
        pd.DataFrame(fit_rows),
        pd.DataFrame(comparison_rows),
    ]
    output = pd.concat(frames, ignore_index=True, sort=False)
    metadata = {
        "experiment": "finite_shot_qaoa_maxcut_cvar",
        "shots": settings.shots,
        "steps": settings.steps,
        "number_of_initializations": settings.number_of_initializations,
        "alpha_grid": json.dumps(list(alphas)),
        "optimizer": "Polyak momentum gradient descent",
        "paired_initial_parameters": True,
        "paired_device_seeds": True,
        "identical_optimizer_settings": True,
        "initialization_base_seed": INITIALIZATION_BASE_SEED,
        **{
            f"optimizer_{key}": value
            for key, value in asdict(settings).items()
            if key
            in {
                "learning_rate",
                "learning_rate_decay_power",
                "learning_rate_decay_offset",
                "momentum",
                "gradient_clip",
                "tail_window",
                "simulator",
            }
        },
    }
    for column, value in metadata.items():
        output[column] = value
    return output


def refresh_fit_records(
    output_csv: Path,
    cvar_fits: dict[str, dict[str, object]],
    qtl_fit: MatchingCurveResult,
    summary: pd.DataFrame,
) -> None:
    """Replace saved fit metadata without altering simulation records."""
    frame = pd.read_csv(output_csv)
    fit_rows = [
        _fit_record(
            "CVaR",
            strategy,
            "one_minus_alpha",
            cvar_fits[strategy],
        )
        for strategy in ("fixed", "ascending")
    ]
    fit_rows.append(
        _fit_record(
            "QTL",
            "fixed",
            "absolute_gamma",
            qtl_fit.params,
        )
    )
    replacement = pd.DataFrame(
        fit_rows
        + fixed_comparison_fit_rows(
            summary,
            qtl_fit,
            alpha_min=float(summary["alpha"].min()),
        )
    )
    metadata_columns = {
        "experiment",
        "shots",
        "steps",
        "number_of_initializations",
        "alpha_grid",
        "optimizer",
        "paired_initial_parameters",
        "paired_device_seeds",
        "identical_optimizer_settings",
        "initialization_base_seed",
    }
    metadata_columns.update(
        column
        for column in frame.columns
        if column.startswith("optimizer_")
    )
    for column in metadata_columns:
        if column not in frame.columns:
            continue
        values = frame[column].dropna()
        if not values.empty:
            replacement[column] = values.iloc[0]
    preserved = frame.loc[
        ~frame["record_type"].isin(
            {"matching_function", "comparison_matching_function"}
        )
    ]
    refreshed = pd.concat(
        [preserved, replacement],
        ignore_index=True,
        sort=False,
    )
    refreshed.to_csv(output_csv, index=False)


def _publication_style() -> None:
    """Apply the Figure (b) style with restrained journal-like refinements."""
    apply_plot_style()
    plt.rcParams.update(
        {
            "font.size": 17,
            "axes.labelsize": 19,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 13,
            "legend.frameon": False,
            "axes.linewidth": 1.25,
            "lines.linewidth": 2.2,
            "xtick.major.size": 7,
            "ytick.major.size": 7,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
        }
    )


def plot_cvar_figure(
    summary: pd.DataFrame,
    fits: dict[str, dict[str, object]],
    output_jpg: Path,
    output_pdf: Path,
) -> None:
    """Render Figure (d) against increasing CVaR risk strength 1-alpha."""
    _publication_style()
    styles = {
        "fixed": ("Fixed $\\alpha$", "#1F4E79", (0, (4, 3)), "o"),
        "ascending": (
            r"Scheduled $\alpha:1\rightarrow\alpha$",
            "#D97706",
            "-",
            "s",
        ),
    }
    figure, axis = plt.subplots(figsize=(8.2, 6.0))
    displayed_values: list[np.ndarray] = []
    for strategy in ("fixed", "ascending"):
        label, color, line_style, marker = styles[strategy]
        series = summary.loc[summary["strategy"] == strategy].sort_values(
            "one_minus_alpha"
        )
        fit = fits[strategy]
        x_dense = np.asarray(
            fit["one_minus_alpha_dense"],
            dtype=float,
        )
        y_dense = np.asarray(fit["ratio_dense"], dtype=float)
        rms_sem = float(
            np.sqrt(np.mean(np.square(series["sem_final_ratio"])))
        )
        displayed_values.extend(
            [y_dense - rms_sem, y_dense + rms_sem]
        )
        axis.fill_between(
            x_dense,
            y_dense - rms_sem,
            y_dense + rms_sem,
            color=color,
            alpha=0.09,
            linewidth=0,
            zorder=1,
        )
        axis.plot(
            x_dense,
            y_dense,
            color=color,
            linestyle=line_style,
            label=label,
            zorder=3,
        )
        axis.errorbar(
            series["one_minus_alpha"],
            series["mean_final_ratio"],
            yerr=series["sem_final_ratio"],
            color=color,
            linestyle="none",
            marker=marker,
            markersize=7.5,
            markeredgecolor="white",
            markeredgewidth=0.6,
            elinewidth=1.1,
            capsize=3,
            zorder=4,
        )

    y_min, y_max = padded_limits(displayed_values, padding_fraction=0.08)
    axis.set(
        xlabel=r"CVaR risk strength $1-\alpha$",
        ylabel="Mean final ratio",
        xlim=(-0.02, 0.92),
        ylim=(y_min, y_max),
    )
    axis.set_xticks(
        sorted(
            summary["one_minus_alpha"]
            .dropna()
            .unique()
            .astype(float)
            .tolist()
        )
    )
    axis.grid(
        axis="y",
        color="#D0D0D0",
        linestyle=(0, (3, 3)),
        linewidth=0.85,
        alpha=0.9,
    )
    axis.grid(axis="x", visible=False)
    axis.tick_params(which="both", direction="in", top=True, right=True)
    axis.legend(loc="upper right")
    axis.text(
        0.02,
        0.97,
        "(d)",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=17,
        fontweight="bold",
    )
    save_figure(figure, output_jpg, output_pdf, dpi=600)


def plot_fixed_alpha_figure(
    summary: pd.DataFrame,
    fixed_fit: dict[str, object],
    output_jpg: Path,
    output_pdf: Path,
) -> None:
    """Render a paper-ready fixed-alpha curve with a shaded SEM envelope."""
    _publication_style()
    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.labelsize": 17,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12.5,
            "axes.linewidth": 1.15,
            "xtick.major.size": 6,
            "ytick.major.size": 6,
            "xtick.minor.size": 3,
            "ytick.minor.size": 3,
        }
    )
    series = summary.loc[summary["strategy"] == "fixed"].sort_values(
        "one_minus_alpha"
    )
    x_dense = np.asarray(
        fixed_fit["one_minus_alpha_dense"],
        dtype=float,
    )
    y_dense = np.asarray(fixed_fit["ratio_dense"], dtype=float)
    sem_dense = np.interp(
        x_dense,
        series["one_minus_alpha"].to_numpy(dtype=float),
        series["sem_final_ratio"].to_numpy(dtype=float),
    )

    figure, axis = plt.subplots(figsize=(7.2, 5.25))
    sem_artist = axis.fill_between(
        x_dense,
        y_dense - sem_dense,
        y_dense + sem_dense,
        color="#1F4E79",
        alpha=0.09,
        linewidth=0,
        label=r"Mean $\pm$ graph-level SEM",
        zorder=1,
    )
    fit_artist, = axis.plot(
        x_dense,
        y_dense,
        color="#1F4E79",
        linestyle=(0, (4, 3)),
        linewidth=2.25,
        label=r"Fixed-$\alpha$ cubic fit",
        zorder=3,
    )
    axis.scatter(
        series["one_minus_alpha"],
        series["mean_final_ratio"],
        marker="o",
        color="#1F4E79",
        s=58,
        edgecolors="white",
        linewidths=0.7,
        zorder=4,
    )
    y_min, y_max = padded_limits(
        [y_dense - sem_dense, y_dense + sem_dense],
        padding_fraction=0.07,
    )
    axis.set(
        xlabel=r"CVaR risk strength $1-\alpha$",
        ylabel="Mean Final Ratio",
        xlim=(-0.02, 0.92),
        ylim=(y_min, y_max),
    )
    axis.set_xticks(
        sorted(
            series["one_minus_alpha"]
            .dropna()
            .unique()
            .astype(float)
            .tolist()
        )
    )
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
    axis.legend(
        handles=[fit_artist, sem_artist],
        loc="upper right",
        handlelength=3.0,
        borderaxespad=0.45,
    )
    save_figure(figure, output_jpg, output_pdf, dpi=600)


def plot_performance_matching_functions(
    summary: pd.DataFrame,
    qtl_fit,
    *,
    alpha_min: float,
    output_jpg: Path,
    output_pdf: Path,
) -> None:
    """Compare fixed CVaR and QTL responses under a log/exp parameter map."""
    _publication_style()
    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.labelsize": 17,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12.5,
            "axes.linewidth": 1.15,
        }
    )
    control_coordinate = np.linspace(0.0, 1.0, DENSE_POINTS)
    cvar_data = summary.loc[summary["strategy"] == "fixed"].sort_values(
        "alpha",
        ascending=False,
    )
    cvar_control = cvar_control_coordinate(
        cvar_data["alpha"].to_numpy(dtype=float),
        alpha_min,
    )
    qtl_control = qtl_fit.parameter_data / FIT_GAMMA_MAX
    cvar_fit = fit_asymmetric_peak_response(
        cvar_control,
        cvar_data["mean_final_ratio"].to_numpy(dtype=float),
        cvar_data["sem_final_ratio"].to_numpy(dtype=float),
    )
    qtl_response_fit = fit_asymmetric_peak_response(
        qtl_control,
        qtl_fit.ratio_data,
        qtl_fit.sem_data,
    )
    cvar_ratio = asymmetric_peak_model(
        control_coordinate,
        float(cvar_fit["peak_value"]),
        float(cvar_fit["peak_parameter"]),
        float(cvar_fit["left_curvature"]),
        float(cvar_fit["right_curvature"]),
    )
    qtl_ratio = asymmetric_peak_model(
        control_coordinate,
        float(qtl_response_fit["peak_value"]),
        float(qtl_response_fit["peak_parameter"]),
        float(qtl_response_fit["left_curvature"]),
        float(qtl_response_fit["right_curvature"]),
    )

    figure, axis = plt.subplots(figsize=(7.35, 5.45))
    axis.plot(
        control_coordinate,
        cvar_ratio,
        color="#1F4E79",
        linestyle="-",
        linewidth=2.35,
        label="Fixed CVaR response",
    )
    axis.plot(
        control_coordinate,
        qtl_ratio,
        color="#C4512D",
        linestyle=(0, (5, 3)),
        linewidth=2.35,
        label="Fixed QTL response",
    )
    axis.scatter(
        cvar_control,
        cvar_data["mean_final_ratio"].to_numpy(dtype=float),
        s=42,
        marker="o",
        facecolor="white",
        edgecolor="#1F4E79",
        linewidth=1.25,
        zorder=4,
        label="CVaR means",
    )
    axis.scatter(
        qtl_control,
        qtl_fit.ratio_data,
        s=42,
        marker="s",
        facecolor="white",
        edgecolor="#C4512D",
        linewidth=1.25,
        zorder=4,
        label="QTL means",
    )
    y_min, y_max = padded_limits(
        [
            cvar_ratio,
            qtl_ratio,
            cvar_data["mean_final_ratio"].to_numpy(dtype=float),
            qtl_fit.ratio_data,
        ],
        padding_fraction=0.10,
    )
    axis.set(
        xlabel=(
            r"Landscape-control coordinate "
            rf"$r=|\gamma|/{FIT_GAMMA_MAX:g}$"
        ),
        ylabel="Mean Final Ratio",
        xlim=(0.0, 1.0),
        ylim=(y_min, y_max),
    )
    axis.grid(
        True,
        color="#D4D4D4",
        linestyle=(0, (3, 3)),
        linewidth=0.75,
        alpha=0.75,
    )
    axis.tick_params(which="both", direction="in", top=False, right=True)
    axis.legend(
        loc="lower left",
        ncol=2,
        handlelength=2.7,
        columnspacing=1.25,
    )
    secondary_axis = axis.secondary_xaxis(
        "top",
        functions=(
            lambda coordinate: alpha_min**np.asarray(coordinate),
            lambda alpha: cvar_control_coordinate(
                np.maximum(
                    np.asarray(alpha),
                    np.finfo(float).tiny,
                ),
                alpha_min,
            ),
        ),
    )
    alpha_ticks = np.array([1.0, 0.8, 0.5, 0.2, alpha_min])
    secondary_axis.set_xticks(alpha_ticks)
    secondary_axis.set_xticklabels(
        [f"{alpha:g}" for alpha in alpha_ticks]
    )
    secondary_axis.set_xlabel(
        (
            r"Equivalent CVaR tail fraction "
            rf"$\alpha={alpha_min:g}^r="
            rf"\exp[-\ln(1/{alpha_min:g})|\gamma|/"
            rf"{FIT_GAMMA_MAX:g}]$"
        ),
        labelpad=8,
    )
    secondary_axis.tick_params(direction="in", pad=4)
    save_figure(figure, output_jpg, output_pdf, dpi=600)


def load_summary_from_output(path: Path) -> pd.DataFrame:
    """Load aggregate CVaR rows from a previously generated output CSV."""
    frame = pd.read_csv(path)
    required = {
        "record_type",
        "strategy",
        "alpha",
        "one_minus_alpha",
        "mean_final_ratio",
        "sem_final_ratio",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    summary = frame.loc[frame["record_type"] == "summary"].copy()
    if summary.empty:
        raise ValueError(f"{path} contains no summary records.")
    return summary


def run_experiment(
    *,
    settings: CVaRSettings,
    alphas: tuple[float, ...],
    workers: int,
    output_csv: Path,
    output_cvar_jpg: Path,
    output_cvar_pdf: Path,
    output_fixed_jpg: Path,
    output_fixed_pdf: Path,
    output_performance_jpg: Path,
    output_performance_pdf: Path,
    qtl_fixed_csv: Path,
    resume: bool = True,
) -> pd.DataFrame:
    """Execute, aggregate, fit, save, and plot the complete experiment."""
    alphas = validate_alphas(alphas, settings.alpha_start)
    qtl_fit = fit_fixed_qtl_curve(qtl_fixed_csv)
    tasks = build_tasks(settings, alphas)
    validate_paired_tasks(tasks)
    checkpoint_path = output_csv.with_suffix(output_csv.suffix + ".partial")
    started_at = time.time()
    restart_frame = execute_tasks(
        tasks,
        workers=workers,
        checkpoint_path=checkpoint_path,
        resume=resume,
    )
    graph_average, summary = aggregate_restart_results(restart_frame)
    cvar_fits = fit_cvar_summary(summary)
    output = build_output_frame(
        restart_frame,
        graph_average,
        summary,
        cvar_fits,
        qtl_fit,
        settings,
        alphas,
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_csv, index=False)
    plot_cvar_figure(
        summary,
        cvar_fits,
        output_cvar_jpg,
        output_cvar_pdf,
    )
    plot_fixed_alpha_figure(
        summary,
        cvar_fits["fixed"],
        output_fixed_jpg,
        output_fixed_pdf,
    )
    plot_performance_matching_functions(
        summary,
        qtl_fit,
        alpha_min=min(alphas),
        output_jpg=output_performance_jpg,
        output_pdf=output_performance_pdf,
    )
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    print(
        f"Saved {output_csv.name} and six figure files in "
        f"{time.time() - started_at:.2f}s.",
        flush=True,
    )
    return output


def replot(
    *,
    output_csv: Path,
    output_cvar_jpg: Path,
    output_cvar_pdf: Path,
    output_fixed_jpg: Path,
    output_fixed_pdf: Path,
    output_performance_jpg: Path,
    output_performance_pdf: Path,
    qtl_fixed_csv: Path,
) -> None:
    """Regenerate both figures from an existing CVaR CSV."""
    summary = load_summary_from_output(output_csv)
    fits = fit_cvar_summary(summary)
    qtl_fit = fit_fixed_qtl_curve(qtl_fixed_csv)
    refresh_fit_records(output_csv, fits, qtl_fit, summary)
    plot_cvar_figure(
        summary,
        fits,
        output_cvar_jpg,
        output_cvar_pdf,
    )
    plot_fixed_alpha_figure(
        summary,
        fits["fixed"],
        output_fixed_jpg,
        output_fixed_pdf,
    )
    plot_performance_matching_functions(
        summary,
        qtl_fit,
        alpha_min=float(summary["alpha"].min()),
        output_jpg=output_performance_jpg,
        output_pdf=output_performance_pdf,
    )


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line interface for simulation and data-only plots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, default=CVaRSettings.shots)
    parser.add_argument("--steps", type=int, default=CVaRSettings.steps)
    parser.add_argument(
        "--num-init-points",
        type=int,
        default=CVaRSettings.number_of_initializations,
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=list(DEFAULT_ALPHAS),
    )
    parser.add_argument(
        "--alpha-start",
        type=float,
        default=CVaRSettings.alpha_start,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--simulator",
        choices=["default.qubit", "lightning.qubit"],
        default=CVaRSettings.simulator,
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--cvar-jpg", type=Path, default=DEFAULT_CVAR_JPG)
    parser.add_argument("--cvar-pdf", type=Path, default=DEFAULT_CVAR_PDF)
    parser.add_argument(
        "--fixed-jpg",
        type=Path,
        default=DEFAULT_FIXED_JPG,
    )
    parser.add_argument(
        "--fixed-pdf",
        type=Path,
        default=DEFAULT_FIXED_PDF,
    )
    parser.add_argument(
        "--performance-jpg",
        type=Path,
        default=DEFAULT_PERFORMANCE_JPG,
    )
    parser.add_argument(
        "--performance-pdf",
        type=Path,
        default=DEFAULT_PERFORMANCE_PDF,
    )
    parser.add_argument(
        "--qtl-fixed-csv",
        type=Path,
        default=DEFAULT_QTL_FIXED_CSV,
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore any compatible CVaR.csv.partial checkpoint.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate figures from the existing --csv without simulations.",
    )
    return parser


def main() -> None:
    """Run the requested simulations or regenerate plots from saved data."""
    arguments = build_parser().parse_args()
    paths = {
        name: Path(value).resolve()
        for name, value in {
            "output_csv": arguments.csv,
            "output_cvar_jpg": arguments.cvar_jpg,
            "output_cvar_pdf": arguments.cvar_pdf,
            "output_fixed_jpg": arguments.fixed_jpg,
            "output_fixed_pdf": arguments.fixed_pdf,
            "output_performance_jpg": arguments.performance_jpg,
            "output_performance_pdf": arguments.performance_pdf,
            "qtl_fixed_csv": arguments.qtl_fixed_csv,
        }.items()
    }
    if arguments.plot_only:
        replot(**paths)
        return

    settings = CVaRSettings(
        shots=arguments.shots,
        steps=arguments.steps,
        number_of_initializations=arguments.num_init_points,
        alpha_start=arguments.alpha_start,
        simulator=arguments.simulator,
    )
    alphas = validate_alphas(arguments.alphas, settings.alpha_start)
    print(
        "CVaR experiment: "
        f"shots={settings.shots}, steps={settings.steps}, "
        f"initializations={settings.number_of_initializations}, "
        f"alphas={list(alphas)}, workers={arguments.workers}, "
        f"simulator={settings.simulator}",
        flush=True,
    )
    output = run_experiment(
        settings=settings,
        alphas=alphas,
        workers=arguments.workers,
        resume=not arguments.no_resume,
        **paths,
    )
    summary = output.loc[output["record_type"] == "summary"]
    print(
        summary[
            [
                "strategy",
                "alpha",
                "mean_final_ratio",
                "sem_final_ratio",
            ]
        ].to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
