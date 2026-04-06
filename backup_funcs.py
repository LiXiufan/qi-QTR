
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time
import math
from dataclasses import dataclass

from IPython.display import display
import pennylane as qml
from pennylane import numpy as qnp

import networkx as nx

plt.style.use("seaborn-v0_8-whitegrid")
np.random.seed(7)


@dataclass(frozen=True)
class ObjectiveSpec:
    name: str
    kind: str
    alpha: float | None = None
    gamma: float | None = None
    gamma_start: float | None = None
    gamma_end: float | None = None
    schedule: str | None = None
    color: str = "C0"


MAIN_OBJECTIVES = [
    ObjectiveSpec("Expectation", "expectation", color="#1f77b4"),
    ObjectiveSpec("CVaR 0.2", "cvar", alpha=0.2, color="#ff7f0e"),
    ObjectiveSpec("QTL gamma=5.0", "qtl", gamma=5.0, color="#2ca02c"),
    ObjectiveSpec("QTL gamma=20.0", "qtl", gamma=20.0, color="#8c564b"),
    ObjectiveSpec(
        "QTL schedule linear",
        "qtl_schedule",
        gamma_start=0,
        gamma_end=200,
        schedule="linear",
        color="#9467bd",
    ),
]

PARAMSHIFT_OBJECTIVES = [
    ObjectiveSpec("Expectation", "expectation", color="#1f77b4"),
    ObjectiveSpec("CVaR 0.2", "cvar", alpha=0.2, color="#ff7f0e"),
    ObjectiveSpec("QTL gamma=5.0", "qtl", gamma=5.0, color="#2ca02c"),
    ObjectiveSpec("QTL gamma=20.0", "qtl", gamma=20.0, color="#8c564b"),
    ObjectiveSpec(
        "QTL schedule linear",
        "qtl_schedule",
        gamma_start=0,
        gamma_end=200,
        schedule="linear",
        color="#9467bd",
    ),
]

SCHEDULE_OBJECTIVES = [
    ObjectiveSpec("Expectation", "expectation", color="#1f77b4"),
    ObjectiveSpec("QTL gamma=20.0", "qtl", gamma=20.0, color="#8c564b"),
    ObjectiveSpec(
        "QTL schedule linear",
        "qtl_schedule",
        gamma_start=0,
        gamma_end=200,
        schedule="linear",
        color="#9467bd",
    ),
    ObjectiveSpec(
        "QTL schedule sigmoid",
        "qtl_schedule",
        gamma_start=0,
        gamma_end=200,
        schedule="sigmoid",
        color="#e377c2",
    ),
    ObjectiveSpec(
        "QTL reverse linear",
        "qtl_schedule",
        gamma_start=200,
        gamma_end=0,
        schedule="linear",
        color="#7f7f7f",
    ),
]

FIXED_GAMMA_GRID = [0.5, 5, 20, 100, 200]
GAMMA_GRID_OBJECTIVES = [
    ObjectiveSpec(f"QTL gamma={gamma}", "qtl", gamma=gamma, color=plt.cm.viridis(i / (len(FIXED_GAMMA_GRID) - 1)))
    for i, gamma in enumerate(FIXED_GAMMA_GRID)
] + [
    ObjectiveSpec("Expectation", "expectation", color="#1f77b4"),
    ObjectiveSpec("CVaR 0.2", "cvar", alpha=0.2, color="#ff7f0e"),
    ObjectiveSpec(
        "QTL schedule linear",
        "qtl_schedule",
        gamma_start=0.25,
        gamma_end=200,
        schedule="linear",
        color="#9467bd",
    ),
]

CURVE_OBJECTIVES = ["Expectation", "CVaR 0.2", "QTL gamma=5.0", "QTL gamma=20.0", "QTL schedule linear"]
HARD_OBJECTIVES = ["Expectation", "CVaR 0.2", "QTL gamma=20.0", "QTL schedule linear"]
BUDGET_OBJECTIVES = ["Expectation", "CVaR 0.2", "QTL gamma=20.0", "QTL schedule linear"]

ALL_OBJECTIVES = {
    obj.name: obj
    for obj in MAIN_OBJECTIVES + SCHEDULE_OBJECTIVES + GAMMA_GRID_OBJECTIVES
}
OBJECTIVE_COLORS = {name: obj.color for name, obj in ALL_OBJECTIVES.items()}

FAST_CONFIG = {
    # "graph_families": ["regular3"],
    "graph_families": ["erdos_renyi"],
    "sizes": [14],
    # "sizes": [8, 10, 12],
    # "depths": [1, 2, 3],
    "depths": [3],
    "seeds": [0, 1, 2],
    "lr": 0.18,
    "lr_decay_power": 0.55,
    "lr_decay_offset": 6.0,
    "beta_mom": 0.7,
    "gamma_lr_penalty": 0.02,
    "grad_clip": 2.5,
    "steps": 150,
    "shots": 1024,
    "gradient_probes": 4,
    "target_ratio": 0.99,
    "curve_n": 14,
    "summary_n": 14,
    "summary_p": 3,
    "budget_n": 14,
    "budget_p": 3,
    "budget_seeds": [0, 1, 2],
    "budget_shots_grid": [64, 128, 256, 512],
    "budget_total_shots": 15360,
    "gamma_grid_n": 14,
    "gamma_grid_p": 3,
    "schedule_n": 14,
    "schedule_p": 3,
}

FULL_CONFIG = {
    **FAST_CONFIG,
    "graph_families": ["erdos_renyi"],
    # "graph_families": ["regular3", "erdos_renyi"],
    "seeds": list(range(8)),
    "steps": 150,
    "shots": 1024,
    "gradient_probes": 8,
    "budget_seeds": list(range(8)),
}

RUN_CONFIG = FAST_CONFIG
print("Running config:", RUN_CONFIG)

EXACT_SMALL_SCALE_CONFIG = {
    **FAST_CONFIG,
    "sizes": [6],
    "depths": [1],
    "seeds": [0],
    "steps": 150,
    "shots": 256,
}




def spsa_optimize(evaluate, init_params, steps, shots, rng, a=0.18, c=0.12, A=8.0):
    params = np.asarray(init_params, dtype=float).copy()
    best_cut_so_far = -np.inf
    history = []

    for step in range(steps):
        delta = rng.choice([-1.0, 1.0], size=len(params))
        ck = c / ((step + 1) ** 0.101)
        ak = a / ((step + 1 + A) ** 0.602)

        loss_plus, plus_metrics = evaluate(params + ck * delta, step)
        loss_minus, minus_metrics = evaluate(params - ck * delta, step)
        grad = (loss_plus - loss_minus) / (2.0 * ck) * delta

        params = np.mod(params - ak * grad, 2 * np.pi)
        loss_cur, cur_metrics = evaluate(params, step)

        best_cut_so_far = max(
            best_cut_so_far,
            plus_metrics["best_cut"],
            minus_metrics["best_cut"],
            cur_metrics["best_cut"],
        )

        history.append(
            {
                "iteration": step + 1,
                "loss": float(loss_cur),
                "mean_ratio": float(cur_metrics["mean_ratio"]),
                "best_ratio_step": float(cur_metrics["best_ratio"]),
                "best_ratio_so_far": float(best_cut_so_far / cur_metrics["c_max"]),
                "best_cut_so_far": int(best_cut_so_far),
                "shots_used": int((step + 1) * 3 * shots),
                "gamma": cur_metrics["gamma"],
            }
        )

    return params, pd.DataFrame(history)


def cvar_loss(losses, alpha):
    k = max(1, int(math.ceil(alpha * len(losses))))
    return float(np.mean(np.partition(losses, k - 1)[:k]))


def qtl_loss(losses, gamma):
    losses = np.asarray(losses, dtype=float)
    if gamma is None or abs(gamma) < 1e-12:
        return float(np.mean(losses))
    loss_min = float(losses.min())
    shifted = losses - loss_min
    return float(loss_min - np.log(np.mean(np.exp(-gamma * shifted))) / gamma)


def sampled_loss(cuts, c_max, spec, step, total_steps):
    losses = c_max - np.asarray(cuts, dtype=float)
    if spec.kind == "expectation":
        return float(np.mean(losses)), np.nan
    gamma = scheduled_gamma(spec, step, total_steps)
    return qtl_loss(losses, gamma), gamma

def cvar_loss_from_distribution(probs, losses, alpha):
    if alpha is None or not (0.0 < alpha <= 1.0):
        raise ValueError("CVaR alpha must lie in (0, 1].")
    order = np.argsort(np.asarray(losses, dtype=float))
    ordered_losses = qml.math.asarray(np.asarray(losses, dtype=float)[order])
    ordered_probs = probs[order]
    cum_probs = qml.math.cumsum(ordered_probs)
    prev_cum_probs = cum_probs - ordered_probs
    tail_mass = qml.math.minimum(ordered_probs, qml.math.maximum(0.0, alpha - prev_cum_probs))
    return qml.math.sum(tail_mass * ordered_losses) / alpha


def exact_loss_from_distribution(probs, cuts, c_max, spec, step, total_steps):
    losses = c_max - cuts
    if spec.kind == "expectation":
        return float(np.dot(probs, losses)), np.nan
    gamma = scheduled_gamma(spec, step, total_steps)
    return qtl_loss_from_distribution(probs, losses, gamma), gamma





    
def run_budget_tradeoff(config, objective_names):
    rows = []
    specs = [ALL_OBJECTIVES[name] for name in objective_names]

    for shots in config["budget_shots_grid"]:
        steps = max(8, config["budget_total_shots"] // (3 * shots))
        local_config = {
            **config,
            "graph_families": config["graph_families"][:1],
            "sizes": [config["budget_n"]],
            "depths": [config["budget_p"]],
            "seeds": config["budget_seeds"],
            "steps": int(steps),
            "shots": int(shots),
            "gradient_probes": 0,
        }
        print(f"Shot budget sweep: shots={shots}, steps={steps}")
        for seed in local_config["seeds"]:
            for spec in specs:
                _, final_summary, _ = run_single_setting(
                    local_config["graph_families"][0],
                    local_config["sizes"][0],
                    local_config["depths"][0],
                    seed,
                    spec,
                    local_config,
                )
                final_summary["shots_per_eval"] = shots
                rows.append(final_summary)
    return pd.DataFrame(rows)


def annotate_instance_difficulty(final_df, quantile=0.4):
    exp_df = final_df[final_df["objective"] == "Expectation"][
        ["graph_family", "n", "p", "seed", "final_mean_ratio"]
    ].rename(columns={"final_mean_ratio": "expectation_final_ratio"})
    thresholds = (
        exp_df.groupby(["graph_family", "n", "p"])["expectation_final_ratio"]
        .quantile(quantile)
        .rename("hard_threshold")
        .reset_index()
    )
    annotated = final_df.merge(exp_df, on=["graph_family", "n", "p", "seed"], how="left")
    annotated = annotated.merge(thresholds, on=["graph_family", "n", "p"], how="left")
    annotated["difficulty"] = np.where(
        annotated["expectation_final_ratio"] <= annotated["hard_threshold"],
        "hard-for-expectation",
        "easy-for-expectation",
    )
    return annotated


def make_exact_objective(n, edges, spec, total_steps):
    _, cuts, c_max = exact_cut_data(n, edges)
    probs_fn = make_exact_qaoa_probs(n, edges)

    def objective(params, step):
        probs = np.asarray(probs_fn(np.mod(np.asarray(params, dtype=float), 2 * np.pi)), dtype=float)
        loss, gamma = exact_loss_from_distribution(probs, cuts, c_max, spec, step, total_steps)
        return float(loss), {
            "mean_ratio": float(np.dot(probs, cuts) / c_max),
            "optimal_mass": float(probs[cuts == c_max].sum()),
            "gamma": gamma,
        }

    return objective, cuts, c_max


def finite_difference_grad(objective, params, step, eps=1e-3):
    grad = np.zeros_like(params, dtype=float)
    for i in range(len(params)):
        shift = np.zeros_like(params)
        shift[i] = eps
        plus, _ = objective(params + shift, step)
        minus, _ = objective(params - shift, step)
        grad[i] = (plus - minus) / (2.0 * eps)
    return grad


def make_qaoa_sampler(n, edges, shots, device_seed):
    dev = qml.device("default.qubit", wires=n, seed=int(device_seed))

    @qml.set_shots(shots)
    @qml.qnode(dev, diff_method=None)
    def circuit(params):
        p = len(params) // 2
        gammas = params[:p]
        betas = params[p:]
        for wire in range(n):
            qml.Hadamard(wires=wire)
        for gamma, beta in zip(gammas, betas):
            for u, v in edges:
                qml.IsingZZ(2.0 * gamma, wires=[u, v])
            for wire in range(n):
                qml.RX(2.0 * beta, wires=wire)
        return qml.sample(wires=range(n))

    return circuit





def make_exact_qaoa_probs(n, edges):
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev, diff_method=None)
    def circuit(params):
        p = len(params) // 2
        gammas = params[:p]
        betas = params[p:]
        for wire in range(n):
            qml.Hadamard(wires=wire)
        for gamma, beta in zip(gammas, betas):
            for u, v in edges:
                qml.IsingZZ(2.0 * gamma, wires=[u, v])
            for wire in range(n):
                qml.RX(2.0 * beta, wires=wire)
        return qml.probs(wires=range(n))

    return circuit


def make_evaluator(n, edges, c_max, shots, spec, total_steps, device_seed):
    sampler = make_qaoa_sampler(n, edges, shots, device_seed=device_seed)

    def evaluate(params, step):
        params = np.asarray(params, dtype=float)
        wrapped = np.mod(params, 2 * np.pi)
        samples = np.asarray(sampler(wrapped), dtype=int)
        cuts = cut_values_from_samples(samples, edges)
        loss, gamma = sampled_loss(cuts, c_max, spec, step, total_steps)
        return loss, {
            "mean_cut": float(np.mean(cuts)),
            "best_cut": int(np.max(cuts)),
            "mean_ratio": float(np.mean(cuts) / c_max),
            "best_ratio": float(np.max(cuts) / c_max),
            "c_max": float(c_max),
            "gamma": gamma,
        }

    return evaluate


def estimate_initial_statistics(evaluate, init_params, probes, rng, epsilon=0.15):
    if probes <= 0:
        return {
            "grad_norm_mean": np.nan,
            "grad_norm_var": np.nan,
            "objective_var": np.nan,
        }
    norms = []
    values = []
    for _ in range(probes):
        delta = rng.choice([-1.0, 1.0], size=len(init_params))
        loss_plus, _ = evaluate(init_params + epsilon * delta, step=0)
        loss_minus, _ = evaluate(init_params - epsilon * delta, step=0)
        grad = (loss_plus - loss_minus) / (2.0 * epsilon) * delta
        norms.append(np.linalg.norm(grad))
        values.extend([loss_plus, loss_minus])
    return {
        "grad_norm_mean": float(np.mean(norms)),
        "grad_norm_var": float(np.var(norms)),
        "objective_var": float(np.var(values)),
    }













