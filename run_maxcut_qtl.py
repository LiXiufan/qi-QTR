from __future__ import annotations

import argparse
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mplconfig_codex"))

import networkx as nx
import numpy as np
import pandas as pd
import pennylane as qml
from pennylane import numpy as qnp


SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_GRAPH_FAMILIES = ["erdos_renyi"]
DEFAULT_SIZES = [8]
DEFAULT_DEPTHS = [2]
DEFAULT_SEEDS = [0, 1, 2, 3, 4]

DEFAULT_FIXED_GAMMAS = [0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
DEFAULT_ASCENDING_FINAL_GAMMAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.5, 2.0]
DEFAULT_FIXED_SHOTS = [1000, 5000, 10000]

FIXED_INIT_BASE_SEED = 20260501
SCHEDULE_INIT_BASE_SEED = 20260429
DEFAULT_NUM_INIT_POINTS = 5
DEFAULT_TAIL_WINDOW = 10

BASE_CONFIG = {
    "graph_families": DEFAULT_GRAPH_FAMILIES,
    "sizes": DEFAULT_SIZES,
    "depths": DEFAULT_DEPTHS,
    "seeds": DEFAULT_SEEDS,
    "lr": 0.18,
    "lr_decay_power": 0.55,
    "lr_decay_offset": 6.0,
    "beta_mom": 0.7,
    "gamma_lr_penalty": 0.02,
    "grad_clip": 2.5,
    "steps": 100,
    "shots": 5000,
}


@dataclass(frozen=True)
class ObjectiveSpec:
    name: str
    kind: str
    gamma: float | None = None
    gamma_start: float | None = None
    gamma_end: float | None = None
    schedule: str | None = None


def sem(values: pd.Series | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    if len(arr) <= 1:
        return 0.0
    return float(arr.std(ddof=1) / np.sqrt(len(arr)))


def make_graph(n: int, family: str, seed: int) -> nx.Graph:
    if family == "regular3":
        return nx.random_regular_graph(3, n, seed=seed)
    if family == "erdos_renyi":
        p = min(0.45, 3.0 / (n - 1))
        graph = nx.gnp_random_graph(n, p, seed=seed)
        while not nx.is_connected(graph):
            seed += 1
            graph = nx.gnp_random_graph(n, p, seed=seed)
        return graph
    raise ValueError(f"Unknown graph family: {family}")


def basis_states(n: int) -> np.ndarray:
    return np.array([list(map(int, f"{i:0{n}b}")) for i in range(2**n)], dtype=int)


def cut_values_from_samples(samples: np.ndarray, edges: list[tuple[int, int]]) -> np.ndarray:
    samples = np.asarray(samples, dtype=int)
    cuts = np.zeros(samples.shape[0], dtype=int)
    for u, v in edges:
        cuts += (samples[:, u] != samples[:, v]).astype(int)
    return cuts


def exact_cut_data(n: int, edges: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray, float]:
    basis = basis_states(n)
    cuts = cut_values_from_samples(basis, edges)
    return basis, cuts.astype(float), float(cuts.max())


def qtl_loss_from_distribution(probs, losses, gamma: float | None):
    probs = qml.math.asarray(probs)
    losses = qml.math.asarray(losses)
    if gamma is None or abs(gamma) < 1e-12:
        return qml.math.dot(probs, losses)
    loss_min = qml.math.min(losses)
    shifted = losses - loss_min
    weighted = qml.math.sum(probs * qml.math.exp(-gamma * shifted))
    return loss_min - qml.math.log(weighted) / gamma


def scheduled_gamma(spec: ObjectiveSpec, step: int, total_steps: int) -> float | None:
    if spec.kind != "qtl_schedule":
        return spec.gamma
    if total_steps <= 1:
        return float(spec.gamma_end)
    frac = step / (total_steps - 1)
    if spec.schedule == "linear":
        schedule_frac = frac
    else:
        raise ValueError(f"Unsupported schedule: {spec.schedule}")
    return float(spec.gamma_start + schedule_frac * (spec.gamma_end - spec.gamma_start))


def objective_loss_from_distribution(
    probs,
    cuts: np.ndarray,
    c_max: float,
    spec: ObjectiveSpec,
    step: int,
    total_steps: int,
):
    losses = c_max - qml.math.asarray(cuts)
    if spec.kind == "expectation":
        return qml.math.dot(probs, losses)
    gamma = scheduled_gamma(spec, step, total_steps)
    return qtl_loss_from_distribution(probs, losses, gamma)


def make_qaoa_prob_qnode(
    n: int,
    edges: list[tuple[int, int]],
    *,
    shots: int,
    device_seed: int,
):
    dev = qml.device("default.qubit", wires=n, seed=int(device_seed))

    @qml.qnode(dev, interface="autograd", diff_method="parameter-shift")
    def base_circuit(params):
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

    return qml.set_shots(base_circuit, shots=shots)


def summarize_probs(probs: np.ndarray, cuts: np.ndarray, c_max: float) -> dict[str, float]:
    optimal_mask = cuts == c_max
    support_mask = probs > 1e-12
    best_cut = float(np.max(cuts[support_mask])) if np.any(support_mask) else float(np.max(cuts))
    return {
        "mean_cut": float(np.dot(probs, cuts)),
        "mean_ratio": float(np.dot(probs, cuts) / c_max),
        "best_cut": best_cut,
        "best_ratio": float(best_cut / c_max),
        "optimal_mass": float(probs[optimal_mask].sum()),
    }


def optimize_parameters(
    circuit,
    cuts: np.ndarray,
    c_max: float,
    init_params: np.ndarray,
    spec: ObjectiveSpec,
    config: dict,
) -> tuple[np.ndarray, pd.DataFrame]:
    params = qnp.array(init_params, requires_grad=True)
    momentum = np.zeros_like(np.asarray(init_params, dtype=float))
    best_cut_so_far = -np.inf
    evals_per_step = 2 * len(init_params) + 2
    history: list[dict[str, float | int]] = []

    for step in range(config["steps"]):
        def loss_fn(theta):
            wrapped = qnp.mod(theta, 2 * np.pi)
            probs = circuit(wrapped)
            return objective_loss_from_distribution(probs, cuts, c_max, spec, step, config["steps"])

        grad = np.asarray(qml.grad(loss_fn)(params), dtype=float)
        grad_norm = float(np.linalg.norm(grad))
        if grad_norm > config["grad_clip"]:
            grad = grad * (config["grad_clip"] / grad_norm)

        momentum = config["beta_mom"] * momentum + (1.0 - config["beta_mom"]) * grad
        step_lr = config["lr"] / ((step + 1 + config["lr_decay_offset"]) ** config["lr_decay_power"])
        gamma_now = scheduled_gamma(spec, step, config["steps"])
        if gamma_now is not None and config["gamma_lr_penalty"] > 0.0:
            step_lr = step_lr / (1.0 + config["gamma_lr_penalty"] * abs(gamma_now))

        updated = np.mod(np.asarray(params, dtype=float) - step_lr * momentum, 2 * np.pi)
        params = qnp.array(updated, requires_grad=True)

        probs = np.asarray(circuit(params), dtype=float)
        metrics = summarize_probs(probs, cuts, c_max)
        best_cut_so_far = max(best_cut_so_far, metrics["best_cut"])
        history.append(
            {
                "iteration": step + 1,
                "loss": float(loss_fn(params)),
                "mean_cut": metrics["mean_cut"],
                "mean_ratio": metrics["mean_ratio"],
                "best_ratio_step": metrics["best_ratio"],
                "best_ratio_so_far": float(best_cut_so_far / c_max),
                "best_cut_so_far": int(best_cut_so_far),
                "optimal_mass": metrics["optimal_mass"],
                "gamma": np.nan if gamma_now is None else float(gamma_now),
                "lr": float(step_lr),
                "grad_norm": float(np.linalg.norm(grad)),
                "circuit_evals": int((step + 1) * evals_per_step),
                "shots_used": int((step + 1) * evals_per_step * config["shots"]),
            }
        )

    return params, pd.DataFrame(history)


def run_single_setting(
    graph_family: str,
    n: int,
    p: int,
    seed: int,
    spec: ObjectiveSpec,
    config: dict,
    *,
    init_id: int,
    init_seed: int,
    init_params: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, object]]:
    graph = make_graph(n, graph_family, seed)
    edges = sorted(graph.edges())
    _, cuts, c_max = exact_cut_data(n, edges)
    circuit = make_qaoa_prob_qnode(
        n,
        edges,
        shots=config["shots"],
        device_seed=2024 + 100 * seed + 10 * p + n,
    )
    _, history = optimize_parameters(circuit, cuts, c_max, init_params, spec, config)
    history["graph_family"] = graph_family
    history["n"] = n
    history["p"] = p
    history["seed"] = seed
    history["objective"] = spec.name
    history["init_id"] = init_id

    final_row = history.iloc[-1]
    summary = {
        "graph_family": graph_family,
        "n": n,
        "p": p,
        "seed": seed,
        "objective": spec.name,
        "init_id": init_id,
        "init_seed": int(init_seed),
        "init_params": np.asarray(init_params, dtype=float).tolist(),
        "final_mean_ratio": float(final_row["mean_ratio"]),
        "final_best_ratio": float(final_row["best_ratio_so_far"]),
        "final_optimal_mass": float(final_row["optimal_mass"]),
        "final_loss": float(final_row["loss"]),
    }
    return history, summary


def build_init_points_by_depth(
    depths: list[int],
    *,
    base_seed: int,
    num_init_points: int,
    high: float,
) -> dict[int, list[tuple[int, np.ndarray]]]:
    init_points: dict[int, list[tuple[int, np.ndarray]]] = {}
    for p in depths:
        init_points[p] = []
        for init_id in range(num_init_points):
            init_seed = base_seed + 1000 * p + init_id
            rng = np.random.default_rng(init_seed)
            init_params = rng.uniform(0.0, high, size=2 * p)
            init_points[p].append((init_seed, init_params))
    return init_points


def fixed_config(shots: int, steps: int) -> dict:
    return {**BASE_CONFIG, "shots": shots, "steps": steps}


def ascending_config(shots: int, steps: int) -> dict:
    return {
        **BASE_CONFIG,
        "shots": shots,
        "steps": steps,
        "lr_decay_power": 0.35,
        "gamma_lr_penalty": 0.01,
    }


def comparison_config(shots: int, steps: int) -> dict:
    return {
        **BASE_CONFIG,
        "sizes": [8, 10, 12],
        "depths": [1, 2, 3],
        "shots": shots,
        "steps": steps,
        "lr": 0.12,
        "lr_decay_power": 0.35,
        "gamma_lr_penalty": 0.01,
    }


def fixed_output_paths(output_dir: Path, shots: int) -> tuple[Path, Path, Path]:
    return (
        output_dir / f"fixed_gamma_all_restarts(shot{shots}).csv",
        output_dir / f"fixed_gamma_restart_avg(shot{shots}).csv",
        output_dir / f"fixed_gamma_shot_{shots}.csv",
    )


def ascending_output_paths(output_dir: Path, shots: int) -> tuple[Path, Path, Path]:
    return (
        output_dir / f"schedule_gamma_all_restarts(shots{shots}).csv",
        output_dir / f"schedule_gamma_restart_avg(shots{shots}).csv",
        output_dir / f"schedule_gamma_restart_group(shots{shots}).csv",
    )


def comparison_output_paths(output_dir: Path, shots: int) -> tuple[Path, Path, Path]:
    return (
        output_dir / f"maxcut_compare_shot{shots}.csv",
        output_dir / f"maxcut_compare_avg_shot{shots}.csv",
        output_dir / f"maxcut_compare_group_shot{shots}.csv",
    )


def run_fixed_tilt(
    *,
    shots: int,
    steps: int,
    gamma_values: list[float],
    output_dir: Path,
    num_init_points: int,
    tail_window: int,
) -> None:
    config = fixed_config(shots, steps)
    init_points_by_depth = build_init_points_by_depth(
        config["depths"],
        base_seed=FIXED_INIT_BASE_SEED,
        num_init_points=num_init_points,
        high=np.pi,
    )
    restart_rows: list[dict[str, object]] = []
    start = time.time()

    for graph_family in config["graph_families"]:
        for n in config["sizes"]:
            for p in config["depths"]:
                for seed in config["seeds"]:
                    for gamma_plot in gamma_values:
                        if abs(gamma_plot) < 1e-12:
                            spec = ObjectiveSpec(name="Expectation", kind="expectation")
                            gamma_raw = 0.0
                        else:
                            spec = ObjectiveSpec(
                                name=f"QTL gamma={gamma_plot:.2f}",
                                kind="qtl",
                                gamma=float(gamma_plot),
                            )
                            gamma_raw = float(gamma_plot)

                        for init_id, (init_seed, init_params) in enumerate(init_points_by_depth[p]):
                            print(
                                f"[fixed] shots={shots} family={graph_family} n={n} p={p} seed={seed} "
                                f"gamma={gamma_plot:.2f} init={init_id}"
                            )
                            history, summary = run_single_setting(
                                graph_family,
                                n,
                                p,
                                seed,
                                spec,
                                config,
                                init_id=init_id,
                                init_seed=init_seed,
                                init_params=init_params,
                            )
                            restart_rows.append(
                                {
                                    "graph_family": graph_family,
                                    "n": n,
                                    "p": p,
                                    "seed": seed,
                                    "gamma_plot": float(gamma_plot),
                                    "gamma_raw": float(gamma_raw),
                                    "init_id": int(summary["init_id"]),
                                    "init_seed": int(summary["init_seed"]),
                                    "init_params": summary["init_params"],
                                    "final_mean_ratio": float(summary["final_mean_ratio"]),
                                    "tail_mean_ratio": float(history["mean_ratio"].tail(min(tail_window, len(history))).mean()),
                                    "peak_mean_ratio": float(history["mean_ratio"].max()),
                                    "final_optimal_mass": float(summary["final_optimal_mass"]),
                                }
                            )

    restart_df = pd.DataFrame(restart_rows)
    avg_df = (
        restart_df.groupby(["graph_family", "n", "p", "seed", "gamma_plot", "gamma_raw"])
        .agg(
            num_init_points=("init_id", "nunique"),
            final_mean_ratio=("final_mean_ratio", "mean"),
            tail_mean_ratio=("tail_mean_ratio", "mean"),
            peak_mean_ratio=("peak_mean_ratio", "mean"),
            final_optimal_mass=("final_optimal_mass", "mean"),
        )
        .reset_index()
    )
    summary_df = (
        avg_df.groupby("gamma_plot")
        .agg(
            mean_peak_ratio=("peak_mean_ratio", "mean"),
            sem_peak_ratio=("peak_mean_ratio", sem),
            mean_tail_ratio=("tail_mean_ratio", "mean"),
            sem_tail_ratio=("tail_mean_ratio", sem),
            mean_final_ratio=("final_mean_ratio", "mean"),
            sem_final_ratio=("final_mean_ratio", sem),
            mean_optimal_mass=("final_optimal_mass", "mean"),
        )
        .reset_index()
        .sort_values("gamma_plot")
    )

    restart_path, avg_path, summary_path = fixed_output_paths(output_dir, shots)
    restart_df.to_csv(restart_path, index=False)
    avg_df.to_csv(avg_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved {restart_path.name}, {avg_path.name}, {summary_path.name} in {time.time() - start:.2f}s")


def run_ascending_tilt(
    *,
    shots: int,
    steps: int,
    final_gamma_values: list[float],
    output_dir: Path,
    num_init_points: int,
    tail_window: int,
) -> None:
    config = ascending_config(shots, steps)
    init_points_by_depth = build_init_points_by_depth(
        config["depths"],
        base_seed=SCHEDULE_INIT_BASE_SEED,
        num_init_points=num_init_points,
        high=2 * np.pi,
    )
    restart_rows: list[dict[str, object]] = []
    start = time.time()

    for graph_family in config["graph_families"]:
        for n in config["sizes"]:
            for p in config["depths"]:
                for seed in config["seeds"]:
                    for gamma_plot in final_gamma_values:
                        spec = ObjectiveSpec(
                            name=f"QTL linear schedule to {gamma_plot:.2f}",
                            kind="qtl_schedule",
                            gamma_start=0.0,
                            gamma_end=float(gamma_plot),
                            schedule="linear",
                        )
                        for init_id, (init_seed, init_params) in enumerate(init_points_by_depth[p]):
                            print(
                                f"[ascending] shots={shots} family={graph_family} n={n} p={p} seed={seed} "
                                f"gamma_end={gamma_plot:.2f} init={init_id}"
                            )
                            history, summary = run_single_setting(
                                graph_family,
                                n,
                                p,
                                seed,
                                spec,
                                config,
                                init_id=init_id,
                                init_seed=init_seed,
                                init_params=init_params,
                            )
                            restart_rows.append(
                                {
                                    "graph_family": graph_family,
                                    "n": n,
                                    "p": p,
                                    "seed": seed,
                                    "gamma_plot": float(gamma_plot),
                                    "gamma_start": 0.0,
                                    "gamma_end": float(gamma_plot),
                                    "init_id": int(summary["init_id"]),
                                    "init_seed": int(summary["init_seed"]),
                                    "init_params": summary["init_params"],
                                    "final_mean_ratio": float(summary["final_mean_ratio"]),
                                    "tail_mean_ratio": float(history["mean_ratio"].tail(min(tail_window, len(history))).mean()),
                                    "peak_mean_ratio": float(history["mean_ratio"].max()),
                                    "final_optimal_mass": float(summary["final_optimal_mass"]),
                                }
                            )

    restart_df = pd.DataFrame(restart_rows)
    avg_df = (
        restart_df.groupby(["graph_family", "n", "p", "seed", "gamma_plot", "gamma_start", "gamma_end"])
        .agg(
            num_init_points=("init_id", "nunique"),
            final_mean_ratio=("final_mean_ratio", "mean"),
            tail_mean_ratio=("tail_mean_ratio", "mean"),
            peak_mean_ratio=("peak_mean_ratio", "mean"),
            final_optimal_mass=("final_optimal_mass", "mean"),
        )
        .reset_index()
    )
    summary_df = (
        avg_df.groupby("gamma_plot")
        .agg(
            mean_peak_ratio=("peak_mean_ratio", "mean"),
            sem_peak_ratio=("peak_mean_ratio", sem),
            mean_tail_ratio=("tail_mean_ratio", "mean"),
            sem_tail_ratio=("tail_mean_ratio", sem),
            mean_final_ratio=("final_mean_ratio", "mean"),
            sem_final_ratio=("final_mean_ratio", sem),
            mean_optimal_mass=("final_optimal_mass", "mean"),
        )
        .reset_index()
        .sort_values("gamma_plot")
    )

    restart_path, avg_path, summary_path = ascending_output_paths(output_dir, shots)
    restart_df.to_csv(restart_path, index=False)
    avg_df.to_csv(avg_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved {restart_path.name}, {avg_path.name}, {summary_path.name} in {time.time() - start:.2f}s")


def run_comparison(
    *,
    shots: int,
    steps: int,
    output_dir: Path,
    num_init_points: int,
) -> None:
    config = comparison_config(shots, steps)
    init_points_by_depth = build_init_points_by_depth(
        config["depths"],
        base_seed=SCHEDULE_INIT_BASE_SEED,
        num_init_points=num_init_points,
        high=2 * np.pi,
    )
    objectives = [
        ObjectiveSpec("Expectation", "expectation"),
        ObjectiveSpec("Fixed QTL", "qtl", gamma=0.4),
        ObjectiveSpec("Ascending QTL", "qtl_schedule", gamma_start=0.0, gamma_end=0.8, schedule="linear"),
    ]

    restart_rows: list[dict[str, object]] = []
    start = time.time()

    for graph_family in config["graph_families"]:
        for n in config["sizes"]:
            for p in config["depths"]:
                for seed in config["seeds"]:
                    for spec in objectives:
                        for init_id, (init_seed, init_params) in enumerate(init_points_by_depth[p]):
                            print(
                                f"[comparison] shots={shots} family={graph_family} n={n} p={p} seed={seed} "
                                f"objective={spec.name} init={init_id}"
                            )
                            history, summary = run_single_setting(
                                graph_family,
                                n,
                                p,
                                seed,
                                spec,
                                config,
                                init_id=init_id,
                                init_seed=init_seed,
                                init_params=init_params,
                            )
                            restart_rows.append(
                                {
                                    "graph_family": graph_family,
                                    "n": n,
                                    "p": p,
                                    "seed": seed,
                                    "objective": spec.name,
                                    "init_id": int(summary["init_id"]),
                                    "init_seed": int(summary["init_seed"]),
                                    "final_mean_ratio": float(summary["final_mean_ratio"]),
                                    "peak_mean_ratio": float(history["mean_ratio"].max()),
                                    "final_optimal_mass": float(summary["final_optimal_mass"]),
                                }
                            )

    restart_df = pd.DataFrame(restart_rows)
    avg_df = (
        restart_df.groupby(["graph_family", "n", "p", "seed", "objective"])
        .agg(
            num_init_points=("init_id", "nunique"),
            final_mean_ratio=("final_mean_ratio", "mean"),
            peak_mean_ratio=("peak_mean_ratio", "mean"),
            final_optimal_mass=("final_optimal_mass", "mean"),
        )
        .reset_index()
    )
    summary_df = (
        avg_df.groupby(["n", "p", "objective"])
        .agg(
            mean_final_ratio=("final_mean_ratio", "mean"),
            sem_final_ratio=("final_mean_ratio", sem),
            mean_peak_ratio=("peak_mean_ratio", "mean"),
            mean_optimal_mass=("final_optimal_mass", "mean"),
        )
        .reset_index()
        .sort_values(["p", "n", "objective"])
    )

    restart_path, avg_path, summary_path = comparison_output_paths(output_dir, shots)
    restart_df.to_csv(restart_path, index=False)
    avg_df.to_csv(avg_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved {restart_path.name}, {avg_path.name}, {summary_path.name} in {time.time() - start:.2f}s")


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clean Python entry point for the NeurIPS MaxCut QTL experiments."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    fixed_parser = subparsers.add_parser("fixed", help="Run the fixed-tilt gamma sweep.")
    fixed_parser.add_argument("--shots", type=int, default=5000)
    fixed_parser.add_argument("--steps", type=int, default=100)
    fixed_parser.add_argument("--num-init-points", type=int, default=DEFAULT_NUM_INIT_POINTS)
    fixed_parser.add_argument("--tail-window", type=int, default=DEFAULT_TAIL_WINDOW)
    fixed_parser.add_argument(
        "--gammas",
        type=parse_float_list,
        default=DEFAULT_FIXED_GAMMAS,
        help="Comma-separated fixed gamma values.",
    )
    fixed_parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)

    ascending_parser = subparsers.add_parser("ascending", help="Run the ascending-tilt sweep.")
    ascending_parser.add_argument("--shots", type=int, default=5000)
    ascending_parser.add_argument("--steps", type=int, default=100)
    ascending_parser.add_argument("--num-init-points", type=int, default=DEFAULT_NUM_INIT_POINTS)
    ascending_parser.add_argument("--tail-window", type=int, default=DEFAULT_TAIL_WINDOW)
    ascending_parser.add_argument(
        "--gamma-ends",
        type=parse_float_list,
        default=DEFAULT_ASCENDING_FINAL_GAMMAS,
        help="Comma-separated final gamma values for the linear schedule.",
    )
    ascending_parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)

    comparison_parser = subparsers.add_parser("comparison", help="Run the fixed-vs-ascending comparison.")
    comparison_parser.add_argument("--shots", type=int, default=5000)
    comparison_parser.add_argument("--steps", type=int, default=100)
    comparison_parser.add_argument("--num-init-points", type=int, default=DEFAULT_NUM_INIT_POINTS)
    comparison_parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)

    all_parser = subparsers.add_parser("all", help="Run the full submission pipeline.")
    all_parser.add_argument("--steps", type=int, default=100)
    all_parser.add_argument("--num-init-points", type=int, default=DEFAULT_NUM_INIT_POINTS)
    all_parser.add_argument("--tail-window", type=int, default=DEFAULT_TAIL_WINDOW)
    all_parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.command == "fixed":
        run_fixed_tilt(
            shots=args.shots,
            steps=args.steps,
            gamma_values=args.gammas,
            output_dir=output_dir,
            num_init_points=args.num_init_points,
            tail_window=args.tail_window,
        )
        return

    if args.command == "ascending":
        run_ascending_tilt(
            shots=args.shots,
            steps=args.steps,
            final_gamma_values=args.gamma_ends,
            output_dir=output_dir,
            num_init_points=args.num_init_points,
            tail_window=args.tail_window,
        )
        return

    if args.command == "comparison":
        run_comparison(
            shots=args.shots,
            steps=args.steps,
            output_dir=output_dir,
            num_init_points=args.num_init_points,
        )
        return

    if args.command == "all":
        for shots in DEFAULT_FIXED_SHOTS:
            run_fixed_tilt(
                shots=shots,
                steps=args.steps,
                gamma_values=DEFAULT_FIXED_GAMMAS,
                output_dir=output_dir,
                num_init_points=args.num_init_points,
                tail_window=args.tail_window,
            )
        run_ascending_tilt(
            shots=5000,
            steps=args.steps,
            final_gamma_values=DEFAULT_ASCENDING_FINAL_GAMMAS,
            output_dir=output_dir,
            num_init_points=args.num_init_points,
            tail_window=args.tail_window,
        )
        run_comparison(
            shots=5000,
            steps=args.steps,
            output_dir=output_dir,
            num_init_points=args.num_init_points,
        )
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
