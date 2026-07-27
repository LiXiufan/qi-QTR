"""MaxCut problem construction and classical result metrics."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class MaxCutProblem:
    """A graph together with its exactly enumerated cut landscape."""

    graph_family: str
    n: int
    seed: int
    edges: tuple[tuple[int, int], ...]
    cut_values: np.ndarray
    maximum_cut: float


def make_graph(n: int, family: str, seed: int) -> nx.Graph:
    """Create a reproducible connected graph from a supported family."""
    if family == "regular3":
        return nx.random_regular_graph(3, n, seed=seed)
    if family == "erdos_renyi":
        edge_probability = min(0.45, 3.0 / (n - 1))
        graph = nx.gnp_random_graph(n, edge_probability, seed=seed)
        retry_seed = seed
        while not nx.is_connected(graph):
            retry_seed += 1
            graph = nx.gnp_random_graph(n, edge_probability, seed=retry_seed)
        return graph
    raise ValueError(f"Unknown graph family: {family}")


def basis_states(n: int) -> np.ndarray:
    """Return all computational-basis states for ``n`` binary variables."""
    return np.array(
        [list(map(int, f"{state:0{n}b}")) for state in range(2**n)],
        dtype=int,
    )


def cut_values_from_samples(
    samples: np.ndarray,
    edges: tuple[tuple[int, int], ...] | list[tuple[int, int]],
) -> np.ndarray:
    """Calculate the cut value of every supplied bitstring."""
    samples = np.asarray(samples, dtype=int)
    cut_values = np.zeros(samples.shape[0], dtype=int)
    for node_u, node_v in edges:
        cut_values += (samples[:, node_u] != samples[:, node_v]).astype(int)
    return cut_values


def exact_cut_data(
    n: int,
    edges: tuple[tuple[int, int], ...] | list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, float]:
    """Enumerate every bitstring and return states, cuts, and exact optimum."""
    states = basis_states(n)
    cut_values = cut_values_from_samples(states, edges).astype(float)
    return states, cut_values, float(cut_values.max())


def build_maxcut_problem(n: int, family: str, seed: int) -> MaxCutProblem:
    """Build a graph and its exact MaxCut data."""
    graph = make_graph(n, family, seed)
    edges = tuple(sorted(graph.edges()))
    _, cut_values, maximum_cut = exact_cut_data(n, edges)
    return MaxCutProblem(
        graph_family=family,
        n=n,
        seed=seed,
        edges=edges,
        cut_values=cut_values,
        maximum_cut=maximum_cut,
    )


def summarize_distribution(
    probabilities: np.ndarray,
    cut_values: np.ndarray,
    maximum_cut: float,
) -> dict[str, float]:
    """Summarize a probability distribution over the MaxCut basis states."""
    probabilities = np.asarray(probabilities, dtype=float)
    cut_values = np.asarray(cut_values, dtype=float)
    optimal_mask = cut_values == maximum_cut
    support_mask = probabilities > 1e-12
    best_cut = (
        float(np.max(cut_values[support_mask]))
        if np.any(support_mask)
        else float(np.max(cut_values))
    )
    mean_cut = float(np.dot(probabilities, cut_values))
    return {
        "mean_cut": mean_cut,
        "mean_ratio": mean_cut / maximum_cut,
        "best_cut": best_cut,
        "best_ratio": best_cut / maximum_cut,
        "optimal_mass": float(probabilities[optimal_mask].sum()),
    }
