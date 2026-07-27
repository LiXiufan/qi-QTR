"""QAOA circuit ansatz and reproducible parameter initialization."""

from __future__ import annotations

import numpy as np
import pennylane as qml


def qaoa_ansatz(
    parameters,
    n: int,
    edges: tuple[tuple[int, int], ...] | list[tuple[int, int]],
) -> None:
    """Apply a standard MaxCut QAOA ansatz to the active PennyLane tape."""
    if len(parameters) % 2:
        raise ValueError("QAOA requires an even number of parameters.")
    depth = len(parameters) // 2
    cost_angles = parameters[:depth]
    mixer_angles = parameters[depth:]

    for wire in range(n):
        qml.Hadamard(wires=wire)
    for cost_angle, mixer_angle in zip(cost_angles, mixer_angles):
        for node_u, node_v in edges:
            qml.IsingZZ(2.0 * cost_angle, wires=[node_u, node_v])
        for wire in range(n):
            qml.RX(2.0 * mixer_angle, wires=wire)


def make_probability_qnode(
    n: int,
    edges: tuple[tuple[int, int], ...] | list[tuple[int, int]],
    *,
    shots: int,
    device_seed: int,
    simulator: str = "default.qubit",
):
    """Construct a finite-shot QAOA probability QNode."""
    device = qml.device(simulator, wires=n, seed=int(device_seed))

    @qml.qnode(device, interface="autograd", diff_method="parameter-shift")
    def circuit(parameters):
        qaoa_ansatz(parameters, n, edges)
        return qml.probs(wires=range(n))

    return qml.set_shots(circuit, shots=shots)


def random_initial_parameters(
    depth: int,
    seed: int,
    *,
    upper_bound: float = 2.0 * np.pi,
) -> np.ndarray:
    """Draw a reproducible QAOA parameter vector."""
    generator = np.random.default_rng(seed)
    return generator.uniform(0.0, upper_bound, size=2 * depth)


def build_initial_points(
    depths: list[int],
    *,
    base_seed: int,
    number_of_points: int,
    upper_bound: float = 2.0 * np.pi,
) -> dict[int, list[tuple[int, np.ndarray]]]:
    """Create reproducible initial points for every requested QAOA depth."""
    points: dict[int, list[tuple[int, np.ndarray]]] = {}
    for depth in depths:
        points[depth] = []
        for initialization_id in range(number_of_points):
            seed = base_seed + 1000 * depth + initialization_id
            parameters = random_initial_parameters(
                depth,
                seed,
                upper_bound=upper_bound,
            )
            points[depth].append((seed, parameters))
    return points
