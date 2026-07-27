"""Command-line entry point for QTL MaxCut experiment sweeps."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments import (
    DEFAULT_ASCENDING_AVERAGE_GAMMAS,
    DEFAULT_FIXED_GAMMAS,
    DEFAULT_FIXED_SHOTS,
    DEFAULT_NUMBER_OF_INITIALIZATIONS,
    run_ascending_tilt_experiment,
    run_fixed_tilt_experiment,
    run_scale_benchmark,
)


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_float_list(raw_value: str) -> list[float]:
    """Parse a comma-separated list of floating-point values."""
    return [
        float(item.strip())
        for item in raw_value.split(",")
        if item.strip()
    ]


def build_parser() -> argparse.ArgumentParser:
    """Build the experiment command-line interface."""
    parser = argparse.ArgumentParser(
        description="Run the QTL MaxCut experiments."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    fixed_parser = subparsers.add_parser(
        "fixed",
        help="Run the fixed-tilt sweep.",
    )
    fixed_parser.add_argument("--shots", type=int, default=5000)
    fixed_parser.add_argument("--steps", type=int, default=100)
    fixed_parser.add_argument(
        "--num-init-points",
        type=int,
        default=DEFAULT_NUMBER_OF_INITIALIZATIONS,
    )
    fixed_parser.add_argument("--tail-window", type=int, default=10)
    fixed_parser.add_argument("--workers", type=int, default=1)
    fixed_parser.add_argument(
        "--simulator",
        choices=["default.qubit", "lightning.qubit"],
        default="default.qubit",
    )
    fixed_parser.add_argument(
        "--gammas",
        type=parse_float_list,
        default=DEFAULT_FIXED_GAMMAS,
    )
    fixed_parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)

    ascending_parser = subparsers.add_parser(
        "ascending",
        help="Run the linearly ascending-tilt sweep.",
    )
    ascending_parser.add_argument("--shots", type=int, default=5000)
    ascending_parser.add_argument("--steps", type=int, default=100)
    ascending_parser.add_argument(
        "--num-init-points",
        type=int,
        default=DEFAULT_NUMBER_OF_INITIALIZATIONS,
    )
    ascending_parser.add_argument("--tail-window", type=int, default=10)
    ascending_parser.add_argument("--workers", type=int, default=1)
    ascending_parser.add_argument(
        "--simulator",
        choices=["default.qubit", "lightning.qubit"],
        default="default.qubit",
    )
    ascending_parser.add_argument(
        "--average-gammas",
        "--gamma-ends",
        dest="average_gammas",
        type=parse_float_list,
        default=DEFAULT_ASCENDING_AVERAGE_GAMMAS,
        help=(
            "Comma-separated average tilts. --gamma-ends is retained as "
            "a compatibility alias but has the same average-gamma meaning."
        ),
    )
    ascending_parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR,
    )

    scale_parser = subparsers.add_parser(
        "scale",
        aliases=["comparison"],
        help="Run the size/depth benchmark.",
    )
    scale_parser.add_argument("--shots", type=int, default=5000)
    scale_parser.add_argument("--steps", type=int, default=100)
    scale_parser.add_argument(
        "--num-init-points",
        type=int,
        default=DEFAULT_NUMBER_OF_INITIALIZATIONS,
    )
    scale_parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
    scale_parser.add_argument(
        "--simulator",
        choices=["default.qubit", "lightning.qubit"],
        default="default.qubit",
    )

    all_parser = subparsers.add_parser(
        "all",
        help="Run every reported experiment.",
    )
    all_parser.add_argument("--steps", type=int, default=100)
    all_parser.add_argument(
        "--num-init-points",
        type=int,
        default=DEFAULT_NUMBER_OF_INITIALIZATIONS,
    )
    all_parser.add_argument("--tail-window", type=int, default=10)
    all_parser.add_argument("--workers", type=int, default=1)
    all_parser.add_argument(
        "--simulator",
        choices=["default.qubit", "lightning.qubit"],
        default="default.qubit",
    )
    all_parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
    return parser


def main() -> None:
    """Dispatch one experiment family from parsed CLI arguments."""
    arguments = build_parser().parse_args()
    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if arguments.command == "fixed":
        run_fixed_tilt_experiment(
            shots=arguments.shots,
            steps=arguments.steps,
            gamma_values=arguments.gammas,
            output_dir=output_dir,
            number_of_initializations=arguments.num_init_points,
            tail_window=arguments.tail_window,
            workers=arguments.workers,
            simulator=arguments.simulator,
        )
    elif arguments.command == "ascending":
        run_ascending_tilt_experiment(
            shots=arguments.shots,
            steps=arguments.steps,
            average_gamma_values=arguments.average_gammas,
            output_dir=output_dir,
            number_of_initializations=arguments.num_init_points,
            tail_window=arguments.tail_window,
            workers=arguments.workers,
            simulator=arguments.simulator,
        )
    elif arguments.command in {"scale", "comparison"}:
        run_scale_benchmark(
            shots=arguments.shots,
            steps=arguments.steps,
            output_dir=output_dir,
            number_of_initializations=arguments.num_init_points,
            simulator=arguments.simulator,
        )
    elif arguments.command == "all":
        for shots in DEFAULT_FIXED_SHOTS:
            run_fixed_tilt_experiment(
                shots=shots,
                steps=arguments.steps,
                gamma_values=DEFAULT_FIXED_GAMMAS,
                output_dir=output_dir,
                number_of_initializations=arguments.num_init_points,
                tail_window=arguments.tail_window,
                workers=arguments.workers,
                simulator=arguments.simulator,
            )
        run_ascending_tilt_experiment(
            shots=5000,
            steps=arguments.steps,
            average_gamma_values=DEFAULT_ASCENDING_AVERAGE_GAMMAS,
            output_dir=output_dir,
            number_of_initializations=arguments.num_init_points,
            tail_window=arguments.tail_window,
            workers=arguments.workers,
            simulator=arguments.simulator,
        )
        run_scale_benchmark(
            shots=5000,
            steps=arguments.steps,
            output_dir=output_dir,
            number_of_initializations=arguments.num_init_points,
            simulator=arguments.simulator,
        )
    else:
        raise ValueError(f"Unsupported command: {arguments.command}")


if __name__ == "__main__":
    main()
