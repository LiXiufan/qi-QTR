"""Run selected Figure (b) gamma points and merge them into existing data."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from ascending_tilt import (
    plot_ascending_tilt,
    plot_ascending_tilt_log,
)
from experiments import SHARED_TILT_OPTIMIZER
from run_experiments import parse_float_list


SCRIPT_DIR = Path(__file__).resolve().parent


def launch_sweep(
    command: str,
    *,
    gamma_values: list[float],
    work_dir: Path,
    shots: int,
    steps: int,
    number_of_initializations: int,
    workers: int,
    simulator: str,
) -> tuple[subprocess.Popen, object, object]:
    """Launch one selected-gamma sweep with dedicated log files."""
    stdout_handle = (work_dir / f"{command}.stdout.log").open(
        "w",
        encoding="utf-8",
    )
    stderr_handle = (work_dir / f"{command}.stderr.log").open(
        "w",
        encoding="utf-8",
    )
    gamma_option = (
        "--gammas"
        if command == "fixed"
        else "--average-gammas"
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-u",
            str(SCRIPT_DIR / "run_experiments.py"),
            command,
            "--shots",
            str(shots),
            "--steps",
            str(steps),
            "--num-init-points",
            str(number_of_initializations),
            "--workers",
            str(workers),
            "--simulator",
            simulator,
            gamma_option,
            ",".join(f"{value:g}" for value in gamma_values),
            "--output-dir",
            str(work_dir),
        ],
        cwd=SCRIPT_DIR,
        stdout=stdout_handle,
        stderr=stderr_handle,
    )
    return process, stdout_handle, stderr_handle


def load_selected_summary(
    path: Path,
    expected_gammas: list[float],
) -> pd.DataFrame:
    """Load and validate a selected-gamma summary."""
    if not path.exists():
        raise FileNotFoundError(f"Missing selected-gamma summary: {path}")
    frame = pd.read_csv(path)
    actual = sorted(frame["gamma_plot"].astype(float).tolist())
    expected = sorted(float(value) for value in expected_gammas)
    if actual != expected:
        raise ValueError(
            f"{path} gamma values {actual} do not match {expected}."
        )
    return frame


def merge_and_plot(
    *,
    gamma_values: list[float],
    work_dir: Path,
    combined_csv: Path,
    output_png: Path,
    output_pdf: Path,
    output_log_png: Path,
    output_log_pdf: Path,
    fit_summary_csv: Path,
    shots: int,
    steps: int,
    number_of_initializations: int,
    simulator: str,
) -> None:
    """Merge selected summaries and regenerate both Figure (b) variants."""
    fixed = load_selected_summary(
        work_dir / f"fixed_gamma_shot_{shots}.csv",
        gamma_values,
    )
    ascending = load_selected_summary(
        work_dir
        / f"schedule_gamma_restart_group(shots{shots})all.csv",
        gamma_values,
    )
    fixed.insert(0, "dataset", "fixed")
    ascending.insert(0, "dataset", "ascending")
    selected = pd.concat([fixed, ascending], ignore_index=True)
    selected["shots"] = shots
    selected["steps"] = steps
    selected["number_of_initializations"] = number_of_initializations
    selected["simulator"] = simulator
    for name, value in SHARED_TILT_OPTIMIZER.items():
        selected[f"optimizer_{name}"] = value

    combined_csv = Path(combined_csv).resolve()
    if combined_csv.exists():
        existing = pd.read_csv(combined_csv)
        combined = pd.concat(
            [existing, selected],
            ignore_index=True,
            sort=False,
        )
    else:
        combined = selected
    combined = (
        combined.drop_duplicates(
            subset=["dataset", "gamma_plot"],
            keep="last",
        )
        .assign(
            _dataset_order=lambda frame: frame["dataset"].map(
                {"fixed": 0, "ascending": 1}
            )
        )
        .sort_values(["_dataset_order", "gamma_plot"])
        .drop(columns="_dataset_order")
        .reset_index(drop=True)
    )
    combined_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(combined_csv, index=False)

    merged_fixed = work_dir / "merged_fixed_summary.csv"
    merged_ascending = work_dir / "merged_ascending_summary.csv"
    combined.loc[combined["dataset"] == "fixed"].drop(
        columns="dataset"
    ).to_csv(merged_fixed, index=False)
    combined.loc[combined["dataset"] == "ascending"].drop(
        columns="dataset"
    ).to_csv(merged_ascending, index=False)

    plot_ascending_tilt(
        merged_fixed,
        merged_ascending,
        output_png,
        output_pdf,
        fit_summary_csv,
    )
    plot_ascending_tilt_log(
        merged_fixed,
        merged_ascending,
        output_log_png,
        output_log_pdf,
    )
    print(f"Saved expanded Figure (b) data: {combined_csv}", flush=True)
    print(f"Saved linear Figure (b): {output_png}", flush=True)
    print(f"Saved log-x Figure (b): {output_log_png}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    """Build the incremental Figure (b) command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gammas",
        type=parse_float_list,
        required=True,
        help="Comma-separated gamma values to execute and merge.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=SCRIPT_DIR / "figure_b_incremental_run",
    )
    parser.add_argument(
        "--combined-csv",
        type=Path,
        default=SCRIPT_DIR / "large_gamma.csv",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=SCRIPT_DIR / "large_gamma_figure_b.png",
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=SCRIPT_DIR / "large_gamma_figure_b.pdf",
    )
    parser.add_argument(
        "--output-log-png",
        type=Path,
        default=SCRIPT_DIR / "large_gamma_figure_b_log.png",
    )
    parser.add_argument(
        "--output-log-pdf",
        type=Path,
        default=SCRIPT_DIR / "large_gamma_figure_b_log.pdf",
    )
    parser.add_argument(
        "--fit-summary-csv",
        type=Path,
        default=SCRIPT_DIR / "large_gamma_fit_summary.csv",
    )
    parser.add_argument("--shots", type=int, default=5000)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--num-init-points", type=int, default=5)
    parser.add_argument("--workers-per-sweep", type=int, default=1)
    parser.add_argument(
        "--simulator",
        choices=["default.qubit", "lightning.qubit"],
        default="lightning.qubit",
    )
    return parser


def main() -> None:
    """Run both selected-gamma sweeps, merge, and regenerate figures."""
    arguments = build_parser().parse_args()
    gamma_values = list(dict.fromkeys(arguments.gammas))
    if not gamma_values:
        raise ValueError("At least one gamma value is required.")
    work_dir = arguments.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    print(
        "Incremental Figure (b) configuration: "
        f"gammas={gamma_values}, shots={arguments.shots}, "
        f"steps={arguments.steps}, "
        f"initializations={arguments.num_init_points}, "
        f"workers_per_sweep={arguments.workers_per_sweep}, "
        f"simulator={arguments.simulator}",
        flush=True,
    )
    running = [
        launch_sweep(
            command,
            gamma_values=gamma_values,
            work_dir=work_dir,
            shots=arguments.shots,
            steps=arguments.steps,
            number_of_initializations=arguments.num_init_points,
            workers=arguments.workers_per_sweep,
            simulator=arguments.simulator,
        )
        for command in ("fixed", "ascending")
    ]
    failures = []
    for (process, stdout_handle, stderr_handle), command in zip(
        running,
        ("fixed", "ascending"),
    ):
        return_code = process.wait()
        stdout_handle.close()
        stderr_handle.close()
        if return_code != 0:
            failures.append(f"{command} exited with code {return_code}")
    if failures:
        raise RuntimeError("; ".join(failures))

    merge_and_plot(
        gamma_values=gamma_values,
        work_dir=work_dir,
        combined_csv=arguments.combined_csv,
        output_png=arguments.output_png,
        output_pdf=arguments.output_pdf,
        output_log_png=arguments.output_log_png,
        output_log_pdf=arguments.output_log_pdf,
        fit_summary_csv=arguments.fit_summary_csv,
        shots=arguments.shots,
        steps=arguments.steps,
        number_of_initializations=arguments.num_init_points,
        simulator=arguments.simulator,
    )


if __name__ == "__main__":
    main()
