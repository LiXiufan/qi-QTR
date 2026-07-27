"""Run every production experiment concurrently, then regenerate all figures."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from experiments import FIGURE_A_FIXED_GAMMAS_BY_SHOTS, FIGURE_B_GAMMAS


SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass
class RunningTask:
    """One experiment subprocess and its associated log files."""

    name: str
    command: list[str]
    process: subprocess.Popen
    stdout_handle: object
    stderr_handle: object


def experiment_commands(
    output_dir: Path,
    *,
    steps: int,
    number_of_initializations: int,
) -> list[tuple[str, list[str]]]:
    """Build the six independent production experiment commands."""
    figure_b_dir = output_dir / "figure_b_latest"
    common = [
        "--steps",
        str(steps),
        "--num-init-points",
        str(number_of_initializations),
    ]
    return [
        (
            f"fixed_{shots}",
            [
                sys.executable,
                "-u",
                str(SCRIPT_DIR / "run_experiments.py"),
                "fixed",
                "--shots",
                str(shots),
                "--gammas",
                ",".join(
                    f"{gamma:g}"
                    for gamma in FIGURE_A_FIXED_GAMMAS_BY_SHOTS[shots]
                ),
                *common,
                "--output-dir",
                str(output_dir),
            ],
        )
        for shots in [1000, 5000, 10000]
    ] + [
        (
            "figure_b_fixed_5000",
            [
                sys.executable,
                "-u",
                str(SCRIPT_DIR / "run_experiments.py"),
                "fixed",
                "--shots",
                "5000",
                "--gammas",
                ",".join(f"{gamma:g}" for gamma in FIGURE_B_GAMMAS),
                *common,
                "--output-dir",
                str(figure_b_dir),
            ],
        ),
        (
            "figure_b_ascending_5000",
            [
                sys.executable,
                "-u",
                str(SCRIPT_DIR / "run_experiments.py"),
                "ascending",
                "--shots",
                "5000",
                *common,
                "--output-dir",
                str(figure_b_dir),
            ],
        ),
        (
            "scale_5000",
            [
                sys.executable,
                "-u",
                str(SCRIPT_DIR / "run_experiments.py"),
                "scale",
                "--shots",
                "5000",
                *common,
                "--output-dir",
                str(output_dir),
            ],
        ),
    ]


def launch_experiments(
    commands: list[tuple[str, list[str]]],
    log_dir: Path,
) -> list[RunningTask]:
    """Launch all independent experiment families."""
    running_tasks = []
    for name, command in commands:
        stdout_handle = (log_dir / f"{name}.stdout.log").open(
            "w",
            encoding="utf-8",
        )
        stderr_handle = (log_dir / f"{name}.stderr.log").open(
            "w",
            encoding="utf-8",
        )
        process = subprocess.Popen(
            command,
            cwd=SCRIPT_DIR,
            stdout=stdout_handle,
            stderr=stderr_handle,
        )
        running_tasks.append(
            RunningTask(
                name=name,
                command=command,
                process=process,
                stdout_handle=stdout_handle,
                stderr_handle=stderr_handle,
            )
        )
        print(f"Started {name} with PID {process.pid}", flush=True)
    return running_tasks


def wait_for_experiments(
    running_tasks: list[RunningTask],
) -> dict[str, int]:
    """Wait for every experiment and return its process exit code."""
    return_codes = {}
    unfinished = {task.name: task for task in running_tasks}
    try:
        while unfinished:
            for name, task in list(unfinished.items()):
                return_code = task.process.poll()
                if return_code is None:
                    continue
                return_codes[name] = return_code
                task.stdout_handle.close()
                task.stderr_handle.close()
                del unfinished[name]
                print(
                    f"Finished {name} with exit code {return_code}",
                    flush=True,
                )
            if unfinished:
                time.sleep(5)
    finally:
        for task in unfinished.values():
            task.stdout_handle.close()
            task.stderr_handle.close()
    return return_codes


def regenerate_figures(output_dir: Path) -> None:
    """Run all plotting scripts against the newly generated CSVs."""
    figure_b_dir = output_dir / "figure_b_latest"
    reference_shape = output_dir / "curve_fitting_poly_tail_summary.csv"
    if not reference_shape.exists():
        reference_shape = (
            SCRIPT_DIR
            / "data_and_figures"
            / "curve_fitting_poly_tail_summary.csv"
        )
    plotting_commands = [
        [
            sys.executable,
            str(SCRIPT_DIR / "fixed_tilt.py"),
            "--shot-1000",
            str(output_dir / "fixed_gamma_shot_1000.csv"),
            "--shot-5000",
            str(output_dir / "fixed_gamma_shot_5000.csv"),
            "--shot-10000",
            str(output_dir / "fixed_gamma_shot_10000.csv"),
            "--reference-shape-csv",
            str(reference_shape),
            "--fit-summary-csv",
            str(output_dir / "fixed_plot_fit_summary.csv"),
            "--output-png",
            str(output_dir / "fixed_plot_results.png"),
            "--output-pdf",
            str(output_dir / "fixed_plot_results.pdf"),
        ],
        [
            sys.executable,
            str(SCRIPT_DIR / "ascending_tilt.py"),
            "--fixed-csv",
            str(figure_b_dir / "fixed_gamma_shot_5000.csv"),
            "--ascending-csv",
            str(
                figure_b_dir
                / "schedule_gamma_restart_group(shots5000)all.csv"
            ),
            "--fit-summary-csv",
            str(figure_b_dir / "large_gamma_fit_summary.csv"),
            "--output-png",
            str(figure_b_dir / "large_gamma_figure_b.png"),
            "--output-pdf",
            str(figure_b_dir / "large_gamma_figure_b.pdf"),
        ],
        [
            sys.executable,
            str(SCRIPT_DIR / "scale_benchmark.py"),
            "--data-csv",
            str(output_dir / "maxcut_compare_avg_shot5000.csv"),
            "--output-png",
            str(output_dir / "maxcut_mean_optimal_mass_plot.png"),
            "--output-pdf",
            str(output_dir / "maxcut_mean_optimal_mass_plot.pdf"),
        ],
    ]
    for command in plotting_commands:
        print(f"Running {Path(command[1]).name}", flush=True)
        subprocess.run(command, cwd=SCRIPT_DIR, check=True)


def build_parser() -> argparse.ArgumentParser:
    """Build the full-pipeline command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Run all production QTL experiments concurrently and then plot."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "data_and_figures",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=SCRIPT_DIR / "full_run_logs",
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--num-init-points", type=int, default=5)
    return parser


def main() -> None:
    """Execute the full data-and-figure production pipeline."""
    arguments = build_parser().parse_args()
    output_dir = arguments.output_dir.resolve()
    log_dir = arguments.log_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc)
    commands = experiment_commands(
        output_dir,
        steps=arguments.steps,
        number_of_initializations=arguments.num_init_points,
    )
    tasks = launch_experiments(commands, log_dir)
    return_codes = wait_for_experiments(tasks)
    failed = {
        name: code
        for name, code in return_codes.items()
        if code != 0
    }

    status = {
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "return_codes": return_codes,
        "figures_regenerated": False,
    }
    if failed:
        (log_dir / "pipeline_status.json").write_text(
            json.dumps(status, indent=2),
            encoding="utf-8",
        )
        raise RuntimeError(f"Experiment failures: {failed}")

    regenerate_figures(output_dir)
    status["figures_regenerated"] = True
    (log_dir / "pipeline_status.json").write_text(
        json.dumps(status, indent=2),
        encoding="utf-8",
    )
    print("Full data and figure pipeline completed.", flush=True)


if __name__ == "__main__":
    main()
