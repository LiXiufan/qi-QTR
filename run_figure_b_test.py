"""Regenerate Figure (b) into isolated test data and figure artifacts."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from ascending_tilt import plot_ascending_tilt
from experiments import (
    DEFAULT_ASCENDING_FINAL_GAMMAS,
    DEFAULT_FIXED_GAMMAS,
    SHARED_TILT_OPTIMIZER,
)


SCRIPT_DIR = Path(__file__).resolve().parent


def _launch_sweep(
    command: str,
    *,
    work_dir: Path,
    shots: int,
    steps: int,
    number_of_initializations: int,
    workers: int,
    simulator: str,
) -> tuple[subprocess.Popen, object, object]:
    stdout_handle = (work_dir / f"{command}.stdout.log").open(
        "w",
        encoding="utf-8",
    )
    stderr_handle = (work_dir / f"{command}.stderr.log").open(
        "w",
        encoding="utf-8",
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
            "--output-dir",
            str(work_dir),
        ],
        cwd=SCRIPT_DIR,
        stdout=stdout_handle,
        stderr=stderr_handle,
    )
    return process, stdout_handle, stderr_handle


def _validate_summary(path: Path, expected_rows: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing expected summary: {path}")
    frame = pd.read_csv(path)
    if len(frame) != expected_rows:
        raise ValueError(
            f"{path} should contain {expected_rows} rows; found {len(frame)}."
        )
    expected_gamma_count = frame["gamma_plot"].nunique()
    if expected_gamma_count != expected_rows:
        raise ValueError(f"{path} contains duplicate gamma values.")
    return frame.sort_values("gamma_plot").reset_index(drop=True)


def finalize_test_artifacts(
    *,
    work_dir: Path,
    output_csv: Path,
    output_png: Path,
    output_pdf: Path,
    fit_summary_csv: Path,
    shots: int,
    steps: int,
    number_of_initializations: int,
    simulator: str,
) -> None:
    """Combine completed summaries and render the isolated Figure (b)."""
    work_dir = Path(work_dir).resolve()
    fixed_path = work_dir / f"fixed_gamma_shot_{shots}.csv"
    ascending_path = (
        work_dir
        / f"schedule_gamma_restart_group(shots{shots})all.csv"
    )
    fixed = _validate_summary(
        fixed_path,
        expected_rows=len(DEFAULT_FIXED_GAMMAS),
    )
    ascending = _validate_summary(
        ascending_path,
        expected_rows=len(DEFAULT_ASCENDING_FINAL_GAMMAS),
    )
    fixed.insert(0, "dataset", "fixed")
    ascending.insert(0, "dataset", "ascending")
    combined = pd.concat([fixed, ascending], ignore_index=True)
    combined["shots"] = shots
    combined["steps"] = steps
    combined["number_of_initializations"] = number_of_initializations
    combined["simulator"] = simulator
    for name, value in SHARED_TILT_OPTIMIZER.items():
        combined[f"optimizer_{name}"] = value

    output_csv = Path(output_csv).resolve()
    output_png = Path(output_png).resolve()
    output_pdf = Path(output_pdf).resolve()
    fit_summary_csv = Path(fit_summary_csv).resolve()
    for path in (
        output_csv,
        output_png,
        output_pdf,
        fit_summary_csv,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_csv, index=False)

    plot_ascending_tilt(
        fixed_path,
        ascending_path,
        output_png,
        output_pdf,
        fit_summary_csv,
    )
    print(f"Saved combined test data: {output_csv}", flush=True)
    print(f"Saved test figure: {output_png}", flush=True)
    print(f"Saved vector test figure: {output_pdf}", flush=True)
    print(f"Saved test fit summary: {fit_summary_csv}", flush=True)


def regenerate_test_figure(
    *,
    work_dir: Path,
    output_csv: Path,
    output_png: Path,
    output_pdf: Path,
    fit_summary_csv: Path,
    shots: int,
    steps: int,
    number_of_initializations: int,
    workers_per_sweep: int,
    simulator: str,
) -> None:
    """Run both sweeps, combine their summaries, and render Figure (b)."""
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    running = [
        _launch_sweep(
            command,
            work_dir=work_dir,
            shots=shots,
            steps=steps,
            number_of_initializations=number_of_initializations,
            workers=workers_per_sweep,
            simulator=simulator,
        )
        for command in ("fixed", "ascending")
    ]

    failures: list[str] = []
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

    finalize_test_artifacts(
        work_dir=work_dir,
        output_csv=output_csv,
        output_png=output_png,
        output_pdf=output_pdf,
        fit_summary_csv=fit_summary_csv,
        shots=shots,
        steps=steps,
        number_of_initializations=number_of_initializations,
        simulator=simulator,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=SCRIPT_DIR / "figure_b_test_run",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=SCRIPT_DIR / "test.csv",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=SCRIPT_DIR / "test.png",
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=SCRIPT_DIR / "test.pdf",
    )
    parser.add_argument(
        "--fit-summary-csv",
        type=Path,
        default=SCRIPT_DIR / "test_fit_summary.csv",
    )
    parser.add_argument("--shots", type=int, default=5000)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--num-init-points", type=int, default=5)
    parser.add_argument("--workers-per-sweep", type=int, default=2)
    parser.add_argument(
        "--simulator",
        choices=["default.qubit", "lightning.qubit"],
        default="lightning.qubit",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help=(
            "Use the explicit screening preset: 1,000 shots, 40 steps, "
            "and three initializations. This changes the workload."
        ),
    )
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    if arguments.fast:
        arguments.shots = 1000
        arguments.steps = 40
        arguments.num_init_points = 3
    print(
        "Figure (b) test configuration: "
        f"shots={arguments.shots}, steps={arguments.steps}, "
        f"initializations={arguments.num_init_points}, "
        f"workers_per_sweep={arguments.workers_per_sweep}, "
        f"simulator={arguments.simulator}",
        flush=True,
    )
    regenerate_test_figure(
        work_dir=arguments.work_dir,
        output_csv=arguments.output_csv,
        output_png=arguments.output_png,
        output_pdf=arguments.output_pdf,
        fit_summary_csv=arguments.fit_summary_csv,
        shots=arguments.shots,
        steps=arguments.steps,
        number_of_initializations=arguments.num_init_points,
        workers_per_sweep=arguments.workers_per_sweep,
        simulator=arguments.simulator,
    )


if __name__ == "__main__":
    main()
