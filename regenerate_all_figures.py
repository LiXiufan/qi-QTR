"""Regenerate every retained figure from the included result CSV files."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def run(script: str, *arguments: str) -> None:
    """Run one plotting entry point from the curated project root."""
    command = [sys.executable, str(SCRIPT_DIR / script), *arguments]
    print(f"Running {script}", flush=True)
    subprocess.run(command, cwd=SCRIPT_DIR, check=True)


def main() -> None:
    """Render Figures (a)-(c) and both retained comparison studies."""
    run("fixed_tilt.py")
    run("ascending_tilt.py")
    run(
        "ascending_tilt.py",
        "--log-x",
        "--output-png",
        str(SCRIPT_DIR / "figure_b_latest" / "large_gamma_figure_b_log.png"),
        "--output-pdf",
        str(SCRIPT_DIR / "figure_b_latest" / "large_gamma_figure_b_log.pdf"),
    )
    run("scale_benchmark.py")
    run("paired_fixed_cvar_qtl.py", "--plot-only")
    run(
        str(Path("parameter_shift_rule_comparison")
            / "parameter_shift_comparison.py"),
        "--plot-only",
    )
    print("All retained figures regenerated successfully.", flush=True)


if __name__ == "__main__":
    main()
