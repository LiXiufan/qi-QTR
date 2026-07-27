"""Validate the curated package's required files and scientific invariants."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent


def load(relative_path: str, required: set[str]) -> pd.DataFrame:
    """Load one required CSV and validate its schema."""
    path = ROOT / relative_path
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    frame = pd.read_csv(path)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    return frame


def require_rows(name: str, frame: pd.DataFrame, expected: int) -> None:
    """Require an exact row count."""
    if len(frame) != expected:
        raise ValueError(f"{name}: expected {expected} rows, found {len(frame)}")


def main() -> None:
    """Check retained row counts, pairing, baselines, and rendered figures."""
    summary_columns = {"gamma_plot", "mean_final_ratio", "sem_final_ratio"}
    figure_a_counts = {
        "fixed_gamma_shot_1000.csv": 14,
        "fixed_gamma_shot_5000.csv": 12,
        "fixed_gamma_shot_10000.csv": 14,
    }
    for filename, expected in figure_a_counts.items():
        frame = load(f"data_and_figures/{filename}", summary_columns)
        require_rows(filename, frame, expected)

    figure_b = load(
        "figure_b_latest/large_gamma.csv",
        summary_columns | {"dataset"},
    )
    require_rows("latest Figure (b)", figure_b, 36)
    if set(figure_b["dataset"]) != {"fixed", "ascending"}:
        raise ValueError("Figure (b) must contain fixed and ascending datasets.")
    for _, group in figure_b.groupby("dataset"):
        if group["gamma_plot"].nunique() != 18:
            raise ValueError("Each Figure (b) dataset must have 18 gamma values.")

    scale = load(
        "data_and_figures/maxcut_compare_avg_shot5000.csv",
        {"n", "p", "objective", "final_optimal_mass"},
    )
    require_rows("Figure (c) graph averages", scale, 135)

    paired = load(
        "paired_fixed_cvar_qtl_5000/paired_restart_results.csv",
        {
            "method",
            "control_index",
            "control_r",
            "graph_seed",
            "init_id",
            "device_seed",
            "initial_parameters",
            "final_mean_ratio",
        },
    )
    require_rows("paired restart results", paired, 350)
    pair_keys = ["control_index", "graph_seed", "init_id"]
    if not (paired.groupby(pair_keys).size() == 2).all():
        raise ValueError("The fixed-CVaR/fixed-QTL results are not fully paired.")
    for _, pair in paired.groupby(pair_keys, sort=False):
        if set(pair["method"]) != {"CVaR", "QTL"}:
            raise ValueError("A paired task does not contain both objectives.")
        if pair["device_seed"].nunique() != 1:
            raise ValueError("A paired task has different device seeds.")
        if pair["initial_parameters"].nunique() != 1:
            raise ValueError("A paired task has different initial parameters.")
    baseline = paired.loc[np.isclose(paired["control_r"], 0.0)]
    baseline_wide = baseline.pivot(
        index=["graph_seed", "init_id"],
        columns="method",
        values="final_mean_ratio",
    )
    if not np.allclose(
        baseline_wide["CVaR"],
        baseline_wide["QTL"],
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("The alpha=1/gamma=0 expectation controls differ.")

    history = load(
        "paired_fixed_cvar_qtl_5000/paired_iteration_history.csv",
        {"method", "control_index", "graph_seed", "init_id", "iteration"},
    )
    require_rows("paired iteration history", history, 35000)

    gradient = load(
        "parameter_shift_rule_comparison/parameter_shift_comparison.csv",
        {"record_type", "method", "iteration", "mean_final_ratio"},
    )
    require_rows("gradient comparison", gradient, 123)
    counts = (
        gradient.loc[gradient["record_type"] == "iteration"]
        .groupby("method")
        .size()
        .to_dict()
    )
    if counts != {"finite_difference": 61, "parameter_shift": 61}:
        raise ValueError(f"Unexpected gradient history counts: {counts}")

    figures = [
        "data_and_figures/fixed_plot_results.png",
        "data_and_figures/fixed_plot_results.pdf",
        "figure_b_latest/large_gamma_figure_b.png",
        "figure_b_latest/large_gamma_figure_b.pdf",
        "figure_b_latest/large_gamma_figure_b_log.png",
        "figure_b_latest/large_gamma_figure_b_log.pdf",
        "data_and_figures/maxcut_mean_optimal_mass_plot.png",
        "data_and_figures/maxcut_mean_optimal_mass_plot.pdf",
        "paired_fixed_cvar_qtl_5000/performance_matching_function.jpg",
        "paired_fixed_cvar_qtl_5000/performance_matching_function.pdf",
        "parameter_shift_rule_comparison/parameter_shift_comparison.png",
        "parameter_shift_rule_comparison/parameter_shift_comparison.pdf",
        "parameter_shift_rule_comparison/parameter_vector_error.png",
        "parameter_shift_rule_comparison/parameter_vector_error.pdf",
    ]
    for relative_path in figures:
        path = ROOT / relative_path
        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty figure: {path}")

    print("Curated package validation passed.")


if __name__ == "__main__":
    main()
