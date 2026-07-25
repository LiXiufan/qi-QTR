"""Shared, presentation-only utilities for the three figure scripts."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "qtl_matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


def apply_plot_style() -> None:
    """Apply a consistent publication-oriented Matplotlib style."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "stix",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 16,
            "axes.labelsize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 13,
            "axes.linewidth": 1.2,
            "lines.linewidth": 2.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "figure.autolayout": True,
        }
    )


def require_columns(
    frame: pd.DataFrame,
    columns: set[str],
    source: Path,
) -> None:
    """Raise a clear error when a plotting input has the wrong schema."""
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(
            f"{source} is missing required columns: {', '.join(missing)}"
        )


def load_curve_data(
    path: Path,
    *,
    value_column: str = "mean_final_ratio",
    sem_column: str = "sem_final_ratio",
    maximum_gamma: float = 4.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load, validate, sort, and bound one tilt summary curve."""
    path = Path(path)
    frame = pd.read_csv(path)
    require_columns(
        frame,
        {"gamma_plot", value_column, sem_column},
        path,
    )
    frame = (
        frame.dropna(subset=["gamma_plot", value_column, sem_column])
        .loc[lambda data: data["gamma_plot"] <= maximum_gamma]
        .sort_values("gamma_plot")
        .drop_duplicates("gamma_plot")
    )
    if len(frame) < 2:
        raise ValueError(f"{path} must contain at least two gamma values.")
    return (
        frame["gamma_plot"].to_numpy(dtype=float),
        frame[value_column].to_numpy(dtype=float),
        np.maximum(frame[sem_column].to_numpy(dtype=float), 0.0),
    )


def smooth_curve(
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    points: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate a smooth, shape-preserving curve through summary points."""
    dense_x = np.linspace(float(x_values.min()), float(x_values.max()), points)
    dense_y = PchipInterpolator(x_values, y_values)(dense_x)
    return dense_x, np.asarray(dense_y, dtype=float)


def padded_limits(
    values: list[np.ndarray],
    *,
    padding_fraction: float = 0.10,
) -> tuple[float, float]:
    """Calculate stable y limits with a small visual margin."""
    minimum = min(float(np.min(value)) for value in values)
    maximum = max(float(np.max(value)) for value in values)
    span = maximum - minimum
    padding = max(0.005, padding_fraction * span)
    return minimum - padding, maximum + padding


def save_figure(
    figure,
    output_png: Path,
    output_pdf: Path,
    *,
    dpi: int = 300,
) -> None:
    """Save a figure in raster and vector formats."""
    output_png = Path(output_png)
    output_pdf = Path(output_pdf)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_png, dpi=dpi, bbox_inches="tight")
    figure.savefig(output_pdf, bbox_inches="tight")
    plt.close(figure)
