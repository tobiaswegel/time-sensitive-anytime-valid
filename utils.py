"""Small helpers for rejection times and plot formatting."""

import math
from pathlib import Path

import numpy as np
from matplotlib.colors import to_rgb


def tau_to_pmf_cdf(tau: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Convert first-hit times to probabilities for times 1 through the horizon."""
    pmf = np.bincount(tau, minlength=horizon + 2)[1 : horizon + 1] / tau.size
    return pmf, np.cumsum(pmf)


def discounted_value_from_tau(tau: np.ndarray, time_scale: float, horizon: int) -> float:
    """Average exponential rewards, assigning zero to paths that never hit."""
    hit = tau <= horizon
    return float(np.mean(np.where(hit, np.exp(-tau / float(time_scale)), 0.0)))


def make_time_bins(plot_horizon: int, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Split observation times into bins with nearly equal widths."""
    bin_width = math.ceil(plot_horizon / n_bins)
    lefts = np.arange(1, plot_horizon + 1, bin_width, dtype=int)
    rights = np.minimum(lefts + bin_width, plot_horizon + 1)
    return lefts, rights


def aggregate_pmf_on_bins(
    pmf: np.ndarray,
    lefts: np.ndarray,
    rights: np.ndarray,
) -> np.ndarray:
    """Add rejection probabilities within each time bin."""
    return np.array(
        [pmf[left - 1 : right - 1].sum() for left, right in zip(lefts, rights)],
        dtype=float,
    )


def fade_color(color: str, blend: float = 0.50) -> tuple[float, float, float]:
    """Blend a color with white; blend=1 gives white."""
    rgb = np.array(to_rgb(color))
    return tuple((1.0 - blend) * rgb + blend * np.ones(3))


def tick_label_decimal(value: float) -> str:
    """Format a decimal tick without unnecessary trailing zeros."""
    if abs(value) < 1e-12:
        return r"$0$"
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return rf"${text}$"


def save_figure(figure, output_path, dpi=450):
    """Save a figure as PNG and return its path."""
    path = Path(output_path).with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    return path
