"""Figure 2: exact Bernoulli Doob and GRO rejection-time distributions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bernoulli import exact_constant_bernoulli_curve, exact_doob_curve
from plotting import plot_rejection_times
from utils import save_figure

# Configuration


P0 = 0.4
P1 = 0.6
ALPHA = 0.05

DEADLINES = [10, 20]
HORIZON = 30
PLOT_MAX_T = 30

DPI = 450


N_BINS_TOTAL = 6

OUT_DIR = Path(".")
OUT_FIG_PNG = OUT_DIR / "figure_2.png"
OUT_CSV = OUT_DIR / "figure_2.csv"
OUT_JSON = OUT_DIR / "figure_2.json"

# Plot style


def configure_fonts() -> None:
    """Set the fonts and sizes used in the paper plot."""
    plt.rcParams.update(
        {
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral"],
            "mathtext.fontset": "stix",
            "mathtext.rm": "STIXGeneral",
            "mathtext.it": "STIXGeneral:italic",
            "mathtext.bf": "STIXGeneral:bold",
            "font.size": 10.5,
            "axes.labelsize": 10.8,
            "legend.fontsize": 8.6,
            "xtick.labelsize": 9.2,
            "ytick.labelsize": 9.2,
            "axes.linewidth": 0.95,
            "lines.linewidth": 2.0,
            "savefig.bbox": "tight",
        }
    )


COL_DOOB_10 = "tab:blue"
COL_DOOB_20 = "tab:orange"
COL_GRO = "tab:green"

CDF_LS_DOOB_10 = "-"
CDF_LS_DOOB_20 = "--"
CDF_LS_GRO = ":"


# Exact deterministic NP event with lexicographic tie-break


# Exact Doob martingale stopping time for Gamma*


# Exact GRO curve


def exact_gro_curve(p0: float, p1: float, alpha: float, horizon: int) -> dict[str, Any]:
    """Compute exact rejection probabilities for the likelihood-ratio strategy."""
    curve = exact_constant_bernoulli_curve(p0, p1, alpha, horizon, p1, "GRO", "gro")
    curve.pop("p_action")
    curve["name"] = "GRO"
    for key in ("pmf", "cdf"):
        curve[key] = np.r_[0.0, curve[key]]
    return curve


# Experiment


def run_experiment() -> dict[str, Any]:
    """Compute the strategies, rejection curves, and summary table."""
    doob_curves = [
        exact_doob_curve(T=T, p0=P0, p1=P1, alpha=ALPHA, horizon=HORIZON) for T in DEADLINES
    ]

    gro_curve = exact_gro_curve(
        p0=P0,
        p1=P1,
        alpha=ALPHA,
        horizon=HORIZON,
    )

    curves = [doob_curves[0], doob_curves[1], gro_curve]

    rows = []
    for curve in curves:
        row = {
            "curve": curve["label"],
            "kind": curve["kind"],
            "hit_by_10": float(curve["cdf"][10]),
            "hit_by_20": float(curve["cdf"][20]),
            "hit_by_30": float(curve["cdf"][30]),
            "terminal_power": curve["power"],
        }

        if "T" in curve:
            row["T"] = curve["T"]
            row["k"] = curve["k"]
            row["m"] = curve["m"]
            row["r"] = curve["r"]
            row["level"] = curve["level"]
            row["c_star"] = curve["c_star"]
            row["conditional_prob_threshold"] = curve.get(
                "conditional_prob_threshold",
                None,
            )
        else:
            row["T"] = None
            row["k"] = None
            row["m"] = None
            row["r"] = None
            row["level"] = None
            row["c_star"] = None
            row["conditional_prob_threshold"] = None

        rows.append(row)

    return {
        "curves": curves,
        "summary": pd.DataFrame(rows),
    }


# Plotting helpers


def curve_color(curve: dict[str, Any]) -> str:
    """Return the plot color assigned to a strategy."""
    if curve["kind"] == "doob_10":
        return COL_DOOB_10
    if curve["kind"] == "doob_20":
        return COL_DOOB_20
    if curve["kind"] == "gro":
        return COL_GRO
    raise ValueError(f"Unknown curve kind: {curve['kind']}")


def curve_cdf_linestyle(curve: dict[str, Any]) -> str:
    """Return the line style assigned to a strategy."""
    if curve["kind"] == "doob_10":
        return CDF_LS_DOOB_10
    if curve["kind"] == "doob_20":
        return CDF_LS_DOOB_20
    if curve["kind"] == "gro":
        return CDF_LS_GRO
    raise ValueError(f"Unknown curve kind: {curve['kind']}")


def ordered_curves_for_plot(curves: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order strategies consistently for the legend and plots."""
    order = ["doob_10", "doob_20", "gro"]
    return [next(curve for curve in curves if curve["kind"] == kind) for kind in order]


def plot_stacked_pmf_cdf(curves, plot_max_t, n_bins, ymax_bin=None):
    """Draw this experiment’s rejection-time distributions in the shared layout."""
    curves = ordered_curves_for_plot(curves)
    colors = [curve_color(c) for c in curves]
    linestyles = [curve_cdf_linestyle(c) for c in curves]
    curves = [{**c, "pmf": c["pmf"][1:], "cdf": c["cdf"][1:]} for c in curves]
    return plot_rejection_times(
        curves,
        colors,
        linestyles,
        min(plot_max_t, HORIZON),
        n_bins,
        style="deadline",
        ymax_bin=ymax_bin,
    )


# Saving


def save_results(result: dict[str, Any]) -> None:
    """Save the parameters and numerical results."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result["summary"].to_csv(OUT_CSV, index=False)

    serializable_curves = []
    for curve in result["curves"]:
        serializable_curves.append(
            {
                key: (value.tolist() if isinstance(value, np.ndarray) else value)
                for key, value in curve.items()
            }
        )

    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "parameters": {
                    "p0": P0,
                    "p1": P1,
                    "alpha": ALPHA,
                    "deadlines": DEADLINES,
                    "horizon": HORIZON,
                    "plot_max_t": PLOT_MAX_T,
                    "deterministic_tie_break": "lexicographic, with 1 > 0",
                    "n_bins_total": N_BINS_TOTAL,
                    "cdf_line_styles": {
                        "Doob T=10": CDF_LS_DOOB_10,
                        "Doob T=20": CDF_LS_DOOB_20,
                        "GRO": CDF_LS_GRO,
                    },
                },
                "summary": result["summary"].to_dict(orient="records"),
                "curves": serializable_curves,
            },
            f,
            indent=2,
        )


# Main


def main() -> None:
    """Run the experiment and save its outputs."""
    configure_fonts()
    plt.close("all")

    result = run_experiment()
    print(result["summary"].to_string(index=False))

    fig = plot_stacked_pmf_cdf(
        curves=result["curves"],
        plot_max_t=PLOT_MAX_T,
        n_bins=N_BINS_TOTAL,
        ymax_bin=None,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_figure(fig, OUT_FIG_PNG, dpi=DPI)
    save_results(result)

    plt.show()

    print("\nSaved:")
    print(f"  {OUT_FIG_PNG}")
    print(f"  {OUT_CSV}")
    print(f"  {OUT_JSON}")


if __name__ == "__main__":
    main()
