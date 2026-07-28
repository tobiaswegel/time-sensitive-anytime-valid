"""
Bernoulli hard-deadline Doob CDFs vs GRO, small-readable stacked figure.

Model:
    pi_0 = Bernoulli(0.4)
    pi_1 = Bernoulli(0.6)

Curves:
    Doob T=10:
        exact deterministic Neyman--Pearson event with lexicographic tie-break,
        stopped using the exact canonical Doob martingale threshold

    Doob T=20:
        exact deterministic Neyman--Pearson event with lexicographic tie-break,
        stopped using the exact canonical Doob martingale threshold

    GRO:
        constant likelihood-ratio e-process, computed exactly by alive-state
        propagation

Output:
    - small stacked PMF-bin / CDF figure as PNG
    - summary CSV
    - curves JSON
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from math import comb
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D


# ============================================================
# Configuration
# ============================================================

P0 = 0.4
P1 = 0.6
ALPHA = 0.05

DEADLINES = [10, 20]
HORIZON = 30
PLOT_MAX_T = 30

DPI = 450

# Small figure, but with readable text after paper scaling.
FIG_WIDTH = 4.15
FIG_HEIGHT = 3.25

# Fewer bins so the top panel remains readable when small.
N_BINS_TOTAL = 6
XTICK_EVERY = 1

OUT_DIR = Path(".")
OUT_FIG_PNG = OUT_DIR / "bernoulli_hard_deadline_vs_gro_stacked_small.png"
OUT_CSV = OUT_DIR / "bernoulli_hard_deadline_vs_gro_summary.csv"
OUT_JSON = OUT_DIR / "bernoulli_hard_deadline_vs_gro_curves.json"


# ============================================================
# Plot style
# ============================================================

def configure_fonts() -> None:
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

BAR_FACE_BLEND = 0.80
BAR_EDGE_BLEND = 0.45
BAR_EDGE_LW = 0.75
BAR_TOP_LW = 1.05

BIN_EDGE_LW = 0.45
BIN_SHADE_COLOR = "0.955"
BIN_EDGE_COLOR = "0.82"

OUTER_MARGIN_FRAC = 0.06
INNER_GAP_FRAC = 0.035

CDF_LW = 1.55

LEGEND_FONTSIZE = 8.4
TICK_FONTSIZE = 9.0
AXIS_LABEL_FONTSIZE = 10.6


# ============================================================
# Basic Bernoulli helpers
# ============================================================

def binom_pmf(n: int, p: float, k: int) -> float:
    if k < 0 or k > n:
        return 0.0
    return comb(n, k) * (p ** k) * ((1.0 - p) ** (n - k))


def binom_tail(n: int, p: float, k: int) -> float:
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return sum(binom_pmf(n, p, j) for j in range(k, n + 1))


# ============================================================
# Exact deterministic NP event with lexicographic tie-break
# ============================================================

def choose_np_boundary(T: int, p0: float, alpha: float) -> dict[str, Any]:
    """
    Find k such that P0(S_T >= k) <= alpha < P0(S_T >= k-1).

    The deterministic NP event Gamma* contains all strings with S_T >= k,
    plus r lexicographically first strings on the boundary S_T = k-1.
    """
    k = next(k for k in range(T + 1) if binom_tail(T, p0, k) <= alpha)
    m = k - 1

    tail_above = binom_tail(T, p0, k)
    atom0 = (p0 ** m) * ((1.0 - p0) ** (T - m))
    r = int(math.floor((alpha - tail_above) / atom0 + 1e-15))
    r = max(0, min(r, comb(T, m)))

    level = tail_above + r * atom0

    return {
        "T": T,
        "k": k,
        "m": m,
        "r": r,
        "tail_above": tail_above,
        "atom0": atom0,
        "level": level,
    }


def unrank_fixed_weight_desc(T: int, m: int, r: int) -> str:
    """
    r-th string, 1-indexed, among length-T binary strings with exactly m ones,
    ordered lexicographically with 1 > 0.
    """
    if not (1 <= r <= comb(T, m)):
        raise ValueError(f"r={r} must lie in [1, C({T},{m})].")

    bits = []
    ones_left = m

    for pos in range(T):
        rem = T - pos
        count_if_one = comb(rem - 1, ones_left - 1) if ones_left > 0 else 0

        if r <= count_if_one:
            bits.append("1")
            ones_left -= 1
        else:
            bits.append("0")
            r -= count_if_one

    return "".join(bits)


def update_cmp(cmp_state: int, bit: int, cbit: int) -> int:
    """
    Comparison of current prefix with c_star prefix under lex order 1 > 0.

      -1: prefix < c_star prefix
       0: prefix = c_star prefix
      +1: prefix > c_star prefix
    """
    if cmp_state != 0:
        return cmp_state
    if bit > cbit:
        return 1
    if bit < cbit:
        return -1
    return 0


# ============================================================
# Exact Doob martingale stopping time for Gamma*
# ============================================================

_CMP_TO_INDEX = {-1: 0, 0: 1, 1: 2}
_INDEX_TO_CMP = {0: -1, 1: 0, 2: 1}


def build_conditional_np_prob_table(
    T: int,
    p0: float,
    k: int,
    m: int,
    c_star: str | None,
) -> np.ndarray:
    """
    Backward dynamic program for

        q[t, s, cmp] = P0(Gamma* | prefix state after t observations),

    where:
        s   = number of ones in the prefix,
        cmp = lexicographic comparison of the prefix with c_star's prefix.

    The canonical Doob martingale is

        U_t^* = P0(Gamma* | X^t) / P0(Gamma*).

    Therefore the exact stopping rule is

        stop when q[t, s, cmp] >= P0(Gamma*) / alpha.

    This is the only substantive change relative to the previous version,
    which stopped only when the prefix forced membership in Gamma*.
    """
    q = np.zeros((T + 1, T + 1, 3), dtype=float)

    # Terminal membership in Gamma*.
    for s in range(T + 1):
        for idx in range(3):
            cmp_state = _INDEX_TO_CMP[idx]

            if s >= k:
                q[T, s, idx] = 1.0
            elif c_star is not None and s == m and cmp_state >= 0:
                # Boundary strings with S_T = m are included iff they are
                # lexicographically >= c_star under the ordering 1 > 0.
                q[T, s, idx] = 1.0
            else:
                q[T, s, idx] = 0.0

    # Backward induction under pi_0.
    for t in range(T - 1, -1, -1):
        cbit = int(c_star[t]) if c_star is not None else 0

        for s in range(t + 1):
            for idx in range(3):
                cmp_state = _INDEX_TO_CMP[idx]

                if c_star is None:
                    cmp0 = 0
                    cmp1 = 0
                else:
                    cmp0 = update_cmp(cmp_state, 0, cbit)
                    cmp1 = update_cmp(cmp_state, 1, cbit)

                q0 = q[t + 1, s, _CMP_TO_INDEX[cmp0]]
                q1 = q[t + 1, s + 1, _CMP_TO_INDEX[cmp1]]

                q[t, s, idx] = (1.0 - p0) * q0 + p0 * q1

    return q


def exact_doob_curve(
    T: int,
    p0: float,
    p1: float,
    alpha: float,
    horizon: int,
) -> dict[str, Any]:
    """
    Exact CDF of the canonical Doob martingale stopping time for the
    deterministic NP event Gamma* with lexicographic tie-break.

    The terminal event Gamma* is the same as before. The stopping rule is now

        U_t^* >= alpha^{-1},

    equivalently

        P0(Gamma* | X^t) >= P0(Gamma*) / alpha.
    """
    boundary = choose_np_boundary(T, p0, alpha)
    k = boundary["k"]
    m = boundary["m"]
    r = boundary["r"]
    level = boundary["level"]

    c_star = unrank_fixed_weight_desc(T, m, r) if r > 0 else None

    q = build_conditional_np_prob_table(
        T=T,
        p0=p0,
        k=k,
        m=m,
        c_star=c_star,
    )

    conditional_prob_threshold = level / alpha

    alive: dict[tuple[int, int], float] = {(0, 0): 1.0}
    pmf = np.zeros(horizon + 1, dtype=float)

    for t in range(1, T + 1):
        new_alive: defaultdict[tuple[int, int], float] = defaultdict(float)
        cbit = int(c_star[t - 1]) if c_star is not None else 0

        for (s, cmp_state), mass in alive.items():
            for bit, prob in ((0, 1.0 - p1), (1, p1)):
                s_new = s + bit

                if c_star is None:
                    cmp_new = 0
                else:
                    cmp_new = update_cmp(cmp_state, bit, cbit)

                mass_new = mass * prob
                q_now = q[t, s_new, _CMP_TO_INDEX[cmp_new]]

                if q_now >= conditional_prob_threshold - 1e-14:
                    pmf[t] += mass_new
                else:
                    new_alive[(s_new, cmp_new)] += mass_new

        alive = dict(new_alive)

    cdf = np.cumsum(pmf)

    # Flat after the deadline.
    if T < horizon:
        cdf[T + 1 :] = cdf[T]

    return {
        **boundary,
        "name": f"Doob T={T}",
        "kind": f"doob_{T}",
        "label": rf"Doob $T={T}$",
        "c_star": c_star,
        "pmf": pmf,
        "cdf": cdf,
        "power": float(cdf[T]),
        "conditional_prob_threshold": float(conditional_prob_threshold),
    }


# ============================================================
# Exact GRO curve
# ============================================================

def exact_gro_curve(
    p0: float,
    p1: float,
    alpha: float,
    horizon: int,
) -> dict[str, Any]:
    """
    Exact CDF of the likelihood-ratio / GRO e-process under pi_1.
    """
    log_threshold = math.log(1.0 / alpha)
    log_e1 = math.log(p1 / p0)
    log_e0 = math.log((1.0 - p1) / (1.0 - p0))

    alive: dict[int, float] = {0: 1.0}
    pmf = np.zeros(horizon + 1, dtype=float)

    for t in range(1, horizon + 1):
        new_alive: defaultdict[int, float] = defaultdict(float)

        for s, mass in alive.items():
            for bit, prob in ((0, 1.0 - p1), (1, p1)):
                s_new = s + bit
                logw = s_new * log_e1 + (t - s_new) * log_e0
                mass_new = mass * prob

                if logw >= log_threshold - 1e-14:
                    pmf[t] += mass_new
                else:
                    new_alive[s_new] += mass_new

        alive = dict(new_alive)

    cdf = np.cumsum(pmf)

    return {
        "name": "GRO",
        "kind": "gro",
        "label": "GRO",
        "pmf": pmf,
        "cdf": cdf,
        "power": float(cdf[-1]),
    }


# ============================================================
# Experiment
# ============================================================

def run_experiment() -> dict[str, Any]:
    doob_curves = [
        exact_doob_curve(T=T, p0=P0, p1=P1, alpha=ALPHA, horizon=HORIZON)
        for T in DEADLINES
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


# ============================================================
# Plotting helpers
# ============================================================

def make_time_bins(plot_horizon: int, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    bin_width = math.ceil(plot_horizon / n_bins)
    lefts = np.arange(1, plot_horizon + 1, bin_width, dtype=int)
    rights = np.minimum(lefts + bin_width, plot_horizon + 1)
    return lefts, rights


def aggregate_pmf_on_bins(
    pmf: np.ndarray,
    lefts: np.ndarray,
    rights: np.ndarray,
) -> np.ndarray:
    # pmf[t] is P(tau=t), so bin [left,right-1] is pmf[left:right].
    return np.array(
        [pmf[left:right].sum() for left, right in zip(lefts, rights)],
        dtype=float,
    )


def fade_color(color: str, blend: float = 0.50) -> tuple[float, float, float]:
    rgb = np.array(to_rgb(color))
    return tuple((1.0 - blend) * rgb + blend * np.ones(3))


def tick_label_decimal(value: float) -> str:
    if abs(value) < 1e-12:
        return r"$0$"
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return rf"${text}$"


def curve_color(curve: dict[str, Any]) -> str:
    if curve["kind"] == "doob_10":
        return COL_DOOB_10
    if curve["kind"] == "doob_20":
        return COL_DOOB_20
    if curve["kind"] == "gro":
        return COL_GRO
    raise ValueError(f"Unknown curve kind: {curve['kind']}")


def curve_cdf_linestyle(curve: dict[str, Any]) -> str:
    if curve["kind"] == "doob_10":
        return CDF_LS_DOOB_10
    if curve["kind"] == "doob_20":
        return CDF_LS_DOOB_20
    if curve["kind"] == "gro":
        return CDF_LS_GRO
    raise ValueError(f"Unknown curve kind: {curve['kind']}")


def ordered_curves_for_plot(curves: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = ["doob_10", "doob_20", "gro"]
    return [next(curve for curve in curves if curve["kind"] == kind) for kind in order]


def plot_stacked_pmf_cdf(
    curves: list[dict[str, Any]],
    plot_max_t: int,
    n_bins: int,
    ymax_bin: float | None = None,
):
    plot_horizon = min(plot_max_t, HORIZON)
    times = np.arange(1, plot_horizon + 1)
    lefts, rights = make_time_bins(plot_horizon, n_bins)

    ordered_curves = ordered_curves_for_plot(curves)

    cdfs = [curve["cdf"][1 : plot_horizon + 1] for curve in ordered_curves]
    labels = [curve["label"] for curve in ordered_curves]
    colors = [curve_color(curve) for curve in ordered_curves]
    cdf_linestyles = [curve_cdf_linestyle(curve) for curve in ordered_curves]

    pmf_bins = [
        aggregate_pmf_on_bins(curve["pmf"], lefts, rights)
        for curve in ordered_curves
    ]

    if ymax_bin is None:
        data_max = max(float(np.max(p)) for p in pmf_bins)
        ymax_bin = max(0.20, 1.08 * data_max)
        ymax_bin = math.ceil(10.0 * ymax_bin) / 10.0

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(FIG_WIDTH, FIG_HEIGHT),
        sharex=False,
        gridspec_kw={"height_ratios": [1.0, 1.0]},
    )

    # ========================================================
    # Top panel: grouped coarse-binned PMF histogram
    # ========================================================

    for j, (left, right) in enumerate(zip(lefts, rights)):
        if j % 2 == 0:
            ax1.axvspan(left, right, color=BIN_SHADE_COLOR, zorder=0)
        ax1.axvline(left, color=BIN_EDGE_COLOR, linewidth=BIN_EDGE_LW, zorder=1)

    ax1.axvline(rights[-1], color=BIN_EDGE_COLOR, linewidth=BIN_EDGE_LW, zorder=1)

    n_channels = len(ordered_curves)

    for j, (left, right) in enumerate(zip(lefts, rights)):
        width = right - left
        usable_width = width * (1.0 - 2.0 * OUTER_MARGIN_FRAC)
        gap = width * INNER_GAP_FRAC
        bar_width = (usable_width - (n_channels - 1) * gap) / n_channels
        x_start = left + width * OUTER_MARGIN_FRAC

        for k, (pmf_bin, color, linestyle) in enumerate(
            zip(pmf_bins, colors, cdf_linestyles)
        ):
            x0 = x_start + k * (bar_width + gap)
            x_center = x0 + 0.5 * bar_width
            y = pmf_bin[j]

            face_color = fade_color(color, blend=BAR_FACE_BLEND)
            edge_color = fade_color(color, blend=BAR_EDGE_BLEND)

            ax1.bar(
                x=x0,
                height=y,
                width=bar_width,
                align="edge",
                color=face_color,
                edgecolor=edge_color,
                linewidth=BAR_EDGE_LW,
                zorder=3,
            )

            ax1.hlines(
                y,
                x0,
                x0 + bar_width,
                color=color,
                linewidth=BAR_TOP_LW,
                zorder=4,
            )

            # Grayscale cue matching the CDF linestyle.
            ax1.vlines(
                x_center,
                0.0,
                y,
                color=color,
                linewidth=0.9,
                linestyle=linestyle,
                zorder=4,
            )

    ax1.set_xlim(1, plot_horizon + 1)
    ax1.set_ylim(0.0, ymax_bin)
    ax1.grid(axis="y", alpha=0.18)

    top_ticks = [0.0, 0.1, 0.2, 0.3, 0.4]
    top_ticks = [tick for tick in top_ticks if tick <= ymax_bin + 1e-12]

    ax1.set_yticks(top_ticks)
    ax1.set_yticklabels(
        [tick_label_decimal(tick) for tick in top_ticks],
        fontsize=TICK_FONTSIZE,
    )
    ax1.tick_params(axis="y", length=2.5, width=0.7, pad=1)

    ax1.set_ylabel(
        r"$\pi_1^{\otimes}(\tau \in \mathrm{bin})$",
        fontsize=AXIS_LABEL_FONTSIZE,
    )

    handles = [
        Line2D(
            [0],
            [0],
            color=color,
            lw=BAR_TOP_LW + 0.9,
            linestyle=linestyle,
            label=label,
        )
        for color, linestyle, label in zip(colors, cdf_linestyles, labels)
    ]

    ax1.legend(
        handles=handles,
        loc="upper center",
        fontsize=LEGEND_FONTSIZE,
        ncol=3,
        frameon=True,
        handlelength=1.8,
        columnspacing=0.8,
        borderpad=0.22,
        bbox_to_anchor=(0.5, 1.03),
    )

    # ========================================================
    # Bottom panel: CDF
    # ========================================================

    for cdf, color, linestyle in zip(cdfs, colors, cdf_linestyles):
        ax2.step(
            times,
            cdf,
            where="post",
            color=color,
            linewidth=CDF_LW,
            linestyle=linestyle,
        )

    ax2.set_xlim(1, plot_horizon)
    ax2.set_ylim(0.0, 1.0)
    ax2.grid(axis="y", alpha=0.18)

    ax2.set_yticks([0.0, 0.5, 1.0])
    ax2.set_yticklabels([r"$0$", r"$0.5$", r"$1$"], fontsize=TICK_FONTSIZE)
    ax2.tick_params(axis="y", length=2.5, width=0.7, pad=1)

    ax2.set_ylabel(
        r"$\pi_1^{\otimes}(\tau \leq t)$",
        fontsize=AXIS_LABEL_FONTSIZE,
    )

    # ========================================================
    # X-axes
    # ========================================================

    centers = 0.5 * (lefts + rights - 1)
    bin_tick_labels = [
        rf"${left}\!-\!{right - 1}$" if (right - left) > 1 else rf"${left}$"
        for left, right in zip(lefts, rights)
    ]

    ax1.set_xticks(centers[::XTICK_EVERY])
    ax1.set_xticklabels(
        bin_tick_labels[::XTICK_EVERY],
        rotation=25,
        ha="right",
        fontsize=TICK_FONTSIZE,
    )
    ax1.tick_params(axis="x", length=2.5, width=0.7, pad=1)
    ax1.set_xlabel("")

    standard_ticks = np.arange(5, plot_horizon + 1, 5)
    ax2.set_xticks(standard_ticks)
    ax2.set_xticklabels(
        [rf"${int(t)}$" for t in standard_ticks],
        fontsize=TICK_FONTSIZE,
    )
    ax2.tick_params(axis="x", length=2.5, width=0.7, pad=1)
    ax2.set_xlabel(r"rejection time", fontsize=AXIS_LABEL_FONTSIZE)

    fig.tight_layout(h_pad=0.28)
    return fig


# ============================================================
# Saving
# ============================================================

def save_results(result: dict[str, Any]) -> None:
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


# ============================================================
# Main
# ============================================================

def main() -> None:
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
    fig.savefig(OUT_FIG_PNG, dpi=DPI)
    save_results(result)

    plt.show()

    print("\nSaved:")
    print(f"  {OUT_FIG_PNG}")
    print(f"  {OUT_CSV}")
    print(f"  {OUT_JSON}")


if __name__ == "__main__":
    main()