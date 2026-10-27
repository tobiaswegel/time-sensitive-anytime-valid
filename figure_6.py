"""Figure 6: exact Bernoulli benchmarks and simulated horizon-aware baselines."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from bernoulli import exact_constant_bernoulli_curve
from bernoulli import exact_doob_curve as doob_curve
from utils import aggregate_pmf_on_bins, fade_color, make_time_bins, save_figure, tau_to_pmf_cdf


def exact_doob_curve(T, p0, p1, alpha, horizon=None):
    """Return exact Doob probabilities for times 1 through the horizon."""
    curve = doob_curve(T, p0, p1, alpha, T if horizon is None else horizon)
    curve.update(label="Doob", kind="doob_exact", q_threshold=curve["conditional_prob_threshold"])
    for key in ("pmf", "cdf"):
        curve[key] = curve[key][1:]
    return curve


def sample_paths(horizon: int, p1: float, n_sim: int, seed: int) -> np.ndarray:
    """Generate independent Bernoulli observation paths with a fixed seed."""
    rng = np.random.default_rng(seed)
    return rng.binomial(1, p1, size=(n_sim, horizon))


# Exact constant Bernoulli betting curves: GRO and EDO


def exact_gro_curve(
    p0: float,
    p1: float,
    alpha: float,
    horizon: int,
) -> dict[str, Any]:
    """Compute exact rejection probabilities for the likelihood-ratio strategy."""
    return exact_constant_bernoulli_curve(
        p0=p0,
        p1=p1,
        alpha=alpha,
        horizon=horizon,
        p_action=p1,
        label="GRO",
        kind="gro_exact",
    )


def bernoulli_edo_growth(
    eta: float,
    p0: float,
    p1: float,
) -> float:
    """Return the optimized log moment defining the EDO exponent."""
    if not (0.0 <= eta < 1.0):
        raise ValueError("eta must lie in [0,1).")

    if eta == 0.0:
        return 0.0

    xi = 1.0 / (1.0 - eta)

    log_lr1 = math.log(p1 / p0)
    log_lr0 = math.log((1.0 - p1) / (1.0 - p0))

    log_z = np.logaddexp(
        math.log(p0) + xi * log_lr1,
        math.log(1.0 - p0) + xi * log_lr0,
    )

    return float((1.0 - eta) * log_z)


def solve_bernoulli_edo_eta(
    T_scale: float,
    p0: float,
    p1: float,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> float:
    """Find the Bernoulli EDO exponent by bisection."""
    if T_scale <= 0:
        raise ValueError("T_scale must be positive.")
    if not (0.0 < p0 < p1 < 1.0):
        raise ValueError("This script assumes 0 < p0 < p1 < 1.")

    target = 1.0 / T_scale
    max_growth = math.log(p1 / p0)

    if target >= max_growth:
        raise ValueError(
            "Bernoulli EDO equation has no solution for this T_scale: "
            f"1/T = {target:.6g} >= log(p1/p0) = {max_growth:.6g}."
        )

    lo = 0.0
    hi = 1.0 - 1e-12

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        value = bernoulli_edo_growth(mid, p0, p1)

        if value < target:
            lo = mid
        else:
            hi = mid

        if hi - lo <= tol:
            break

    return 0.5 * (lo + hi)


def bernoulli_edo_p_star(
    eta: float,
    p0: float,
    p1: float,
) -> float:
    """Return the Bernoulli parameter of the tilted EDO bet."""
    if not (0.0 <= eta < 1.0):
        raise ValueError("eta must lie in [0,1).")

    xi = 1.0 / (1.0 - eta)

    log_lr1 = math.log(p1 / p0)
    log_lr0 = math.log((1.0 - p1) / (1.0 - p0))

    log_w1 = math.log(p0) + xi * log_lr1
    log_w0 = math.log(1.0 - p0) + xi * log_lr0
    log_z = np.logaddexp(log_w1, log_w0)

    return float(math.exp(log_w1 - log_z))


def exact_edo_uncapped_curve(
    p0: float,
    p1: float,
    alpha: float,
    horizon: int,
    T_scale: float | None = None,
) -> dict[str, Any]:
    """Compute exact rejection probabilities for the uncapped EDO strategy."""
    if T_scale is None:
        T_scale = float(horizon)

    eta_star = solve_bernoulli_edo_eta(
        T_scale=T_scale,
        p0=p0,
        p1=p1,
    )

    p_star = bernoulli_edo_p_star(
        eta=eta_star,
        p0=p0,
        p1=p1,
    )

    curve = exact_constant_bernoulli_curve(
        p0=p0,
        p1=p1,
        alpha=alpha,
        horizon=horizon,
        p_action=p_star,
        label="EDO",
        kind="edo_exact",
    )

    curve["eta_star"] = float(eta_star)
    curve["p_star"] = float(p_star)
    curve["T_scale"] = float(T_scale)

    return curve


# STaR-Bets and Taga baselines


def safe_lambda_max(p0: float, eps: float = 1e-3) -> float:
    """Return the largest stake allowed by the endpoint safety margin."""
    return (1.0 - eps) / p0


def taga_empirical_kelly_lambda(
    score_sum: np.ndarray,
    second_moment_sum: np.ndarray,
    p0: float,
    eps: float = 1e-3,
) -> np.ndarray:
    """Estimate the Kelly stake from past scores and squared scores."""
    lam = np.divide(
        score_sum,
        second_moment_sum,
        out=np.zeros_like(score_sum, dtype=float),
        where=second_moment_sum > 0.0,
    )
    return np.clip(lam, 0.0, safe_lambda_max(p0, eps=eps))


def simulate_starbets_cdf(
    horizon: int,
    p0: float,
    p1: float,
    alpha: float,
    n_sim: int,
    seed: int,
    c: float,
    safe_eps: float = 1e-3,
    second_moment_clip: str = "paper",
) -> dict[str, Any]:
    """Simulate the target-recalculating STaR-Bets rule on Bernoulli paths."""
    x = sample_paths(horizon, p1, n_sim, seed)

    log_threshold = math.log(1.0 / alpha)
    logw = np.zeros(n_sim, dtype=float)
    second_moment_sum = np.zeros(n_sim, dtype=float)
    tau = np.full(n_sim, horizon + 1, dtype=int)
    alive = np.ones(n_sim, dtype=bool)

    for t in range(1, horizon + 1):
        idx = np.flatnonzero(alive)
        if idx.size == 0:
            break

        denom = max(t - 1, 1)
        variance_proxy = second_moment_sum[idx] / denom + c * p0 * horizon / float(denom * denom)

        if second_moment_clip == "paper":
            clip_upper = p0 * (1.0 - p0)
        elif second_moment_clip == "unit":
            clip_upper = 1.0
        else:
            raise ValueError("second_moment_clip must be 'paper' or 'unit'.")

        variance_proxy = np.clip(variance_proxy, 1e-12, clip_upper)

        remaining = horizon - t + 1
        budget = np.maximum(log_threshold - logw[idx], 0.0)
        ell = np.sqrt(2.0 * budget / (remaining * variance_proxy))
        ell = np.clip(ell, 0.0, safe_lambda_max(p0, eps=safe_eps))

        xt = x[idx, t - 1]
        logw[idx] += np.log1p(ell * (xt - p0))
        second_moment_sum[idx] += (xt - p0) ** 2

        hit = logw[idx] >= log_threshold
        tau[idx[hit]] = t
        alive[idx[hit]] = False

    pmf, cdf = tau_to_pmf_cdf(tau, horizon)

    return {
        "label": "STaR-Bets",
        "kind": "starbets_mc",
        "pmf": pmf,
        "cdf": cdf,
        "tau": tau,
    }


def epsilon_schedule(t: int, horizon: int, onset: float, trend: str) -> float:
    """Return the probability of an aggressive bet at the current time."""
    start = onset * horizon
    if t < start:
        return 0.0

    denom = max(horizon - start, 1e-12)
    frac = min(max((t - start) / denom, 0.0), 1.0)

    if trend == "linear":
        return frac
    if trend == "quadratic":
        return frac * frac

    raise ValueError("trend must be 'linear' or 'quadratic'.")


def simulate_taga_hedged_epsilon_cdf(
    horizon: int,
    p0: float,
    p1: float,
    alpha: float,
    n_sim: int,
    seed: int,
    safe_eps: float,
    schedules: tuple[tuple[float, str], ...] = (
        (0.25, "linear"),
        (0.5, "linear"),
        (0.75, "linear"),
        (0.25, "quadratic"),
        (0.5, "quadratic"),
        (0.75, "quadratic"),
    ),
) -> dict[str, Any]:
    """Simulate the Taga-inspired mixture of six randomized betting schedules."""
    rng = np.random.default_rng(seed)
    x = rng.binomial(1, p1, size=(n_sim, horizon))

    log_threshold = math.log(1.0 / alpha)
    lam_max = safe_lambda_max(p0, eps=safe_eps)
    k_schedules = len(schedules)

    tau = np.full(n_sim, horizon + 1, dtype=int)
    wealth = np.ones((n_sim, k_schedules), dtype=float)
    score_sum = np.zeros(n_sim, dtype=float)
    second_moment_sum = np.zeros(n_sim, dtype=float)
    alive = np.ones(n_sim, dtype=bool)

    for t in range(1, horizon + 1):
        idx = np.flatnonzero(alive)
        if idx.size == 0:
            break

        lam_emp = taga_empirical_kelly_lambda(
            score_sum[idx],
            second_moment_sum[idx],
            p0,
            eps=safe_eps,
        )

        xt = x[idx, t - 1]

        for j, (onset, trend) in enumerate(schedules):
            eps_t = epsilon_schedule(t, horizon, onset, trend)
            use_aggressive = rng.binomial(1, eps_t, size=idx.size).astype(bool)
            lam = np.where(use_aggressive, lam_max, lam_emp)
            wealth[idx, j] *= 1.0 + lam * (xt - p0)

        score_sum[idx] += xt - p0
        second_moment_sum[idx] += (xt - p0) ** 2

        mixture_wealth = wealth[idx].mean(axis=1)
        hit = np.log(mixture_wealth) >= log_threshold
        tau[idx[hit]] = t
        alive[idx[hit]] = False

    pmf, cdf = tau_to_pmf_cdf(tau, horizon)

    return {
        "label": "Taga et al.",
        "kind": "taga_hedged_mc",
        "pmf": pmf,
        "cdf": cdf,
        "tau": tau,
    }


# Build curves


def build_curves_for_horizon(args: argparse.Namespace, horizon: int) -> list[dict[str, Any]]:
    """Compute all five comparison strategies for one deadline."""
    curves: list[dict[str, Any]] = []

    curves.append(
        exact_doob_curve(
            T=horizon,
            p0=args.p0,
            p1=args.p1,
            alpha=args.alpha,
            horizon=horizon,
        )
    )

    curves.append(
        exact_gro_curve(
            p0=args.p0,
            p1=args.p1,
            alpha=args.alpha,
            horizon=horizon,
        )
    )

    curves.append(
        exact_edo_uncapped_curve(
            p0=args.p0,
            p1=args.p1,
            alpha=args.alpha,
            horizon=horizon,
            T_scale=float(horizon),
        )
    )

    curves.append(
        simulate_starbets_cdf(
            horizon=horizon,
            p0=args.p0,
            p1=args.p1,
            alpha=args.alpha,
            n_sim=args.n_sim,
            seed=args.seed,
            c=args.star_c,
            safe_eps=args.safe_eps,
            second_moment_clip=args.star_clip,
        )
    )

    curves.append(
        simulate_taga_hedged_epsilon_cdf(
            horizon=horizon,
            p0=args.p0,
            p1=args.p1,
            alpha=args.alpha,
            n_sim=args.n_sim,
            seed=args.seed + 107,
            safe_eps=args.safe_eps,
        )
    )

    return curves


# Plotting helpers


def configure_matplotlib(paper_style: bool = True) -> None:
    """Use a headless plotting backend and optionally apply paper styling."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-hard-deadline-stacked")

    import matplotlib

    # Save-first workflow. The saved PNG is displayed at the very end of main().
    matplotlib.use("Agg", force=True)

    import matplotlib.pyplot as plt  # noqa: F401

    if paper_style:
        plt.rcParams.update(
            {
                "font.family": "serif",
                "font.serif": [
                    "Times New Roman",
                    "Times",
                    "Nimbus Roman",
                    "STIXGeneral",
                ],
                "mathtext.fontset": "stix",
                "mathtext.rm": "STIXGeneral",
                "mathtext.it": "STIXGeneral:italic",
                "mathtext.bf": "STIXGeneral:bold",
                "font.size": 10.0,
                "axes.labelsize": 10.2,
                "legend.fontsize": 8.2,
                "xtick.labelsize": 8.8,
                "ytick.labelsize": 8.8,
                "axes.linewidth": 0.9,
                "lines.linewidth": 1.8,
                "savefig.bbox": "tight",
            }
        )


def style_for_kind(kind: str) -> dict[str, Any]:
    """Return the color and line style for a comparison strategy."""
    if kind == "doob_exact":
        return {"color": "black", "ls": "-", "lw": 2.3}
    if kind == "gro_exact":
        return {"color": "tab:blue", "ls": "--", "lw": 1.9}
    if kind == "edo_exact":
        return {"color": "tab:green", "ls": (0, (3, 1, 1, 1)), "lw": 1.9}
    if kind == "starbets_mc":
        return {"color": "tab:orange", "ls": "-.", "lw": 1.9}
    if kind == "taga_hedged_mc":
        return {"color": "tab:purple", "ls": ":", "lw": 1.9}
    raise ValueError(f"Unknown kind: {kind}")


def ordered_curves(curves: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order comparison strategies consistently across panels."""
    order = ["doob_exact", "gro_exact", "edo_exact", "starbets_mc", "taga_hedged_mc"]
    return [next(curve for curve in curves if curve["kind"] == kind) for kind in order]


def plot_one_panel(
    ax_top,
    ax_bottom,
    curves: list[dict[str, Any]],
    horizon: int,
    n_bins_total: int,
    ymax_top: float | None = None,
) -> tuple[list[Any], list[str]]:
    """Draw one deadline’s binned rejection probabilities and CDFs."""
    curves = ordered_curves(curves)
    times = np.arange(1, horizon + 1)
    lefts, rights = make_time_bins(horizon, n_bins_total)
    pmf_bins = [aggregate_pmf_on_bins(curve["pmf"], lefts, rights) for curve in curves]

    if ymax_top is None:
        data_max = max(float(np.max(p)) for p in pmf_bins)
        ymax_top = max(0.20, 1.08 * data_max)
        ymax_top = math.ceil(10.0 * ymax_top) / 10.0

    # --------------------------------------------------------
    # Top panel: grouped coarse-binned PMF histogram
    # --------------------------------------------------------
    bin_shade_color = "0.955"
    bin_edge_color = "0.82"
    n_channels = len(curves)

    outer_margin_frac = 0.06
    inner_gap_frac = 0.025

    bar_face_blend = 0.80
    bar_edge_blend = 0.45
    bar_edge_lw = 0.72
    bar_top_lw = 1.00

    for j, (left, right) in enumerate(zip(lefts, rights)):
        if j % 2 == 0:
            ax_top.axvspan(left, right, color=bin_shade_color, zorder=0)
        ax_top.axvline(left, color=bin_edge_color, linewidth=0.45, zorder=1)
    ax_top.axvline(rights[-1], color=bin_edge_color, linewidth=0.45, zorder=1)

    for j, (left, right) in enumerate(zip(lefts, rights)):
        width = right - left
        usable_width = width * (1.0 - 2.0 * outer_margin_frac)
        gap = width * inner_gap_frac
        bar_width = (usable_width - (n_channels - 1) * gap) / n_channels
        x_start = left + width * outer_margin_frac

        for k, curve in enumerate(curves):
            style = style_for_kind(curve["kind"])
            color = style["color"]
            linestyle = style["ls"]
            y = pmf_bins[k][j]

            x0 = x_start + k * (bar_width + gap)
            x_center = x0 + 0.5 * bar_width

            face_color = fade_color(color, blend=bar_face_blend)
            edge_color = fade_color(color, blend=bar_edge_blend)

            ax_top.bar(
                x=x0,
                height=y,
                width=bar_width,
                align="edge",
                color=face_color,
                edgecolor=edge_color,
                linewidth=bar_edge_lw,
                zorder=3,
            )

            ax_top.hlines(
                y,
                x0,
                x0 + bar_width,
                color=color,
                linewidth=bar_top_lw,
                zorder=4,
            )

            ax_top.vlines(
                x_center,
                0.0,
                y,
                color=color,
                linewidth=0.9,
                linestyle=linestyle,
                zorder=4,
            )

    ax_top.set_xlim(1, horizon + 1)
    ax_top.set_ylim(0.0, ymax_top)
    ax_top.grid(axis="y", alpha=0.18)
    ax_top.set_title(rf"$T={horizon}$", pad=6.0)

    centers = 0.5 * (lefts + rights - 1)
    bin_tick_labels = [
        rf"${left}\!-\!{right - 1}$" if (right - left) > 1 else rf"${left}$"
        for left, right in zip(lefts, rights)
    ]

    ax_top.set_xticks(centers)
    ax_top.set_xticklabels(bin_tick_labels, rotation=25, ha="right")
    ax_top.tick_params(axis="x", length=2.5, width=0.7, pad=1)
    ax_top.tick_params(axis="y", length=2.5, width=0.7, pad=1)

    # Legend handles.
    from matplotlib.lines import Line2D

    handles = []
    labels = []
    for curve in curves:
        style = style_for_kind(curve["kind"])
        handles.append(
            Line2D(
                [0],
                [0],
                color=style["color"],
                lw=style["lw"],
                linestyle=style["ls"],
                label=curve["label"],
            )
        )
        labels.append(curve["label"])

    # --------------------------------------------------------
    # Bottom panel: CDF
    # --------------------------------------------------------
    for curve in curves:
        style = style_for_kind(curve["kind"])
        ax_bottom.step(
            times,
            curve["cdf"],
            where="post",
            color=style["color"],
            linestyle=style["ls"],
            linewidth=style["lw"],
        )

    ax_bottom.set_xlim(1, horizon)
    ax_bottom.set_ylim(0.0, 1.0)
    ax_bottom.grid(axis="y", alpha=0.18)
    ax_bottom.tick_params(axis="x", length=2.5, width=0.7, pad=1)
    ax_bottom.tick_params(axis="y", length=2.5, width=0.7, pad=1)

    if horizon <= 15:
        xticks = np.arange(1, horizon + 1, 1)
    elif horizon <= 30:
        xticks = np.arange(5, horizon + 1, 5)
    else:
        xticks = np.arange(20, horizon + 1, 20)
        if xticks.size == 0 or xticks[-1] != horizon:
            xticks = np.append(xticks, horizon)

    ax_bottom.set_xticks(xticks)
    ax_bottom.set_xlabel(r"time $t$")

    return handles, labels


def plot_cdfs_multi_horizon(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Save side-by-side comparisons for all requested deadlines."""
    configure_matplotlib(paper_style=args.paper_style)

    import matplotlib.pyplot as plt

    horizons = list(args.horizons)
    n_panels = len(horizons)

    fig = plt.figure(
        figsize=(args.panel_width * n_panels, args.figure_height),
        constrained_layout=False,
    )

    gs = fig.add_gridspec(
        2,
        n_panels,
        height_ratios=[1.0, 1.0],
        hspace=args.hspace,
        wspace=args.wspace,
    )

    axes_top = [fig.add_subplot(gs[0, j]) for j in range(n_panels)]
    axes_bottom = [fig.add_subplot(gs[1, j]) for j in range(n_panels)]

    summaries: list[dict[str, Any]] = []
    legend_handles = None
    legend_labels = None

    for j, horizon in enumerate(horizons):
        curves = build_curves_for_horizon(args, horizon)

        handles, labels = plot_one_panel(
            ax_top=axes_top[j],
            ax_bottom=axes_bottom[j],
            curves=curves,
            horizon=horizon,
            n_bins_total=args.n_bins_total,
            ymax_top=None,
        )

        if legend_handles is None:
            legend_handles = handles
            legend_labels = labels

        summaries.append({"horizon": horizon, "curves": curves})

    # Left-side labels only on the first column.
    axes_top[0].set_ylabel(r"$\pi_1^{\otimes}(\tau \in \mathrm{bin})$")
    axes_bottom[0].set_ylabel(r"$\pi_1^{\otimes}(\tau \leq t)$")

    # Hide repeated y tick labels to reduce clutter.
    for j in range(1, n_panels):
        axes_top[j].set_yticklabels([])
        axes_bottom[j].set_yticklabels([])

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=min(5, len(legend_labels)),
        fontsize=args.legend_fontsize,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )

    fig.tight_layout(rect=[0, 0.0, 1, 0.94])
    save_figure(fig, args.output, dpi=args.dpi)
    plt.close(fig)

    return summaries


def display_plot_if_possible(path: str) -> None:
    """Display a saved figure when running in an IPython notebook."""
    try:
        from IPython import get_ipython

        if get_ipython() is None:
            return

        from IPython.display import Image, display

        display(Image(filename=path))
    except (ImportError, OSError) as exc:
        print(f"[display skipped] Could not display {path}: {exc}")


# CLI


def parse_args() -> argparse.Namespace:
    """Read experiment and output options from the command line."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot stacked PMF-bin / CDF panels for Bernoulli hard-deadline tests, "
            "showing exact Doob, exact GRO, uncapped EDO, STaR-Bets, and Taga et al."
        )
    )

    parser.add_argument("--p0", type=float, default=0.4, help="Null Bernoulli mean.")
    parser.add_argument("--p1", type=float, default=0.6, help="Alternative Bernoulli mean.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Test level.")

    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[10, 25, 100],
        help="Hard deadlines T to plot as side-by-side panels.",
    )

    parser.add_argument(
        "--n-sim",
        type=int,
        default=20000,
        help="Monte Carlo simulations for STaR-Bets and Taga.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed.")

    parser.add_argument(
        "--star-c",
        type=float,
        default=1.0,
        help="Variance-proxy bonus constant for STaR-Bets.",
    )

    parser.add_argument(
        "--star-clip",
        choices=["paper", "unit"],
        default="paper",
        help="Second-moment clipping for STaR-Bets.",
    )

    parser.add_argument(
        "--safe-eps",
        type=float,
        default=1e-3,
        help="Endpoint clipping margin for Taga-style bets.",
    )

    parser.add_argument(
        "--n-bins-total",
        type=int,
        default=6,
        help="Number of coarse PMF bins in the top panel of each column.",
    )

    parser.add_argument(
        "--output",
        default="figure_6.png",
        help="Figure path; both PNG and PDF versions are saved.",
    )
    parser.add_argument("--dpi", type=int, default=400, help="Output image DPI.")
    parser.add_argument("--panel-width", type=float, default=4.0, help="Width per panel.")
    parser.add_argument("--figure-height", type=float, default=4.2, help="Total figure height.")
    parser.add_argument("--legend-fontsize", type=float, default=8.2, help="Legend font size.")
    parser.add_argument("--hspace", type=float, default=0.22, help="Vertical subplot spacing.")
    parser.add_argument("--wspace", type=float, default=0.14, help="Horizontal subplot spacing.")

    parser.add_argument("--paper-style", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Display the saved figure at the end when running in notebook/IPython.",
    )

    return parser.parse_args()


# Main


def main() -> None:
    """Run the experiment and save its outputs."""
    args = parse_args()
    summaries = plot_cdfs_multi_horizon(args)

    print(
        f"Saved plots: {Path(args.output).with_suffix('.png')} and {Path(args.output).with_suffix('.pdf')}"
    )

    for summary in summaries:
        H = summary["horizon"]
        curves = summary["curves"]

        doob = next(curve for curve in curves if curve["kind"] == "doob_exact")
        edo = next(curve for curve in curves if curve["kind"] == "edo_exact")

        print(f"\n=== T = {H} ===")
        print("Exact Doob/NP uses the exact Proposition 2 threshold.")
        print(f"k: {doob['k']}")
        print(f"m = k - 1: {doob['m']}")
        print(f"r tied strings included: {doob['r']}")
        print(f"P0(Gamma*): {doob['level']:.12g}")
        print(f"conditional-prob threshold = P0(Gamma*) / alpha: {doob['q_threshold']:.12g}")

        if doob.get("c_star") is not None:
            print(f"c_star: {doob['c_star']}")

        print(f"EDO eta*: {edo['eta_star']:.12g}")
        print(f"EDO p*: {edo['p_star']:.12g}")

        for curve in ordered_curves(curves):
            print(f"{curve['label']}: hit by T = {curve['cdf'][-1]:.12g}")

    if args.show:
        display_plot_if_possible(str(Path(args.output).with_suffix(".png")))


if __name__ == "__main__":
    main()
