r"""
Power of the Bernoulli constant strategy from the proposition.

This script:
    - computes the constant-strategy quantities for a Bernoulli model,
    - estimates power under pi_1^{\otimes},
    - plots power together with the proposition lower/upper bounds,
    - marks the threshold T = 1 / KL(pi_1 || pi_0).

Model:
    pi_0 = Bernoulli(p0)
    pi_1 = Bernoulli(p1)

Colab-safe:
    uses parse_known_args() so Jupyter/Colab's internal arguments are ignored.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


# ============================================================
# Core helpers
# ============================================================

def logsumexp_pair(a: float, b: float) -> float:
    """Stable log(exp(a) + exp(b)) for two scalars."""
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def check_probability(p: float, name: str) -> None:
    if not (0.0 < p < 1.0):
        raise ValueError(f"{name} must lie in (0, 1).")


def bernoulli_kl(p1: float, p0: float) -> float:
    """KL(Bern(p1) || Bern(p0))."""
    check_probability(p0, "p0")
    check_probability(p1, "p1")
    return (
        p1 * math.log(p1 / p0)
        + (1.0 - p1) * math.log((1.0 - p1) / (1.0 - p0))
    )


def bernoulli_likelihood_ratio_values(p0: float, p1: float) -> tuple[float, float]:
    """
    Likelihood-ratio values under X=0 and X=1:
        l0 = (1-p1)/(1-p0)
        l1 = p1/p0
    """
    return (1.0 - p1) / (1.0 - p0), p1 / p0


def bernoulli_constant_e_variable(
    p0: float,
    p1: float,
    eta: float,
) -> tuple[float, float]:
    """
    Constant e-variable associated with the eta-tilted construction.

    Returns:
        f0, f1
    """
    l0, l1 = bernoulli_likelihood_ratio_values(p0, p1)
    exponent = 1.0 / (1.0 - eta)
    normalizer = (1.0 - p0) * (l0 ** exponent) + p0 * (l1 ** exponent)
    return (l0 ** exponent) / normalizer, (l1 ** exponent) / normalizer


def log_optimal_eta_moment(p0: float, p1: float, eta: float) -> float:
    """
    Computes:
        (1 - eta) log E_{pi_0}[L^{1/(1-eta)}]
    """
    l0, l1 = bernoulli_likelihood_ratio_values(p0, p1)
    exponent = 1.0 / (1.0 - eta)
    log_z = logsumexp_pair(
        math.log1p(-p0) + exponent * math.log(l0),
        math.log(p0) + exponent * math.log(l1),
    )
    return (1.0 - eta) * log_z


def solve_eta_star(
    p0: float,
    p1: float,
    time_scale: float,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> dict[str, float]:
    """
    Solve eta_star from:
        log_optimal_eta_moment(p0, p1, eta_star) = 1 / T
    """
    if time_scale <= 0.0:
        raise ValueError("time_scale must be positive.")

    target = 1.0 / time_scale
    lo = 1e-12
    hi = 1.0 - 1e-12

    flo = log_optimal_eta_moment(p0, p1, lo) - target
    fhi = log_optimal_eta_moment(p0, p1, hi) - target

    if flo >= 0.0 or fhi <= 0.0:
        raise ValueError("No eta_star in (0, 1) for this T.")

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fmid = log_optimal_eta_moment(p0, p1, mid) - target

        if fmid <= 0.0:
            lo = mid
        else:
            hi = mid

        if hi - lo <= tol:
            break

    eta_star = 0.5 * (lo + hi)
    f0_star, f1_star = bernoulli_constant_e_variable(p0, p1, eta_star)

    return {
        "eta_star": float(eta_star),
        "f0_star": float(f0_star),
        "f1_star": float(f1_star),
        "B_star": float(max(f0_star, f1_star)),
    }


def compute_drift(p1: float, f0: float, f1: float) -> float:
    """Log-wealth drift under pi_1."""
    return (1.0 - p1) * math.log(f0) + p1 * math.log(f1)


def log_mgf_under_pi1(p1: float, f0: float, f1: float, kappa: float) -> float:
    """log E_{pi_1}[f(X)^kappa]."""
    return math.log((1.0 - p1) * (f0 ** kappa) + p1 * (f1 ** kappa))


def solve_kappa(
    p1: float,
    f0: float,
    f1: float,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> float:
    """
    Solve kappa > 0 such that:
        E_{pi_1}[f(X)^kappa] = 1.
    """
    drift = compute_drift(p1, f0, f1)
    if drift >= 0.0:
        return 0.0

    lo = 1e-12
    hi = 1.0

    while log_mgf_under_pi1(p1, f0, f1, hi) <= 0.0:
        hi *= 2.0
        if hi > 1e6:
            raise RuntimeError("Could not bracket kappa root.")

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        val = log_mgf_under_pi1(p1, f0, f1, mid)

        if val <= 0.0:
            lo = mid
        else:
            hi = mid

        if hi - lo <= tol:
            break

    return float(0.5 * (lo + hi))


# ============================================================
# Confidence intervals and power estimation
# ============================================================

def wilson_interval(
    successes: int,
    trials: int,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    """Wilson confidence interval for a Bernoulli proportion."""
    if trials <= 0:
        return 0.0, 1.0

    phat = successes / trials
    denom = 1.0 + z * z / trials
    center = (phat + z * z / (2.0 * trials)) / denom
    radius = (
        z
        * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * trials)) / trials)
        / denom
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def estimate_power_monte_carlo(
    p1: float,
    alpha: float,
    f0: float,
    f1: float,
    kappa: float,
    n_sim: int,
    batch_size: int,
    max_steps: int,
    trunc_tol: float,
    seed: int,
) -> dict[str, Any]:
    r"""
    Estimate power under pi_1^{\otimes} by Monte Carlo.

    Uses an early-stopping truncation rule for deeply sub-threshold paths,
    together with an upper bound on the residual tail contribution.
    """
    rng = np.random.default_rng(seed)

    log_thr = math.log(1.0 / alpha)
    step0 = math.log(f0)
    step1 = math.log(f1)

    if not (kappa > 0.0):
        return {
            "power_estimate": 1.0,
            "ci_low": 1.0,
            "ci_high": 1.0,
            "hit_count": n_sim,
            "n_sim": n_sim,
            "cutoff_bias_upper": 0.0,
            "method": "exact-power-one",
        }

    log_cutoff = (math.log(trunc_tol) - kappa * math.log(alpha)) / kappa

    total_hits = 0
    total_tail_upper = 0.0

    remaining = n_sim
    while remaining > 0:
        current_batch = min(batch_size, remaining)
        remaining -= current_batch

        logw = np.zeros(current_batch, dtype=float)
        alive = np.ones(current_batch, dtype=bool)
        batch_hits = 0

        for _ in range(max_steps):
            if not np.any(alive):
                break

            x = rng.binomial(1, p1, size=int(np.sum(alive)))
            logw_alive = logw[alive]
            logw_alive += np.where(x == 1, step1, step0)
            logw[alive] = logw_alive

            hit_mask = alive & (logw >= log_thr)
            batch_hits += int(np.sum(hit_mask))
            alive[hit_mask] = False

            cutoff_mask = alive & (logw <= log_cutoff)
            total_tail_upper += float(np.sum(trunc_tol * cutoff_mask))
            alive[cutoff_mask] = False

        if np.any(alive):
            total_tail_upper += float(
                np.sum(
                    np.minimum(
                        np.exp(kappa * (math.log(alpha) + logw[alive])),
                        1.0,
                    )
                )
            )

        total_hits += batch_hits

    power_estimate = total_hits / n_sim
    ci_low, ci_high = wilson_interval(total_hits, n_sim)
    ci_high = min(1.0, ci_high + total_tail_upper / n_sim)

    return {
        "power_estimate": power_estimate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "hit_count": total_hits,
        "n_sim": n_sim,
        "cutoff_bias_upper": total_tail_upper / n_sim,
        "method": "monte-carlo",
    }


# ============================================================
# Result construction
# ============================================================

def build_row(
    time_scale: float,
    p0: float,
    p1: float,
    alpha: float,
    n_sim: int,
    batch_size: int,
    max_steps: int,
    trunc_tol: float,
    seed: int,
) -> dict[str, Any]:
    theorem = solve_eta_star(p0=p0, p1=p1, time_scale=time_scale)

    f0 = theorem["f0_star"]
    f1 = theorem["f1_star"]
    drift = compute_drift(p1=p1, f0=f0, f1=f1)
    threshold = 1.0 / bernoulli_kl(p1, p0)

    row: dict[str, Any] = {
        "T": float(time_scale),
        "threshold_T": float(threshold),
        "eta_star": theorem["eta_star"],
        "f0_star": f0,
        "f1_star": f1,
        "B_star": theorem["B_star"],
        "drift": drift,
    }

    if drift >= -1e-14:
        row.update(
            {
                "kappa": 0.0,
                "power_estimate": 1.0,
                "ci_low": 1.0,
                "ci_high": 1.0,
                "prop_lower": 1.0,
                "prop_upper": 1.0,
                "method": "exact-power-one",
                "hit_count": None,
                "n_sim": None,
                "cutoff_bias_upper": 0.0,
            }
        )
        return row

    kappa = solve_kappa(p1=p1, f0=f0, f1=f1)
    mc = estimate_power_monte_carlo(
        p1=p1,
        alpha=alpha,
        f0=f0,
        f1=f1,
        kappa=kappa,
        n_sim=n_sim,
        batch_size=batch_size,
        max_steps=max_steps,
        trunc_tol=trunc_tol,
        seed=seed,
    )

    row.update(
        {
            "kappa": kappa,
            "power_estimate": mc["power_estimate"],
            "ci_low": mc["ci_low"],
            "ci_high": mc["ci_high"],
            "prop_lower": (alpha / theorem["B_star"]) ** kappa,
            "prop_upper": alpha ** kappa,
            "method": mc["method"],
            "hit_count": mc["hit_count"],
            "n_sim": mc["n_sim"],
            "cutoff_bias_upper": mc["cutoff_bias_upper"],
        }
    )
    return row


def sanitize_row(row: dict[str, Any]) -> dict[str, Any]:
    return dict(row)


def save_results(
    results: list[dict[str, Any]],
    args: argparse.Namespace,
    output_path: str | Path,
) -> None:
    payload = {
        "parameters": {
            "p0": args.p0,
            "p1": args.p1,
            "alpha": args.alpha,
            "T_values": list(args.T_values),
            "n_sim": args.n_sim,
            "batch_size": args.batch_size,
            "max_steps": args.max_steps,
            "trunc_tol": args.trunc_tol,
            "seed": args.seed,
        },
        "results": [sanitize_row(row) for row in results],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_results(input_path: str | Path) -> list[dict[str, Any]]:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Results file not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    return payload["results"]


# ============================================================
# Plotting
# ============================================================

def configure_plot_fonts(plt_module) -> None:
    plt_module.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral"],
            "mathtext.fontset": "stix",
            "mathtext.rm": "STIXGeneral",
            "mathtext.it": "STIXGeneral:italic",
            "mathtext.bf": "STIXGeneral:bold",
            "font.size": 14,
            "axes.labelsize": 16,
            "legend.fontsize": 12,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "axes.linewidth": 1.1,
        }
    )


def maybe_plot(
    results: list[dict[str, Any]],
    output_path: str,
    show_plot: bool,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-power-proposition")

    import matplotlib

    if not show_plot:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    configure_plot_fonts(plt)

    t_values = np.array([row["T"] for row in results], dtype=float)
    power = np.array([row["power_estimate"] for row in results], dtype=float)
    ci_low = np.array([row["ci_low"] for row in results], dtype=float)
    ci_high = np.array([row["ci_high"] for row in results], dtype=float)
    lower = np.array([row["prop_lower"] for row in results], dtype=float)
    upper = np.array([row["prop_upper"] for row in results], dtype=float)
    threshold_t = float(results[0]["threshold_T"])

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.errorbar(
        t_values,
        power,
        yerr=np.vstack([power - ci_low, ci_high - power]),
        fmt="o-",
        color="tab:blue",
        ecolor="tab:blue",
        elinewidth=1.25,
        capsize=3.5,
        capthick=1.25,
        markersize=5.5,
        linewidth=2.0,
        label="Power",
    )

    ax.plot(
        t_values,
        lower,
        linestyle="--",
        color="tab:orange",
        linewidth=2.0,
        label="Lower bound",
    )
    ax.plot(
        t_values,
        upper,
        linestyle="-.",
        color="tab:green",
        linewidth=2.0,
        label="Upper bound",
    )
    ax.axvline(
        threshold_t,
        color="black",
        linestyle=":",
        linewidth=1.6,
        label=rf"$1/\mathrm{{KL}} = {threshold_t:.3f}$",
    )

    ax.set_xlabel(r"time scale $T$")
    ax.set_ylabel(r"power under $\pi_1^{\otimes}$")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)

    ax.legend(
        loc="lower right",
        framealpha=0.95,
        handlelength=2.3,
        borderpad=0.35,
        labelspacing=0.35,
    )

    fig.tight_layout()

    save_path = output_path or "power_proposition_bernoulli.png"
    fig.savefig(save_path, dpi=450, bbox_inches="tight")
    print(f"Saved plot to {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the power of the Bernoulli constant strategy from the proposition, "
            "and mark the threshold T = 1 / KL(pi1 || pi0)."
        )
    )

    parser.add_argument("--p0", type=float, default=0.5, help="Null Bernoulli parameter.")
    parser.add_argument("--p1", type=float, default=0.7, help="Alternative Bernoulli parameter.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Type-I level.")

    parser.add_argument(
        "--T-values",
        type=float,
        nargs="+",
        default=[4, 6, 8, 10, 11, 12, 13, 14, 16, 20, 30],
        help="Time scales T to evaluate.",
    )

    parser.add_argument(
        "--n-sim",
        type=int,
        default=5000,
        help="Monte Carlo paths for T below threshold.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Simulation batch size.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200000,
        help="Maximum steps per path.",
    )
    parser.add_argument(
        "--trunc-tol",
        type=float,
        default=1e-8,
        help=(
            "Per-path upper-bound tolerance used to stop deep sub-threshold "
            "paths early."
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")

    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting and only print the table.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional path to save the figure.",
    )
    parser.add_argument(
        "--results-path",
        type=str,
        default="",
        help="Optional JSON path for saving computed results or reloading them later.",
    )
    parser.add_argument(
        "--reuse-results",
        action="store_true",
        help="Load results from --results-path and skip recomputing.",
    )

    # Colab/Jupyter passes its own arguments, e.g. "-f kernel.json".
    # parse_known_args keeps this script usable both in notebooks and from CLI.
    args, _ = parser.parse_known_args()
    return args


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()

    if args.reuse_results and not args.results_path:
        raise ValueError("--reuse-results requires --results-path.")

    if args.reuse_results:
        results = load_results(args.results_path)
    else:
        results = []
        for idx, time_scale in enumerate(args.T_values):
            row = build_row(
                time_scale=float(time_scale),
                p0=args.p0,
                p1=args.p1,
                alpha=args.alpha,
                n_sim=args.n_sim,
                batch_size=args.batch_size,
                max_steps=args.max_steps,
                trunc_tol=args.trunc_tol,
                seed=args.seed + idx,
            )
            results.append(row)

        if args.results_path:
            save_results(results, args, args.results_path)

    header = (
        "      T      drift      kappa  power_est   ci_low  ci_high  "
        "prop_low prop_up method"
    )
    print(header)
    print("-" * len(header))

    for row in results:
        print(
            f"{row['T']:7.2f} "
            f"{row['drift']:10.6f} "
            f"{row['kappa']:10.6f} "
            f"{row['power_estimate']:10.6f} "
            f"{row['ci_low']:8.6f} "
            f"{row['ci_high']:8.6f} "
            f"{row['prop_lower']:8.6f} "
            f"{row['prop_upper']:8.6f} "
            f"{row['method']}"
        )

    if not args.no_plot or args.output:
        maybe_plot(results, output_path=args.output, show_plot=not args.no_plot)


if __name__ == "__main__":
    main()