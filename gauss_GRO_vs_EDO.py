"""
Gaussian uncapped GRO vs uncapped exponential-reward approximation.

Model:
    pi_0 = N(0, 1)
    pi_1 = N(1/4, 1)

Strategies:
    GRO:        a = mu
    EDO/Approx: a*(T) = mu + 2 sigma^2 / (mu T)

Outputs:
    - stacked PMF-bin / CDF figure
    - exact step-CDF figure
    - summary CSV
    - curves JSON

No Bellman. No capping.
Uses common random numbers for cleaner curve comparisons.
"""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D


# ============================================================
# Configuration
# ============================================================

MU = 1.0 / 4.0
SIGMA = 1.0

ALPHA = 0.05
U0 = ALPHA

T_VALUES = [8.0, 14.0, 60.0]

HORIZON = 250
PLOT_MAX_T = 120

N_SIM_ALT = 300_000
N_SIM_NULL = 100_000
BATCH_SIZE = 25_000
SEED = 0

FIG_WIDTH = 6.2
FIG_HEIGHT = 4.35
DPI = 450

EXACT_CDF_FIG_WIDTH = 6.2
EXACT_CDF_FIG_HEIGHT = 3.4

N_BINS_TOTAL = 15
XTICK_EVERY = 2

OUT_DIR = Path(".")
OUT_STACKED_PNG = OUT_DIR / "gaussian_uncapped_gro_vs_approx_stacked.png"
OUT_STACKED_PDF = OUT_DIR / "gaussian_uncapped_gro_vs_approx_stacked.pdf"
OUT_CSV = OUT_DIR / "gaussian_uncapped_gro_vs_approx_summary.csv"
OUT_JSON = OUT_DIR / "gaussian_uncapped_gro_vs_approx_curves.json"


# ============================================================
# Plot style
# ============================================================

def configure_fonts() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral"],
            "mathtext.fontset": "stix",
            "mathtext.rm": "STIXGeneral",
            "mathtext.it": "STIXGeneral:italic",
            "mathtext.bf": "STIXGeneral:bold",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "legend.fontsize": 10.5,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.linewidth": 1.0,
            "lines.linewidth": 2.3,
            "savefig.bbox": "tight",
        }
    )


COL_GRO = "tab:purple"
COL_BY_T = {
    8.0: "tab:blue",
    14.0: "tab:orange",
    60.0: "tab:green",
}

# CDF line styles for black-and-white readability.
CDF_LS_GRO = "-"
CDF_LS_BY_T = {
    8.0: "--",
    14.0: "-.",
    60.0: ":",
}

# Top panel: grouped histogram
BAR_ALPHA = 0.18
BAR_EDGE_BLEND = 0.50
BAR_EDGE_LW = 0.85
BAR_TOP_LW = 1.15

BIN_EDGE_LW = 0.55
BIN_SHADE_COLOR = "0.955"
BIN_EDGE_COLOR = "0.82"

OUTER_MARGIN_FRAC = 0.07
INNER_GAP_FRAC = 0.040

# Bottom panel: CDF
CDF_LW = 1.45
EXACT_CDF_LW = 1.6

LEGEND_FONTSIZE = 8.2
TICK_FONTSIZE = 8.5
AXIS_LABEL_FONTSIZE = 10.2


# ============================================================
# Gaussian betting helpers
# ============================================================

def gaussian_kl(mu: float, sigma: float) -> float:
    """KL(N(mu, sigma^2) || N(0, sigma^2))."""
    return mu * mu / (2.0 * sigma * sigma)


def gaussian_log_e(action_a: float, x: np.ndarray, sigma: float) -> np.ndarray:
    """
    Gaussian e-variable:
        E_a(x) = exp(a x / sigma^2 - a^2 / (2 sigma^2)).
    """
    return action_a * x / (sigma * sigma) - (
        action_a * action_a
    ) / (2.0 * sigma * sigma)


def gaussian_gro_action(mu: float) -> float:
    """GRO action for N(0, sigma^2) vs N(mu, sigma^2)."""
    return float(mu)


def gaussian_approx_action(mu: float, sigma: float, time_scale: float) -> float:
    """
    Exponential-reward stationary approximation:
        a*(T) = mu + 2 sigma^2 / (mu T).
    """
    if time_scale <= 0.0:
        raise ValueError("time_scale must be positive.")
    if abs(mu) <= 1e-15:
        raise ValueError("mu must be nonzero.")
    return float(mu + 2.0 * sigma * sigma / (mu * time_scale))


def gaussian_eta_star(mu: float, sigma: float, time_scale: float) -> float:
    """
    Gaussian closed form:
        eta* = 2 sigma^2 / (mu^2 T + 2 sigma^2).
    """
    return float(
        2.0 * sigma * sigma / (mu * mu * time_scale + 2.0 * sigma * sigma)
    )


def fmt_t_value(time_scale: float) -> int | float:
    return int(time_scale) if float(time_scale).is_integer() else float(time_scale)


# ============================================================
# Simulation helpers
# ============================================================

def tau_to_pmf_cdf(tau: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    pmf = np.bincount(tau, minlength=horizon + 2)[1 : horizon + 1] / tau.size
    return pmf, np.cumsum(pmf)


def discounted_value_from_tau(
    tau: np.ndarray, time_scale: float, horizon: int
) -> float:
    hit = tau <= horizon
    return float(np.mean(np.where(hit, np.exp(-tau / float(time_scale)), 0.0)))


def first_hit_times_from_paths(
    x_paths: np.ndarray,
    sigma: float,
    action_a: float,
    horizon: int,
    u0: float,
) -> np.ndarray:
    """Compute first hitting times for one constant action on pre-generated paths."""
    log_e = gaussian_log_e(action_a, x_paths[:, :horizon], sigma)
    cum_log_u = math.log(u0) + np.cumsum(log_e, axis=1)

    hit_matrix = cum_log_u >= -1e-14
    hit_any = hit_matrix.any(axis=1)

    tau = np.full(x_paths.shape[0], horizon + 1, dtype=np.int32)
    tau[hit_any] = np.argmax(hit_matrix[hit_any], axis=1).astype(np.int32) + 1
    return tau


def simulate_policies_common_random_numbers(
    mu_true: float,
    sigma: float,
    policies: list[dict],
    horizon: int,
    n_sim: int,
    batch_size: int,
    seed: int,
    u0: float,
) -> dict[str, dict]:
    """
    Simulate all policies on common Gaussian paths.
    This reduces visual Monte Carlo variation between curves.
    """
    rng = np.random.default_rng(seed)

    tau_by_policy = {
        policy["name"]: np.empty(n_sim, dtype=np.int32)
        for policy in policies
    }

    for start in range(0, n_sim, batch_size):
        end = min(start + batch_size, n_sim)
        x_paths = rng.normal(
            loc=mu_true,
            scale=sigma,
            size=(end - start, horizon),
        )

        for policy in policies:
            tau_by_policy[policy["name"]][start:end] = first_hit_times_from_paths(
                x_paths=x_paths,
                sigma=sigma,
                action_a=policy["action_a"],
                horizon=horizon,
                u0=u0,
            )

    results = {}
    for policy in policies:
        tau = tau_by_policy[policy["name"]]
        pmf, cdf = tau_to_pmf_cdf(tau, horizon)
        results[policy["name"]] = {
            "tau": tau,
            "pmf": pmf,
            "cdf": cdf,
            "n_sim": int(n_sim),
        }

    return results


# ============================================================
# Experiment
# ============================================================

def validate_config() -> None:
    if SIGMA <= 0.0:
        raise ValueError("SIGMA must be positive.")
    if not (0.0 < ALPHA < 1.0):
        raise ValueError("ALPHA must lie in (0, 1).")
    if not (0.0 < U0 < 1.0):
        raise ValueError("U0 must lie in (0, 1).")
    if HORIZON < 1:
        raise ValueError("HORIZON must be at least 1.")
    if PLOT_MAX_T > HORIZON:
        raise ValueError("PLOT_MAX_T cannot exceed HORIZON.")
    if any(float(t) <= 0.0 for t in T_VALUES):
        raise ValueError("All T values must be positive.")

    missing_colors = set(map(float, T_VALUES)) - set(COL_BY_T)
    if missing_colors:
        raise ValueError(f"Missing colors for T values: {sorted(missing_colors)}")

    missing_linestyles = set(map(float, T_VALUES)) - set(CDF_LS_BY_T)
    if missing_linestyles:
        raise ValueError(
            f"Missing CDF line styles for T values: {sorted(missing_linestyles)}"
        )


def make_policies() -> list[dict]:
    policies = [
        {
            "name": "GRO",
            "kind": "gro",
            "label": "GRO",
            "T": None,
            "eta_star": None,
            "action_a": gaussian_gro_action(MU),
        }
    ]

    for time_scale in T_VALUES:
        policies.append(
            {
                "name": f"EDO_T_{fmt_t_value(time_scale)}",
                "kind": "approx",
                "label": rf"EDO $T={fmt_t_value(time_scale)}$",
                "T": float(time_scale),
                "eta_star": gaussian_eta_star(MU, SIGMA, time_scale),
                "action_a": gaussian_approx_action(MU, SIGMA, time_scale),
            }
        )

    return policies


def run_experiment() -> dict:
    validate_config()

    kl_10 = gaussian_kl(MU, SIGMA)
    policies = make_policies()

    print(f"pi_0 = N(0, {SIGMA**2:g})")
    print(f"pi_1 = N({MU:g}, {SIGMA**2:g})")
    print(f"alpha = {ALPHA}")
    print(f"T values = {T_VALUES}")
    print(f"horizon = {HORIZON}")
    print(f"plot max t = {PLOT_MAX_T}")
    print(f"KL(pi_1 || pi_0) = {kl_10:.8g}")
    print(f"1 / KL(pi_1 || pi_0) = {1.0 / kl_10:.8g}")
    print("capping = False")
    print("common random numbers = True")
    print("\nPolicies:")

    for policy in policies:
        eta = "-" if policy["eta_star"] is None else f"{policy['eta_star']:.6g}"
        print(f"  {policy['label']:<14} a = {policy['action_a']:.6g}, eta = {eta}")

    print()

    alt_results = simulate_policies_common_random_numbers(
        mu_true=MU,
        sigma=SIGMA,
        policies=policies,
        horizon=HORIZON,
        n_sim=N_SIM_ALT,
        batch_size=BATCH_SIZE,
        seed=SEED + 123,
        u0=U0,
    )

    null_results = simulate_policies_common_random_numbers(
        mu_true=0.0,
        sigma=SIGMA,
        policies=policies,
        horizon=HORIZON,
        n_sim=N_SIM_NULL,
        batch_size=BATCH_SIZE,
        seed=SEED + 456,
        u0=U0,
    )

    curves = []
    rows = []

    for policy in policies:
        name = policy["name"]
        alt = alt_results[name]
        null = null_results[name]

        curves.append(
            {
                "kind": policy["kind"],
                "label": policy["label"],
                "T": policy["T"],
                "eta_star": policy["eta_star"],
                "action_a": float(policy["action_a"]),
                "pmf": alt["pmf"],
                "cdf": alt["cdf"],
                "n_sim": alt["n_sim"],
            }
        )

        row = {
            "policy": policy["label"],
            "T": policy["T"],
            "eta_star": policy["eta_star"],
            "action_a": float(policy["action_a"]),
            f"hit_by_{PLOT_MAX_T}_under_pi1": float(alt["cdf"][PLOT_MAX_T - 1]),
            "hit_by_H_under_pi1": float(alt["cdf"][-1]),
            "hit_by_H_under_pi0": float(null["cdf"][-1]),
        }

        for time_scale in T_VALUES:
            row[f"exp_reward_T{fmt_t_value(time_scale)}"] = discounted_value_from_tau(
                alt["tau"],
                time_scale,
                HORIZON,
            )

        rows.append(row)

    return {
        "policies": policies,
        "curves": curves,
        "summary": pd.DataFrame(rows),
        "kl_10": kl_10,
        "inv_kl": 1.0 / kl_10,
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
    return np.array(
        [pmf[left - 1 : right - 1].sum() for left, right in zip(lefts, rights)],
        dtype=float,
    )


def fade_color(color: str, blend: float = 0.50) -> tuple[float, float, float]:
    """
    Blend a color with white.
    blend=0 gives the original color; blend=1 gives white.
    """
    rgb = np.array(to_rgb(color))
    return tuple((1.0 - blend) * rgb + blend * np.ones(3))


def curve_color(curve: dict) -> str:
    if curve["kind"] == "gro":
        return COL_GRO
    return COL_BY_T[float(curve["T"])]


def curve_cdf_linestyle(curve: dict) -> str:
    if curve["kind"] == "gro":
        return CDF_LS_GRO
    return CDF_LS_BY_T[float(curve["T"])]


def ordered_curves_for_plot(curves: list[dict]) -> list[dict]:
    ordered = [curve for curve in curves if curve["kind"] == "gro"]
    ordered += [
        next(
            curve
            for curve in curves
            if curve["kind"] == "approx" and float(curve["T"]) == float(time_scale)
        )
        for time_scale in T_VALUES
    ]
    return ordered


# ============================================================
# Figure 1: stacked PMF-bin + CDF
# ============================================================

def plot_stacked_pmf_cdf(
    curves: list[dict],
    plot_max_t: int,
    n_bins: int,
    ymax_bin: float | None = None,
):
    plot_horizon = min(plot_max_t, HORIZON)
    times = np.arange(1, plot_horizon + 1)
    lefts, rights = make_time_bins(plot_horizon, n_bins)

    ordered_curves = ordered_curves_for_plot(curves)

    pmfs = [curve["pmf"][:plot_horizon] for curve in ordered_curves]
    cdfs = [curve["cdf"][:plot_horizon] for curve in ordered_curves]
    labels = [curve["label"] for curve in ordered_curves]
    colors = [curve_color(curve) for curve in ordered_curves]
    cdf_linestyles = [curve_cdf_linestyle(curve) for curve in ordered_curves]

    pmf_bins = [aggregate_pmf_on_bins(pmf, lefts, rights) for pmf in pmfs]

    if ymax_bin is None:
        ymax_bin = max(0.11, 1.08 * max(float(np.max(p)) for p in pmf_bins))

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

            edge_color = fade_color(color, blend=BAR_EDGE_BLEND)

            ax1.bar(
                x=x0,
                height=y,
                width=bar_width,
                align="edge",
                color=color,
                alpha=BAR_ALPHA,
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

            # Central vertical cue matching the CDF linestyle.
            ax1.vlines(
                x_center,
                0.0,
                y,
                color=color,
                linewidth=0.9,
                linestyle=linestyle,
                alpha=0.85,
                zorder=4,
            )

    ax1.set_xlim(1, plot_horizon + 1)
    ax1.set_ylim(0.0, ymax_bin)
    ax1.grid(axis="y", alpha=0.18)

    top_ticks = [0.0, 0.05, 0.10]
    top_ticks = [tick for tick in top_ticks if tick <= ymax_bin + 1e-12]

    ax1.set_yticks(top_ticks)
    ax1.set_yticklabels(
        [r"$0$" if tick == 0.0 else rf"${tick:.2f}$" for tick in top_ticks],
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
        loc="upper right",
        fontsize=LEGEND_FONTSIZE,
        ncol=2,
        frameon=True,
        handlelength=2.2,
        columnspacing=1.0,
        borderpad=0.25,
    )

    # ========================================================
    # Bottom panel: exact discrete-time CDF
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
    # Top panel: bin tick labels only.
    # Bottom panel: standard time ticks and sole x-label.
    # ========================================================
    centers = 0.5 * (lefts + rights - 1)
    bin_tick_labels = [
        rf"${left}\!-\!{right - 1}$" if (right - left) > 1 else rf"${left}$"
        for left, right in zip(lefts, rights)
    ]

    ax1.set_xticks(centers[::XTICK_EVERY])
    ax1.set_xticklabels(
        bin_tick_labels[::XTICK_EVERY],
        rotation=30,
        ha="right",
        fontsize=TICK_FONTSIZE,
    )
    ax1.tick_params(axis="x", length=2.5, width=0.7, pad=1)
    ax1.set_xlabel("")

    standard_ticks = np.arange(0, plot_horizon + 1, 20)
    standard_ticks[0] = 1

    ax2.set_xticks(standard_ticks)
    ax2.set_xticklabels(
        [rf"${int(t)}$" for t in standard_ticks],
        fontsize=TICK_FONTSIZE,
    )
    ax2.tick_params(axis="x", length=2.5, width=0.7, pad=1)
    ax2.set_xlabel(r"rejection time", fontsize=AXIS_LABEL_FONTSIZE)

    fig.tight_layout(h_pad=0.45)
    return fig



# ============================================================
# Saving
# ============================================================

def save_results(result: dict) -> None:
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
                    "mu": MU,
                    "sigma": SIGMA,
                    "alpha": ALPHA,
                    "u0": U0,
                    "T_values": T_VALUES,
                    "horizon": HORIZON,
                    "plot_max_t": PLOT_MAX_T,
                    "n_sim_alt": N_SIM_ALT,
                    "n_sim_null": N_SIM_NULL,
                    "batch_size": BATCH_SIZE,
                    "seed": SEED,
                    "capping": False,
                    "common_random_numbers": True,
                    "n_bins_total": N_BINS_TOTAL,
                    "cdf_line_styles": {
                        "GRO": CDF_LS_GRO,
                        **{
                            f"EDO T={fmt_t_value(t)}": CDF_LS_BY_T[float(t)]
                            for t in T_VALUES
                        },
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

    fig_stacked = plot_stacked_pmf_cdf(
        curves=result["curves"],
        plot_max_t=PLOT_MAX_T,
        n_bins=N_BINS_TOTAL,
        ymax_bin=None,
    )

    fig_stacked.savefig(OUT_STACKED_PNG, dpi=DPI)
    fig_stacked.savefig(OUT_STACKED_PDF)

    save_results(result)

    print("\nSaved:")
    print(f"  {OUT_STACKED_PNG}")
    print(f"  {OUT_STACKED_PDF}")
    print(f"  {OUT_CSV}")
    print(f"  {OUT_JSON}")

    plt.show()


if __name__ == "__main__":
    main()