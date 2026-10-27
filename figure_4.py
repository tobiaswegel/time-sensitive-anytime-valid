"""Figure 4: Gaussian GRO and EDO with time scales 8, 14, and 60."""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plotting import plot_rejection_times
from utils import discounted_value_from_tau, save_figure, tau_to_pmf_cdf

# Configuration


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

DPI = 450


N_BINS_TOTAL = 15

OUT_DIR = Path(".")
OUT_FIG_PNG = OUT_DIR / "figure_4.png"
OUT_CSV = OUT_DIR / "figure_4.csv"
OUT_JSON = OUT_DIR / "figure_4.json"

# Plot style


def configure_fonts() -> None:
    """Set the fonts and sizes used in the paper plot."""
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

CDF_LS_GRO = "-"
CDF_LS_BY_T = {
    8.0: "--",
    14.0: "-.",
    60.0: ":",
}


# Gaussian betting helpers


def gaussian_kl(mu: float, sigma: float) -> float:
    """Return KL(N(mu, sigma squared) || N(0, sigma squared))."""
    return mu * mu / (2.0 * sigma * sigma)


def gaussian_log_e(action_a: float, x: np.ndarray, sigma: float) -> np.ndarray:
    """Return log payoffs for a Gaussian-shift bet."""
    return action_a * x / (sigma * sigma) - (action_a * action_a) / (2.0 * sigma * sigma)


def gaussian_gro_action(mu: float) -> float:
    """Return the Gaussian growth-rate-optimal shift."""
    return float(mu)


def gaussian_approx_action(mu: float, sigma: float, time_scale: float) -> float:
    """Return the constant Gaussian EDO shift for the given time scale."""
    if time_scale <= 0.0:
        raise ValueError("time_scale must be positive.")
    if abs(mu) <= 1e-15:
        raise ValueError("mu must be nonzero.")
    return float(mu + 2.0 * sigma * sigma / (mu * time_scale))


def gaussian_eta_star(mu: float, sigma: float, time_scale: float) -> float:
    """Return the Gaussian EDO exponent in closed form."""
    return float(2.0 * sigma * sigma / (mu * mu * time_scale + 2.0 * sigma * sigma))


def fmt_t_value(time_scale: float) -> int | float:
    """Display an integer time scale without a decimal point."""
    return int(time_scale) if float(time_scale).is_integer() else float(time_scale)


# Simulation helpers


def first_hit_times_from_paths(
    x_paths: np.ndarray,
    sigma: float,
    action_a: float,
    horizon: int,
    u0: float,
) -> np.ndarray:
    """Find the first boundary crossing on each Gaussian observation path."""
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
    """Compare strategies on the same simulated observation paths."""
    rng = np.random.default_rng(seed)

    tau_by_policy = {policy["name"]: np.empty(n_sim, dtype=np.int32) for policy in policies}

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


# Experiment


def validate_config() -> None:
    """Check that the experiment settings are valid."""
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
        raise ValueError(f"Missing CDF line styles for T values: {sorted(missing_linestyles)}")


def make_policies() -> list[dict]:
    """Build the GRO and EDO strategies for the configured time scales."""
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
    """Compute the strategies, rejection curves, and summary table."""
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


# Plotting helpers


def curve_color(curve: dict) -> str:
    """Return the plot color assigned to a strategy."""
    if curve["kind"] == "gro":
        return COL_GRO
    return COL_BY_T[float(curve["T"])]


def curve_cdf_linestyle(curve: dict) -> str:
    """Return the line style assigned to a strategy."""
    if curve["kind"] == "gro":
        return CDF_LS_GRO
    return CDF_LS_BY_T[float(curve["T"])]


def ordered_curves_for_plot(curves: list[dict]) -> list[dict]:
    """Order strategies consistently for the legend and plots."""
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


def plot_stacked_pmf_cdf(curves, plot_max_t, n_bins, ymax_bin=None):
    """Draw this experiment’s rejection-time distributions in the shared layout."""
    curves = ordered_curves_for_plot(curves)
    colors = [curve_color(c) for c in curves]
    linestyles = [curve_cdf_linestyle(c) for c in curves]
    return plot_rejection_times(
        curves,
        colors,
        linestyles,
        min(plot_max_t, HORIZON),
        n_bins,
        style="gaussian",
        ymax_bin=ymax_bin,
    )


# Saving


def save_results(result: dict) -> None:
    """Save the parameters and numerical results."""
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
                        **{f"EDO T={fmt_t_value(t)}": CDF_LS_BY_T[float(t)] for t in T_VALUES},
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

    fig_stacked = plot_stacked_pmf_cdf(
        curves=result["curves"],
        plot_max_t=PLOT_MAX_T,
        n_bins=N_BINS_TOTAL,
        ymax_bin=None,
    )

    save_figure(fig_stacked, OUT_FIG_PNG, dpi=DPI)

    save_results(result)

    print("\nSaved:")
    print(f"  {OUT_FIG_PNG}")
    print(f"  {OUT_CSV}")
    print(f"  {OUT_JSON}")

    plt.show()


if __name__ == "__main__":
    main()
