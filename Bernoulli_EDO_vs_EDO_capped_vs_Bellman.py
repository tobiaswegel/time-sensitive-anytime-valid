"""
Bernoulli EDO vs EDO capped vs Bellman.

Model:
    pi_0 = Bernoulli(1/2)
    pi_1 = Bernoulli(2/3)

Reward:
    R(t) = exp(-t / T), with T = 30.

Strategies:
    EDO:
        constant exponential-decay approximation q*(T)

    EDO capped:
        same q*(T), but with action-level overshoot capping

    Bellman:
        finite-horizon backward induction over rescaled wealth u = alpha W,
        with action-level overshoot capping

Output:
    - stacked PMF-bin/CDF figure
    - summary CSV
    - curves JSON

Uses common random numbers for cleaner curve comparisons.
"""

from __future__ import annotations

import json
import math
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

P0 = 1.0 / 2.0
P1 = 2.0 / 3.0

ALPHA = 0.05
U0 = ALPHA

TIME_SCALE = 30.0

HORIZON = 180
PLOT_MAX_T = 120

N_WEALTH = 401
N_ACTIONS = 401
MIN_U = 1e-10

N_SIM_ALT = 300_000
N_SIM_NULL = 100_000
BATCH_SIZE = 50_000
SEED = 0

FIG_WIDTH = 6.2
FIG_HEIGHT = 4.35
DPI = 450

N_BINS_TOTAL = 15
XTICK_EVERY = 2

OUT_DIR = Path(".")
OUT_FIG_PNG = OUT_DIR / "bernoulli_edo_bellman_stacked_colorblind.png"
OUT_FIG_PDF = OUT_DIR / "bernoulli_edo_bellman_stacked_colorblind.pdf"
OUT_CSV = OUT_DIR / "bernoulli_edo_bellman_colorblind_summary.csv"
OUT_JSON = OUT_DIR / "bernoulli_edo_bellman_colorblind_curves.json"


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


COL_BELLMAN = "tab:orange"
COL_EDO = "tab:blue"
COL_EDO_CAP = "tab:green"

CDF_LS_BELLMAN = "-"
CDF_LS_EDO = "--"
CDF_LS_EDO_CAP = ":"

# Top panel: grouped histogram
BAR_FACE_BLEND = 0.82
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

LEGEND_FONTSIZE = 8.2
TICK_FONTSIZE = 8.5
AXIS_LABEL_FONTSIZE = 10.2


# ============================================================
# Bernoulli math helpers
# ============================================================

def check_probability(p: float, name: str) -> None:
    if not (0.0 < float(p) < 1.0):
        raise ValueError(f"{name} must lie in (0, 1).")


def bernoulli_kl(p: float, q: float) -> float:
    check_probability(p, "p")
    check_probability(q, "q")
    return p * math.log(p / q) + (1.0 - p) * math.log((1.0 - p) / (1.0 - q))


def logit(p: float) -> float:
    check_probability(p, "p")
    return math.log(p / (1.0 - p))


def expit(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def eta_renyi_product(eta: float, p0: float, p1: float) -> float:
    """
    Computes eta * D_{1/(1-eta)}(Bern(p1) || Bern(p0)).

    Stable form:
        (1 - eta) log E_{p0}[(d p1 / d p0)^{1/(1-eta)}].
    """
    if eta < 0.0 or eta >= 1.0:
        raise ValueError("eta must lie in [0, 1).")
    if eta == 0.0:
        return 0.0

    r = 1.0 / (1.0 - eta)
    log_l1 = math.log(p1 / p0)
    log_l0 = math.log((1.0 - p1) / (1.0 - p0))

    a = math.log(p0) + r * log_l1
    b = math.log(1.0 - p0) + r * log_l0

    m = max(a, b)
    log_sum = m + math.log(math.exp(a - m) + math.exp(b - m))
    return (1.0 - eta) * log_sum


def solve_eta_star(
    p0: float,
    p1: float,
    time_scale: float,
    tol: float = 1e-12,
    max_iter: int = 250,
) -> float:
    """Solve eta * D_{1/(1-eta)}(Bern(p1) || Bern(p0)) = 1 / T."""
    if time_scale <= 0.0:
        raise ValueError("time_scale must be positive.")

    target = 1.0 / time_scale
    endpoint = math.log(max(p1 / p0, (1.0 - p1) / (1.0 - p0)))

    if target >= endpoint:
        raise ValueError(
            f"No finite eta_star exists: 1/T={target:.6g}, endpoint={endpoint:.6g}."
        )

    lo = 0.0
    hi = 1.0 - 1e-14

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        value = eta_renyi_product(mid, p0, p1)

        if value < target:
            lo = mid
        else:
            hi = mid

        if hi - lo <= tol:
            break

    return 0.5 * (lo + hi)


def bernoulli_constant_approximation(p0: float, p1: float, time_scale: float) -> dict[str, float]:
    """Constant exponential-reward approximation: the L_eta tilt of p0."""
    eta_star = solve_eta_star(p0, p1, time_scale)
    q_star_logit = (logit(p1) - eta_star * logit(p0)) / (1.0 - eta_star)
    q_star = expit(q_star_logit)

    return {
        "T": float(time_scale),
        "eta_star": float(eta_star),
        "q_star": float(q_star),
        "rho": float(math.exp(-1.0 / time_scale)),
    }


def bernoulli_log_e(q: np.ndarray | float, x: np.ndarray, p0: float) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    x = np.asarray(x, dtype=int)
    return np.where(x == 1, np.log(q / p0), np.log((1.0 - q) / (1.0 - p0)))


def cap_bernoulli_action(
    q: np.ndarray | float,
    u: np.ndarray,
    p0: float,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Action-level overshoot cap.

    If q > p0, cap q downward so that u * q/p0 <= 1.
    If q < p0, cap q upward so that u * (1-q)/(1-p0) <= 1.
    """
    q_arr = np.asarray(q, dtype=float)
    u_arr = np.asarray(u, dtype=float)

    if np.any(u_arr <= 0.0) or np.any(u_arr > 1.0):
        raise ValueError("u must lie in (0, 1].")

    capped = np.array(q_arr + np.zeros_like(u_arr), dtype=float, copy=True)

    upper_for_one = p0 / u_arr
    lower_for_zero = 1.0 - (1.0 - p0) / u_arr

    high = capped > p0
    low = capped < p0

    capped = np.where(high, np.minimum(capped, upper_for_one), capped)
    capped = np.where(low, np.maximum(capped, lower_for_zero), capped)

    return np.clip(capped, eps, 1.0 - eps)


# ============================================================
# Bellman solver
# ============================================================

def make_wealth_grid(n_wealth: int, min_u: float) -> np.ndarray:
    if n_wealth < 2:
        raise ValueError("n_wealth must be at least 2.")
    if not (0.0 < min_u < 1.0):
        raise ValueError("min_u must lie in (0, 1).")

    grid = np.exp(np.linspace(math.log(min_u), 0.0, n_wealth))
    grid[-1] = 1.0
    return grid


def precompute_action_matrices(
    p0: float,
    u_grid: np.ndarray,
    n_actions: int,
    action_cap: bool = True,
) -> dict[str, np.ndarray]:
    q_raw = np.linspace(1e-12, 1.0 - 1e-12, n_actions)
    u_states = u_grid[:-1]

    if action_cap:
        q_eff = cap_bernoulli_action(q_raw[:, None], u_states[None, :], p0)
    else:
        q_eff = np.broadcast_to(q_raw[:, None], (n_actions, u_states.size)).copy()

    e1 = q_eff / p0
    e0 = (1.0 - q_eff) / (1.0 - p0)

    u1 = u_states[None, :] * e1
    u0_next = u_states[None, :] * e0

    hit1 = u1 >= 1.0 - 1e-14
    hit0 = u0_next >= 1.0 - 1e-14

    return {"q_eff": q_eff, "u1": u1, "u0": u0_next, "hit1": hit1, "hit0": hit0}


def solve_bernoulli_bellman(
    p0: float,
    p1: float,
    time_scale: float,
    horizon: int,
    n_wealth: int,
    min_u: float,
    n_actions: int,
    u0: float,
    action_cap: bool = True,
) -> dict[str, Any]:
    """
    Finite-horizon backward induction for reward exp(-tau/T).

    State:
        u = alpha * W in (0, 1].

    Boundary:
        u = 1 means rejection has already happened.
    """
    u_grid = make_wealth_grid(n_wealth, min_u)

    mats = precompute_action_matrices(
        p0=p0,
        u_grid=u_grid,
        n_actions=n_actions,
        action_cap=action_cap,
    )

    q_eff = mats["q_eff"]
    u1 = mats["u1"]
    u0_next = mats["u0"]
    hit1 = mats["hit1"]
    hit0 = mats["hit0"]

    n_actions_eff, n_states = q_eff.shape
    state_cols = np.arange(n_states)

    values = np.zeros((horizon + 1, n_wealth), dtype=float)
    policy_q = np.full((horizon, n_wealth), p1, dtype=float)

    rewards = np.exp(-np.arange(horizon + 1, dtype=float) / float(time_scale))
    values[:, -1] = rewards

    u1_flat = u1.ravel()
    u0_flat = u0_next.ravel()

    for t in range(horizon - 1, -1, -1):
        v_next = values[t + 1]

        cont1 = np.interp(u1_flat, u_grid, v_next).reshape(n_actions_eff, n_states)
        cont0 = np.interp(u0_flat, u_grid, v_next).reshape(n_actions_eff, n_states)

        val1 = np.where(hit1, rewards[t + 1], cont1)
        val0 = np.where(hit0, rewards[t + 1], cont0)

        expected = p1 * val1 + (1.0 - p1) * val0
        best_idx = np.argmax(expected, axis=0)

        values[t, :-1] = expected[best_idx, state_cols]
        policy_q[t, :-1] = q_eff[best_idx, state_cols]
        values[t, -1] = rewards[t]
        policy_q[t, -1] = policy_q[t, -2]

    return {
        "p0": float(p0),
        "p1": float(p1),
        "T": float(time_scale),
        "horizon": int(horizon),
        "u_grid": u_grid,
        "values": values,
        "policy_q": policy_q,
        "action_cap": bool(action_cap),
        "value_t0": float(np.interp(u0, u_grid, values[0])),
    }


# ============================================================
# Simulation helpers
# ============================================================

def tau_to_pmf_cdf(tau: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    pmf = np.bincount(tau, minlength=horizon + 2)[1 : horizon + 1] / tau.size
    return pmf, np.cumsum(pmf)


def discounted_value_from_tau(tau: np.ndarray, time_scale: float, horizon: int) -> float:
    hit = tau <= horizon
    return float(np.mean(np.where(hit, np.exp(-tau / float(time_scale)), 0.0)))


def simulate_constant_from_paths(
    x_paths: np.ndarray,
    p0: float,
    q: float,
    horizon: int,
    u0: float,
    action_cap: bool,
) -> np.ndarray:
    n_sim = x_paths.shape[0]

    log_u = np.full(n_sim, math.log(u0), dtype=float)
    tau = np.full(n_sim, horizon + 1, dtype=np.int32)
    alive = np.ones(n_sim, dtype=bool)

    for t in range(horizon):
        idx = np.flatnonzero(alive)
        if idx.size == 0:
            break

        u = np.exp(np.clip(log_u[idx], -745.0, 0.0))

        if action_cap:
            q_eff = cap_bernoulli_action(q, u, p0)
        else:
            q_eff = np.full(idx.size, float(q), dtype=float)

        x = x_paths[idx, t]
        next_log_u = log_u[idx] + bernoulli_log_e(q_eff, x, p0)
        hit = next_log_u >= -1e-14

        tau[idx[hit]] = t + 1
        alive[idx[hit]] = False

        next_log_u[hit] = 0.0
        log_u[idx] = np.minimum(next_log_u, 0.0)

    return tau


def simulate_bellman_from_paths(
    x_paths: np.ndarray,
    bellman_result: dict[str, Any],
    horizon: int,
    u0: float,
    action_cap: bool = True,
) -> np.ndarray:
    p0 = bellman_result["p0"]
    u_grid = bellman_result["u_grid"]
    policy_q = bellman_result["policy_q"]

    n_sim = x_paths.shape[0]

    log_u = np.full(n_sim, math.log(u0), dtype=float)
    tau = np.full(n_sim, horizon + 1, dtype=np.int32)
    alive = np.ones(n_sim, dtype=bool)

    for t in range(horizon):
        idx = np.flatnonzero(alive)
        if idx.size == 0:
            break

        u = np.exp(np.clip(log_u[idx], -745.0, 0.0))
        q = np.interp(u, u_grid, policy_q[t])

        if action_cap:
            q = cap_bernoulli_action(q, u, p0)

        x = x_paths[idx, t]
        next_log_u = log_u[idx] + bernoulli_log_e(q, x, p0)
        hit = next_log_u >= -1e-14

        tau[idx[hit]] = t + 1
        alive[idx[hit]] = False

        next_log_u[hit] = 0.0
        log_u[idx] = np.minimum(next_log_u, 0.0)

    return tau


def simulate_policies_common_random_numbers(
    p_true: float,
    p0: float,
    q_edo: float,
    bellman_result: dict[str, Any],
    horizon: int,
    n_sim: int,
    batch_size: int,
    seed: int,
    u0: float,
) -> dict[str, dict[str, Any]]:
    """Simulate all policies on common Bernoulli paths."""
    rng = np.random.default_rng(seed)

    policy_names = ["EDO", "EDO capped", "Bellman"]
    tau_by_policy = {name: np.empty(n_sim, dtype=np.int32) for name in policy_names}

    for start in range(0, n_sim, batch_size):
        end = min(start + batch_size, n_sim)
        x_paths = rng.binomial(1, p_true, size=(end - start, horizon)).astype(np.int8)

        tau_by_policy["EDO"][start:end] = simulate_constant_from_paths(
            x_paths=x_paths,
            p0=p0,
            q=q_edo,
            horizon=horizon,
            u0=u0,
            action_cap=False,
        )

        tau_by_policy["EDO capped"][start:end] = simulate_constant_from_paths(
            x_paths=x_paths,
            p0=p0,
            q=q_edo,
            horizon=horizon,
            u0=u0,
            action_cap=True,
        )

        tau_by_policy["Bellman"][start:end] = simulate_bellman_from_paths(
            x_paths=x_paths,
            bellman_result=bellman_result,
            horizon=horizon,
            u0=u0,
            action_cap=True,
        )

    results = {}
    for name in policy_names:
        tau = tau_by_policy[name]
        pmf, cdf = tau_to_pmf_cdf(tau, horizon)
        results[name] = {"tau": tau, "pmf": pmf, "cdf": cdf, "n_sim": int(n_sim)}

    return results


# ============================================================
# Experiment
# ============================================================

def validate_config() -> None:
    check_probability(P0, "P0")
    check_probability(P1, "P1")

    if abs(P1 - P0) <= 1e-15:
        raise ValueError("P0 and P1 must be different.")
    if not (0.0 < ALPHA < 1.0):
        raise ValueError("ALPHA must lie in (0, 1).")
    if not (0.0 < U0 < 1.0):
        raise ValueError("U0 must lie in (0, 1).")
    if TIME_SCALE <= 0.0:
        raise ValueError("TIME_SCALE must be positive.")
    if HORIZON < 1:
        raise ValueError("HORIZON must be at least 1.")
    if PLOT_MAX_T > HORIZON:
        raise ValueError("PLOT_MAX_T cannot exceed HORIZON.")
    if N_WEALTH < 2:
        raise ValueError("N_WEALTH must be at least 2.")
    if N_ACTIONS < 2:
        raise ValueError("N_ACTIONS must be at least 2.")


def make_curve(
    name: str,
    kind: str,
    label: str,
    tau_result: dict[str, Any],
    q: float | None,
    eta_star: float | None,
    bellman_value_t0: float | None = None,
) -> dict[str, Any]:
    curve = {
        "name": name,
        "kind": kind,
        "label": label,
        "T": float(TIME_SCALE),
        "q": None if q is None else float(q),
        "eta_star": None if eta_star is None else float(eta_star),
        "pmf": tau_result["pmf"],
        "cdf": tau_result["cdf"],
        "n_sim": tau_result["n_sim"],
    }

    if bellman_value_t0 is not None:
        curve["value_t0"] = float(bellman_value_t0)

    return curve


def run_experiment() -> dict[str, Any]:
    validate_config()

    kl_10 = bernoulli_kl(P1, P0)
    approx = bernoulli_constant_approximation(P0, P1, TIME_SCALE)

    print(f"pi_0 = Bern({P0:g})")
    print(f"pi_1 = Bern({P1:g})")
    print(f"alpha = {ALPHA}")
    print(f"T = {TIME_SCALE}")
    print(f"horizon = {HORIZON}")
    print(f"plot max t = {PLOT_MAX_T}")
    print(f"KL(pi_1 || pi_0) = {kl_10:.8g}")
    print(f"1 / KL(pi_1 || pi_0) = {1.0 / kl_10:.8g}")
    print(f"eta_star = {approx['eta_star']:.8g}")
    print(f"q_star = {approx['q_star']:.8g}")
    print("Bellman capping = True")
    print("EDO capping = False")
    print("EDO capped capping = True")
    print("common random numbers = True")
    print()

    print("Solving Bellman...")
    bellman_result = solve_bernoulli_bellman(
        p0=P0,
        p1=P1,
        time_scale=TIME_SCALE,
        horizon=HORIZON,
        n_wealth=N_WEALTH,
        min_u=MIN_U,
        n_actions=N_ACTIONS,
        u0=U0,
        action_cap=True,
    )

    print("Simulating under pi_1...")
    alt_results = simulate_policies_common_random_numbers(
        p_true=P1,
        p0=P0,
        q_edo=approx["q_star"],
        bellman_result=bellman_result,
        horizon=HORIZON,
        n_sim=N_SIM_ALT,
        batch_size=BATCH_SIZE,
        seed=SEED + 123,
        u0=U0,
    )

    print("Simulating under pi_0...")
    null_results = simulate_policies_common_random_numbers(
        p_true=P0,
        p0=P0,
        q_edo=approx["q_star"],
        bellman_result=bellman_result,
        horizon=HORIZON,
        n_sim=N_SIM_NULL,
        batch_size=BATCH_SIZE,
        seed=SEED + 456,
        u0=U0,
    )

    curves = [
        make_curve(
            name="Bellman",
            kind="bellman",
            label="Bellman",
            tau_result=alt_results["Bellman"],
            q=None,
            eta_star=None,
            bellman_value_t0=bellman_result["value_t0"],
        ),
        make_curve(
            name="EDO",
            kind="edo",
            label="EDO",
            tau_result=alt_results["EDO"],
            q=approx["q_star"],
            eta_star=approx["eta_star"],
        ),
        make_curve(
            name="EDO capped",
            kind="edo_capped",
            label="EDO capped",
            tau_result=alt_results["EDO capped"],
            q=approx["q_star"],
            eta_star=approx["eta_star"],
        ),
    ]

    rows = []
    for curve in curves:
        name = curve["name"]
        alt = alt_results[name]
        null = null_results[name]

        rows.append(
            {
                "policy": curve["label"],
                "T": TIME_SCALE,
                "eta_star": curve["eta_star"],
                "q": curve["q"],
                "bellman_value_t0": curve.get("value_t0"),
                f"hit_by_{PLOT_MAX_T}_under_pi1": float(alt["cdf"][PLOT_MAX_T - 1]),
                "hit_by_H_under_pi1": float(alt["cdf"][-1]),
                "hit_by_H_under_pi0": float(null["cdf"][-1]),
                f"exp_reward_T{int(TIME_SCALE)}": discounted_value_from_tau(
                    alt["tau"],
                    TIME_SCALE,
                    HORIZON,
                ),
                "mean_tau_truncated_under_pi1": float(
                    np.mean(np.minimum(alt["tau"], HORIZON + 1))
                ),
            }
        )

    return {
        "approximation": approx,
        "bellman": bellman_result,
        "curves": curves,
        "summary": pd.DataFrame(rows),
        "kl_10": kl_10,
        "inv_kl": 1.0 / kl_10,
    }


# ============================================================
# Plotting
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


def tick_label_decimal(value: float) -> str:
    """Format tick labels as 0, 0.1, 0.2, etc., without trailing zeros."""
    if abs(value) < 1e-12:
        return r"$0$"
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return rf"${text}$"


def curve_color(curve: dict[str, Any]) -> str:
    if curve["kind"] == "bellman":
        return COL_BELLMAN
    if curve["kind"] == "edo":
        return COL_EDO
    if curve["kind"] == "edo_capped":
        return COL_EDO_CAP
    raise ValueError(f"Unknown curve kind: {curve['kind']}")


def curve_cdf_linestyle(curve: dict[str, Any]) -> str:
    if curve["kind"] == "bellman":
        return CDF_LS_BELLMAN
    if curve["kind"] == "edo":
        return CDF_LS_EDO
    if curve["kind"] == "edo_capped":
        return CDF_LS_EDO_CAP
    raise ValueError(f"Unknown curve kind: {curve['kind']}")


def ordered_curves_for_plot(curves: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = ["bellman", "edo", "edo_capped"]
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

    pmfs = [curve["pmf"][:plot_horizon] for curve in ordered_curves]
    cdfs = [curve["cdf"][:plot_horizon] for curve in ordered_curves]
    labels = [curve["label"] for curve in ordered_curves]
    colors = [curve_color(curve) for curve in ordered_curves]
    cdf_linestyles = [curve_cdf_linestyle(curve) for curve in ordered_curves]

    pmf_bins = [aggregate_pmf_on_bins(pmf, lefts, rights) for pmf in pmfs]

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

            # Central vertical cue matching the CDF linestyle. This makes the
            # histogram readable in grayscale and for color-vision deficiencies.
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
    ax1.set_yticklabels([tick_label_decimal(tick) for tick in top_ticks], fontsize=TICK_FONTSIZE)
    ax1.tick_params(axis="y", length=2.5, width=0.7, pad=1)

    ax1.set_ylabel(
        r"$\pi_1^{\otimes}(\tau \in \mathrm{bin})$",
        fontsize=AXIS_LABEL_FONTSIZE,
    )

    # Legend uses the same line styles as the CDF panel, so it remains
    # interpretable in grayscale.
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
    # Top panel uses bin/channel tick labels only.
    # Bottom panel uses standard time ticks and the sole x-label.
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
                    "u0": U0,
                    "T": TIME_SCALE,
                    "horizon": HORIZON,
                    "plot_max_t": PLOT_MAX_T,
                    "n_wealth": N_WEALTH,
                    "n_actions": N_ACTIONS,
                    "min_u": MIN_U,
                    "n_sim_alt": N_SIM_ALT,
                    "n_sim_null": N_SIM_NULL,
                    "batch_size": BATCH_SIZE,
                    "seed": SEED,
                    "bellman_capping": True,
                    "edo_capping": False,
                    "edo_capped_capping": True,
                    "common_random_numbers": True,
                    "colorblind_friendly_style": True,
                    "data_alpha_transparency": False,
                    "n_bins_total": N_BINS_TOTAL,
                    "cdf_line_styles": {
                        "Bellman": CDF_LS_BELLMAN,
                        "EDO": CDF_LS_EDO,
                        "EDO capped": CDF_LS_EDO_CAP,
                    },
                    "kl_pi1_pi0": result["kl_10"],
                    "inv_kl_pi1_pi0": result["inv_kl"],
                    "eta_star": result["approximation"]["eta_star"],
                    "q_star": result["approximation"]["q_star"],
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
    fig.savefig(OUT_FIG_PDF)
    save_results(result)

    print("\nSaved:")
    print(f"  {OUT_FIG_PNG}")
    print(f"  {OUT_FIG_PDF}")
    print(f"  {OUT_CSV}")
    print(f"  {OUT_JSON}")


if __name__ == "__main__":
    main()
