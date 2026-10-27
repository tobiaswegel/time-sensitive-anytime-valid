"""Figure 1: Gaussian Bellman policies for hard, logistic, and exponential rewards."""

import argparse
import json
import math
import os

import numpy as np

from utils import save_figure


def build_lambda_grid(mu0, mu1, sigma, n_actions, lambda_max, two_sided):
    """Create the Gaussian betting-action grid in the alternative’s direction."""
    if sigma <= 0.0:
        raise ValueError("sigma must be positive.")
    if n_actions < 2:
        raise ValueError("n_actions must be at least 2.")
    if lambda_max is None:
        lambda_max = 4.0 / sigma
    if lambda_max <= 0.0:
        raise ValueError("lambda_max must be positive.")

    if two_sided:
        return np.linspace(-lambda_max, lambda_max, n_actions)
    direction = 1.0 if mu1 >= mu0 else -1.0
    return direction * np.linspace(0.0, lambda_max, n_actions)


def gaussian_e_multiplier(x, mu0, sigma, lambdas):
    """Return Gaussian bet payoffs for every action and quadrature node."""
    centered = x[None, :] - mu0
    lam = lambdas[:, None]
    log_f = lam * centered - 0.5 * (sigma * lam) ** 2
    return np.exp(np.clip(log_f, -745.0, 700.0))


def gaussian_alt_quadrature(mu1, sigma, n_quad):
    """Return Gaussian quadrature nodes and probabilities under the alternative."""
    if n_quad < 3:
        raise ValueError("n_quad must be at least 3.")
    nodes, weights = np.polynomial.hermite.hermgauss(n_quad)
    x = mu1 + math.sqrt(2.0) * sigma * nodes
    prob = weights / math.sqrt(math.pi)
    return x, prob / prob.sum()


def precompute_interpolation(u_grid, multipliers):
    """Locate next-wealth values on the wealth grid for fast interpolation."""
    points = np.minimum(multipliers[:, :, None] * u_grid[None, None, :], 1.0)
    idx = np.searchsorted(u_grid, points, side="right") - 1
    idx = np.clip(idx, 0, len(u_grid) - 2)
    left = u_grid[idx]
    right = u_grid[idx + 1]
    weight = np.where(right > left, (points - left) / (right - left), 0.0)
    return points, idx, weight


def interpolate_with_precompute(values, idx, weight):
    """Interpolate values using precomputed grid indices and weights."""
    left = values[idx]
    right = values[idx + 1]
    return left + weight * (right - left)


def wealth_edges_from_log_grid(log_u):
    """Find cell edges around the plotted log-wealth grid points."""
    if log_u.size == 1:
        return np.array([log_u[0] - 0.5, log_u[0] + 0.5], dtype=float)
    mids = 0.5 * (log_u[:-1] + log_u[1:])
    first = log_u[0] - 0.5 * (log_u[1] - log_u[0])
    last = log_u[-1] + 0.5 * (log_u[-1] - log_u[-2])
    return np.concatenate(([first], mids, [last]))


def build_wealth_grid(n_wealth, grid_type, min_u):
    """Create a linear or logarithmic wealth grid, including zero and one."""
    if n_wealth < 2:
        raise ValueError("n_wealth must be at least 2.")
    if grid_type == "linear":
        return np.linspace(0.0, 1.0, n_wealth)
    if grid_type == "log":
        if not (0.0 < min_u < 1.0):
            raise ValueError("min_u must lie in (0, 1) for the log wealth grid.")
        return np.concatenate(([0.0], np.geomspace(min_u, 1.0, n_wealth - 1)))
    raise ValueError("grid_type must be 'linear' or 'log'.")


def build_exponential_rewards(time_scale, horizon):
    """Return rewards exp(-t/time_scale) for times 1 through the horizon."""
    return np.exp(-np.arange(1, horizon + 1, dtype=float) / time_scale)


def build_indicator_rewards(time_scale, horizon):
    """Return reward one through the deadline and zero afterwards."""
    return (np.arange(1, horizon + 1, dtype=float) <= time_scale).astype(float)


def build_logistic_rewards(time_scale, beta, horizon):
    """Return rewards with a smooth logistic drop around the deadline."""
    if beta <= 0.0:
        raise ValueError("beta must be positive.")
    x = (np.arange(1, horizon + 1, dtype=float) - time_scale) / beta
    return 1.0 / (1.0 + np.exp(x))


def build_linear_rewards(horizon):
    """Return rewards decreasing linearly over the horizon."""
    return (horizon + 1.0 - np.arange(1, horizon + 1, dtype=float)) / horizon


def solve_optimal_policy_heatmap(
    mu0,
    mu1,
    sigma,
    rewards,
    horizon,
    n_wealth,
    n_actions,
    n_quad,
    lambda_max=None,
    two_sided=False,
    wealth_grid="linear",
    min_u=1e-6,
):
    """Solve the Gaussian Bellman recursion on wealth and action grids."""
    if sigma <= 0.0:
        raise ValueError("sigma must be positive.")
    if horizon < 1:
        raise ValueError("horizon must be at least 1.")
    if len(rewards) != horizon:
        raise ValueError("rewards must have length equal to horizon.")

    u_grid = build_wealth_grid(n_wealth, wealth_grid, min_u)
    lambda_values = build_lambda_grid(mu0, mu1, sigma, n_actions, lambda_max, two_sided)
    x_nodes, x_weights = gaussian_alt_quadrature(mu1, sigma, n_quad)
    multipliers = gaussian_e_multiplier(x_nodes, mu0, sigma, lambda_values)

    next_u, idx, weight = precompute_interpolation(u_grid, multipliers)
    cross = next_u >= 1.0 - 1e-15

    values = np.zeros((horizon + 1, n_wealth), dtype=float)
    policy_idx = np.zeros((horizon, n_wealth), dtype=int)

    for t in range(horizon - 1, -1, -1):
        continuation = interpolate_with_precompute(values[t + 1], idx, weight)
        reward_t = rewards[t]
        payoff = np.where(cross, reward_t, continuation)
        candidates = np.sum(x_weights[None, :, None] * payoff, axis=1)

        policy_idx[t] = np.argmax(candidates, axis=0)
        values[t] = np.max(candidates, axis=0)
        values[t, -1] = values[t, -2]
        policy_idx[t, -1] = policy_idx[t, -2]

    policy_lambda = lambda_values[policy_idx]
    return {
        "u_grid": u_grid,
        "lambda_values": lambda_values,
        "policy_idx": policy_idx,
        "policy_lambda": policy_lambda,
        "values": values,
        "rewards": np.array(rewards, dtype=float),
        "horizon": int(horizon),
        "mu0": float(mu0),
        "mu1": float(mu1),
        "sigma": float(sigma),
        "n_quad": int(n_quad),
    }


def policy_lambda_at_wealths(result, t, current_u):
    """Interpolate the saved policy at the current wealth values."""
    return np.interp(current_u, result["u_grid"], result["policy_lambda"][t])


def simulate_policy_cdf(result, n_sim, seed, u0, cdf_mode):
    """Estimate rejection probabilities using normal or quadrature observations."""
    horizon = result["policy_lambda"].shape[0]

    rng = np.random.default_rng(seed)
    if not (0.0 < u0 < 1.0):
        raise ValueError("u0 must lie in (0, 1) for the CDF simulation.")
    if cdf_mode == "quadrature":
        x_nodes, x_weights = gaussian_alt_quadrature(
            result["mu1"],
            result["sigma"],
            result["n_quad"],
        )
        node_idx = np.arange(x_nodes.size)
    elif cdf_mode != "normal":
        raise ValueError("cdf_mode must be 'quadrature' or 'normal'.")

    u = np.full(n_sim, float(u0), dtype=float)
    tau = np.full(n_sim, horizon + 1, dtype=int)
    alive = np.ones(n_sim, dtype=bool)

    for t in range(horizon):
        if not np.any(alive):
            break
        idx = np.flatnonzero(alive)
        current_u = u[idx]
        lam = policy_lambda_at_wealths(result, t, current_u)
        if cdf_mode == "quadrature":
            x = x_nodes[rng.choice(node_idx, size=idx.size, p=x_weights)]
        else:
            x = rng.normal(result["mu1"], result["sigma"], size=idx.size)
        multiplier = np.exp(
            np.clip(
                lam * (x - result["mu0"]) - 0.5 * (result["sigma"] * lam) ** 2,
                -745.0,
                700.0,
            )
        )
        next_u = current_u * multiplier
        hit = next_u >= 1.0
        tau[idx[hit]] = t + 1
        u[idx[~hit]] = next_u[~hit]
        alive[idx[hit]] = False

    pmf = np.bincount(tau, minlength=horizon + 2)[1 : horizon + 1] / n_sim
    cdf = np.cumsum(pmf)
    return {
        "cdf": cdf,
        "pmf": pmf,
    }


def nearest_grid_indices(u_grid, values):
    """Find the closest wealth-grid point to each supplied value."""
    idx = np.searchsorted(u_grid, values, side="left")
    idx = np.clip(idx, 0, len(u_grid) - 1)
    left_idx = np.maximum(idx - 1, 0)
    choose_left = np.abs(values - u_grid[left_idx]) <= np.abs(values - u_grid[idx])
    return np.where(choose_left, left_idx, idx)


def reachable_policy_mask(result, u0):
    """Mark wealth states reachable under the discretized policy dynamics."""
    if not (0.0 < u0 < 1.0):
        raise ValueError("u0 must lie in (0, 1) for reachable-state plotting.")

    horizon, n_wealth = result["policy_lambda"].shape
    u_grid = result["u_grid"]
    mask = np.zeros((horizon, n_wealth - 1), dtype=bool)
    states = np.array([u0], dtype=float)
    x_nodes, _ = gaussian_alt_quadrature(result["mu1"], result["sigma"], result["n_quad"])

    for t in range(horizon):
        states = states[(states > 0.0) & (states < 1.0)]
        if states.size == 0:
            break

        idx = nearest_grid_indices(u_grid, states)
        positive = idx > 0
        mask[t, idx[positive] - 1] = True

        lam = policy_lambda_at_wealths(result, t, states)
        multipliers = gaussian_e_multiplier(x_nodes, result["mu0"], result["sigma"], lam)
        next_states = (states[:, None] * multipliers).ravel()
        next_states = next_states[(next_states > 0.0) & (next_states < 1.0)]
        if next_states.size == 0:
            states = next_states
            continue

        next_idx = np.unique(nearest_grid_indices(u_grid, next_states))
        next_idx = next_idx[(next_idx > 0) & (next_idx < n_wealth)]
        states = u_grid[next_idx]

    return mask


def save_results(results, args, output_path):
    """Save the parameters and numerical results."""

    def serialize_result(result):
        """Convert one policy’s arrays to JSON-compatible lists."""
        return {
            "u_grid": result["u_grid"].tolist(),
            "lambda_values": result["lambda_values"].tolist(),
            "policy_lambda": result["policy_lambda"].tolist(),
            "values": result["values"].tolist(),
            "value_t0": result["values"][0].tolist(),
            "rewards": result["rewards"].tolist(),
            "cdf": result["cdf"].tolist(),
            "mu0": result["mu0"],
            "mu1": result["mu1"],
            "sigma": result["sigma"],
            "n_quad": result["n_quad"],
        }

    payload = {
        "parameters": {
            "mu0": args.mu0,
            "mu1": args.mu1,
            "sigma": args.sigma,
            "time_scale": args.time_scale,
            "exp_time_scale": args.time_scale
            if args.exp_time_scale is None
            else args.exp_time_scale,
            "beta": args.beta,
            "horizon": args.horizon,
            "n_wealth": args.n_wealth,
            "wealth_grid": args.wealth_grid,
            "min_u": args.min_u,
            "n_actions": args.n_actions,
            "n_quad": args.n_quad,
            "lambda_max": args.lambda_max,
            "two_sided": args.two_sided,
            "cdf_mode": args.cdf_mode,
            "color_limits": args.color_limits,
            "color_quantile": args.color_quantile,
            "n_sim_cdf": args.n_sim_cdf,
            "seed": args.seed,
            "u0_cdf": args.u0_cdf,
            "include_linear": args.include_linear,
        },
        "indicator_result": serialize_result(results["indicator_result"]),
        "exp_result": serialize_result(results["exp_result"]),
        "logistic_result": serialize_result(results["logistic_result"]),
    }
    if "linear_result" in results:
        payload["linear_result"] = serialize_result(results["linear_result"])
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_results(input_path):
    """Load previously saved numerical results."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Results file not found: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    def deserialize_result(result):
        """Restore a saved policy’s lists as NumPy arrays."""
        return {
            "u_grid": np.array(result["u_grid"], dtype=float),
            "lambda_values": np.array(result["lambda_values"], dtype=float),
            "policy_lambda": np.array(result["policy_lambda"], dtype=float),
            "values": np.array(result["values"], dtype=float),
            "value_t0": np.array(result["value_t0"], dtype=float),
            "rewards": np.array(result["rewards"], dtype=float),
            "cdf": np.array(result["cdf"], dtype=float),
            "mu0": float(result["mu0"]),
            "mu1": float(result["mu1"]),
            "sigma": float(result["sigma"]),
            "n_quad": int(result["n_quad"]),
        }

    results = {
        "indicator_result": deserialize_result(payload["indicator_result"]),
        "exp_result": deserialize_result(payload["exp_result"]),
        "logistic_result": deserialize_result(payload["logistic_result"]),
        "parameters": payload["parameters"],
    }
    if "linear_result" in payload:
        results["linear_result"] = deserialize_result(payload["linear_result"])
    return results


def policy_quantity(result, quantity):
    """Return actions or bet payoffs for heatmap coloring."""
    if quantity == "lambda":
        return result["policy_lambda"]
    if quantity == "f_mu0":
        lam = result["policy_lambda"]
        return np.exp(-0.5 * (result["sigma"] * lam) ** 2)
    if quantity == "f_mu1":
        lam = result["policy_lambda"]
        return np.exp(lam * (result["mu1"] - result["mu0"]) - 0.5 * (result["sigma"] * lam) ** 2)
    raise ValueError("quantity must be 'lambda', 'f_mu0', or 'f_mu1'.")


def finite_heat_values(heat_panels):
    """Collect finite heatmap values after masking."""
    values = [panel["heat"][np.isfinite(panel["heat"])] for panel in heat_panels]
    values = [value for value in values if value.size > 0]
    if not values:
        raise ValueError("No finite heatmap cells remain after applying plot masks.")
    return np.concatenate(values)


def expanded_limits(vmin, vmax):
    """Widen constant color limits so the color scale remains valid."""
    if vmin < vmax:
        return float(vmin), float(vmax)
    delta = max(abs(vmin), 1.0) * 1e-6
    return float(vmin - delta), float(vmax + delta)


def color_limits_for_values(values, color_limits, color_quantile):
    """Choose color limits from the full range or robust percentiles."""
    if color_limits == "robust":
        if not (50.0 < color_quantile <= 100.0):
            raise ValueError("color_quantile must lie in (50, 100].")
        low_q = 100.0 - color_quantile
        return expanded_limits(
            np.nanpercentile(values, low_q),
            np.nanpercentile(values, color_quantile),
        )
    if color_limits in {"global", "panel"}:
        return expanded_limits(np.nanmin(values), np.nanmax(values))
    raise ValueError("color_limits must be 'global', 'robust', or 'panel'.")


def mesh_kwargs_for_values(values, bet_scale, color_limits, color_quantile):
    """Build linear, log, or symmetric-log color-scale settings."""
    from matplotlib.colors import LogNorm, SymLogNorm

    vmin, vmax = color_limits_for_values(values, color_limits, color_quantile)
    if bet_scale == "log":
        if not (vmin > 0.0 and vmax > 0.0):
            raise ValueError("--bet-scale log requires positive plotted values.")
        return {"norm": LogNorm(vmin=vmin, vmax=vmax)}
    if bet_scale == "symlog":
        finite_abs = np.abs(values[np.isfinite(values)])
        positive_abs = finite_abs[finite_abs > 0.0]
        linthresh = float(np.nanpercentile(positive_abs, 5.0)) if positive_abs.size else 1e-12
        return {"norm": SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax)}
    if bet_scale == "linear":
        return {"vmin": vmin, "vmax": vmax}
    raise ValueError("bet_scale must be 'linear', 'log', or 'symlog'.")


def maybe_plot(
    results,
    quantity,
    output_path,
    show_plot,
    max_t_plot,
    reachable_only,
    u0,
    bet_scale,
    color_limits,
    color_quantile,
):
    """Draw and save the figure, optionally displaying it."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-optimal-bet-heatmap")
    import matplotlib

    if not show_plot:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral"],
            "mathtext.fontset": "stix",
            "mathtext.rm": "STIXGeneral",
            "mathtext.it": "STIXGeneral:italic",
            "mathtext.bf": "STIXGeneral:bold",
        }
    )

    if quantity == "lambda":
        colorbar_label = r"optimal Gaussian shift $a$"
    elif quantity == "f_mu0":
        colorbar_label = r"optimal $f_t(\mu_0)$"
    else:
        colorbar_label = r"optimal $f_t(\mu_1)$"

    panel_results = [
        ("indicator_result", "hard deadline reward"),
        ("logistic_result", "logistic reward"),
        ("exp_result", "exponential reward"),
    ]
    if "linear_result" in results:
        panel_results.append(("linear_result", "linear reward"))

    panel_data = [
        (key, title, policy_quantity(results[key], quantity)) for key, title in panel_results
    ]
    horizon = panel_data[0][2].shape[0]
    if max_t_plot is None:
        plot_horizon = max(1, horizon // 4)
    else:
        plot_horizon = min(horizon, max(1, int(max_t_plot) + 1))
    u_grid = results["indicator_result"]["u_grid"]
    positive_u = np.maximum(u_grid[1:], 1e-300)
    log_u = np.log(positive_u)
    y_edges = wealth_edges_from_log_grid(log_u)

    heat_panels = []
    for key, title, data in panel_data:
        heat = data[:plot_horizon, 1:].astype(float, copy=True)
        zero_value = results[key]["values"][:plot_horizon, 1:] <= 0.0
        heat = np.where(zero_value, np.nan, heat)
        if reachable_only:
            mask = reachable_policy_mask(results[key], u0)[:plot_horizon]
            heat = np.where(mask, heat, np.nan)
        if bet_scale == "log":
            heat = np.where(heat > 0.0, heat, np.nan)
        if np.any(np.isfinite(heat)):
            heat_panels.append(
                {
                    "heat": heat,
                    "title": title,
                    "rewards": results[key]["rewards"][:plot_horizon],
                    "cdf": results[key]["cdf"][:plot_horizon],
                }
            )
    if not heat_panels:
        raise ValueError("No finite heatmap cells remain after applying plot masks.")
    shared_mesh_kwargs = None
    if color_limits != "panel":
        shared_mesh_kwargs = mesh_kwargs_for_values(
            finite_heat_values(heat_panels),
            bet_scale,
            color_limits,
            color_quantile,
        )
    t_values = np.arange(1, plot_horizon + 1, dtype=float)
    x_edges = np.arange(1, plot_horizon + 2, dtype=float) - 0.5

    n_cols = len(heat_panels)
    fig_width = max(8, 3 * n_cols)
    fig = plt.figure(figsize=(fig_width, 2.9), constrained_layout=True)
    grid = GridSpec(3, n_cols, figure=fig, height_ratios=[0.55, 2.0, 0.8])
    reward_axes = []
    heatmap_axes = []
    cdf_axes = []
    shared_x = None
    for i in range(n_cols):
        reward_ax = fig.add_subplot(grid[0, i], sharex=shared_x)
        if shared_x is None:
            shared_x = reward_ax
        heatmap_ax = fig.add_subplot(grid[1, i], sharex=shared_x)
        cdf_ax = fig.add_subplot(grid[2, i], sharex=shared_x)
        reward_axes.append(reward_ax)
        heatmap_axes.append(heatmap_ax)
        cdf_axes.append(cdf_ax)
    image = None
    panel_images = []
    for col, panel in enumerate(heat_panels):
        ax = heatmap_axes[col]
        reward_ax = reward_axes[col]
        reward_ax.plot(t_values, panel["rewards"], color="#2f2f2f", lw=1.5)
        reward_ax.set_title(panel["title"])
        reward_ax.set_xlim(0.5, plot_horizon + 0.5)
        reward_ax.grid(True, alpha=0.2)
        reward_ax.tick_params(axis="x", labelbottom=False)
        if col > 0:
            reward_ax.tick_params(axis="y", labelleft=False)
        else:
            reward_ax.set_ylabel(r"$R(t)$")

        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad(color="white")
        mesh_kwargs = shared_mesh_kwargs
        if mesh_kwargs is None:
            mesh_kwargs = mesh_kwargs_for_values(
                panel["heat"][np.isfinite(panel["heat"])],
                bet_scale,
                "panel",
                color_quantile,
            )
        image = ax.pcolormesh(
            x_edges,
            y_edges,
            panel["heat"].T,
            shading="auto",
            cmap=cmap,
            **mesh_kwargs,
        )
        panel_images.append(image)
        ax.tick_params(axis="x", labelbottom=False)
        if col == 0:
            ax.set_ylabel(r"log-wealth $\log w $")
            ax.annotate(
                r"hopeless region",
                xy=(5.0, -3.0),
                xytext=(30.5, -15),
                fontsize=11,
                ha="left",
                va="center",
                color="black",
            )
        else:
            ax.tick_params(axis="y", labelleft=False)

        cdf_ax = cdf_axes[col]
        cdf_ax.plot(t_values, panel["cdf"], color="#1f1f1f", lw=1.6)
        cdf_ax.set_ylim(0.0, 1.0)
        cdf_ax.grid(True, alpha=0.2)
        cdf_ax.set_xlabel("time t")
        if col == 0:
            cdf_ax.set_ylabel(r"$\pi_1^\otimes(\tau \leq t)$")
        else:
            cdf_ax.tick_params(axis="y", labelleft=False)

    if color_limits == "panel":
        last_cbar = None
        for image, ax in zip(panel_images, heatmap_axes):
            last_cbar = fig.colorbar(image, ax=ax, location="right", pad=0.02)
            last_cbar.ax.tick_params(labelsize=7)
        last_cbar.set_label(colorbar_label)
    else:
        cbar = fig.colorbar(image, ax=heatmap_axes, location="right", pad=0.02)
        cbar.set_label(colorbar_label)

    save_path = output_path or "figure_1.png"
    path = save_figure(fig, save_path, dpi=400)
    print(f"Saved plot: {path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def solve_and_simulate(args, rewards, seed):
    """Solve one reward’s policy and estimate its rejection-time distribution."""
    result = solve_optimal_policy_heatmap(
        mu0=args.mu0,
        mu1=args.mu1,
        sigma=args.sigma,
        rewards=rewards,
        horizon=args.horizon,
        n_wealth=args.n_wealth,
        n_actions=args.n_actions,
        n_quad=args.n_quad,
        lambda_max=args.lambda_max,
        two_sided=args.two_sided,
        wealth_grid=args.wealth_grid,
        min_u=args.min_u,
    )
    result.update(
        simulate_policy_cdf(
            result,
            n_sim=args.n_sim_cdf,
            seed=seed,
            u0=args.u0_cdf,
            cdf_mode=args.cdf_mode,
        )
    )
    return result


def parse_args():
    """Read experiment and output options from the command line."""
    parser = argparse.ArgumentParser(
        description=(
            "Solve the finite-horizon Gaussian Bellman problem over exponential-tilt "
            "e-variables and plot the optimal action as a heatmap over time and wealth."
        )
    )
    parser.add_argument("--mu0", type=float, default=0.0, help="Null Gaussian mean.")
    parser.add_argument("--mu1", type=float, default=0.5, help="Alternative Gaussian mean.")
    parser.add_argument(
        "--sigma", type=float, default=1.0, help="Common Gaussian standard deviation."
    )
    parser.add_argument(
        "--time-scale",
        type=float,
        default=10.0,
        help="Parameter T used by the hard-deadline, logistic, and default exponential rewards.",
    )
    parser.add_argument(
        "--exp-time-scale",
        type=float,
        default=None,
        help="Parameter T used by the exponential reward. Defaults to --time-scale.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=2.0,
        help="Slope parameter beta for the logistic reward.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=40,
        help="Finite horizon used for backward induction.",
    )
    parser.add_argument(
        "--n-wealth",
        type=int,
        default=401,
        help="Number of rescaled wealth grid points in [0, 1].",
    )
    parser.add_argument(
        "--wealth-grid",
        choices=["linear", "log"],
        default="log",
        help="Wealth grid used by the Bellman solver.",
    )
    parser.add_argument(
        "--min-u",
        type=float,
        default=1e-8,
        help="Smallest positive wealth point for --wealth-grid log.",
    )
    parser.add_argument("--n-actions", type=int, default=401, help="Number of lambda grid points.")
    parser.add_argument(
        "--lambda-max",
        type=float,
        default=None,
        help="Largest absolute Gaussian tilt. Defaults to 4 / sigma.",
    )
    parser.add_argument(
        "--two-sided",
        action="store_true",
        help="Allow both positive and negative tilts instead of only the mu1-mu0 direction.",
    )
    parser.add_argument(
        "--n-quad",
        type=int,
        default=41,
        help="Gauss-Hermite quadrature nodes for expectations under the alternative.",
    )
    parser.add_argument(
        "--n-sim-cdf",
        type=int,
        default=5000,
        help="Monte Carlo paths used for the stopping-time CDF row.",
    )
    parser.add_argument(
        "--cdf-mode",
        choices=["quadrature", "normal"],
        default="quadrature",
        help=(
            "Use the Bellman quadrature dynamics for internally consistent CDFs, "
            "or true Gaussian Monte Carlo with 'normal'."
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for the CDF simulations.")
    parser.add_argument(
        "--u0-cdf",
        type=float,
        default=0.05,
        help="Initial rescaled wealth used in the CDF simulations.",
    )
    parser.add_argument(
        "--quantity",
        choices=["lambda", "f_mu0", "f_mu1"],
        default="lambda",
        help="Which scalar action quantity to plot.",
    )
    parser.add_argument(
        "--bet-scale",
        choices=["linear", "log", "symlog"],
        default="linear",
        help=(
            "Color scale for the heatmap. 'log' masks nonpositive values; "
            "'symlog' is useful for two-sided signed tilts."
        ),
    )
    parser.add_argument(
        "--color-limits",
        choices=["global", "robust", "panel"],
        default="robust",
        help=(
            "How heatmap color limits are chosen. 'robust' clips rare extremes, "
            "'global' uses one full range, and 'panel' gives each panel its own range."
        ),
    )
    parser.add_argument(
        "--color-quantile",
        type=float,
        default=99.0,
        help="Upper percentile used by --color-limits robust; lower percentile is 100 minus this value.",
    )
    parser.add_argument(
        "--reachable-only",
        action="store_true",
        help="Only plot grid cells reachable from --u0-cdf under each panel's own optimal policy.",
    )
    parser.add_argument(
        "--no-linear",
        dest="include_linear",
        action="store_false",
        help="Skip computing and plotting the linear reward panel.",
    )
    parser.set_defaults(include_linear=True)
    parser.add_argument(
        "--max-t-plot",
        type=int,
        default=None,
        help="Largest time index t to display in the heatmaps. Default is horizon/4.",
    )
    parser.add_argument("--no-plot", action="store_true", help="Do not display the plot window.")
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional path to save the heatmap instead of the default filename.",
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
        help="Load results from --results-path and skip recomputing the Bellman solve.",
    )
    return parser.parse_args()


def main():
    """Run the experiment and save its outputs."""
    args = parse_args()
    exp_time_scale = args.time_scale if args.exp_time_scale is None else args.exp_time_scale
    if args.reuse_results and not args.results_path:
        raise ValueError("--reuse-results requires --results-path.")

    if args.reuse_results:
        results = load_results(args.results_path)
        if not args.include_linear:
            results.pop("linear_result", None)
    else:
        indicator_result = solve_and_simulate(
            args,
            build_indicator_rewards(args.time_scale, args.horizon),
            args.seed + 1,
        )
        exp_result = solve_and_simulate(
            args,
            build_exponential_rewards(exp_time_scale, args.horizon),
            args.seed + 2,
        )
        logistic_result = solve_and_simulate(
            args,
            build_logistic_rewards(args.time_scale, args.beta, args.horizon),
            args.seed + 3,
        )
        results = {
            "indicator_result": indicator_result,
            "exp_result": exp_result,
            "logistic_result": logistic_result,
        }
        if args.include_linear:
            results["linear_result"] = solve_and_simulate(
                args,
                build_linear_rewards(args.horizon),
                args.seed + 4,
            )
        if args.results_path:
            save_results(results, args, args.results_path)

    maybe_plot(
        results=results,
        quantity=args.quantity,
        output_path=args.output,
        show_plot=not args.no_plot,
        max_t_plot=args.max_t_plot,
        reachable_only=args.reachable_only,
        u0=args.u0_cdf,
        bet_scale=args.bet_scale,
        color_limits=args.color_limits,
        color_quantile=args.color_quantile,
    )


if __name__ == "__main__":
    main()
