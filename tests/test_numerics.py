"""Independent checks of exact curves, betting validity, and EDO identities."""

import itertools
import math
import unittest

import numpy as np
from numpy.testing import assert_allclose

import bernoulli
import figure_1 as heatmap
import figure_3 as bellman
import figure_4 as gaussian
import figure_5 as power


class NumericalChecks(unittest.TestCase):
    """Check the numerical methods against small problems and analytic identities."""

    def test_constant_curve_by_enumeration(self):
        """Compare the exact recursion with all binary paths for a small horizon."""
        horizon, p0, p1, action, alpha = 8, 0.4, 0.6, 0.8, 0.2
        pmf = np.zeros(horizon)
        for path in itertools.product((0, 1), repeat=horizon):
            weight = p1 ** sum(path) * (1 - p1) ** (horizon - sum(path))
            wealth = 1.0
            for t, x in enumerate(path):
                wealth *= action / p0 if x else (1 - action) / (1 - p0)
                if wealth >= 1 / alpha - 1e-12:
                    pmf[t] += weight
                    break
        curve = bernoulli.exact_constant_bernoulli_curve(
            p0, p1, alpha, horizon, action, "test", "test"
        )
        assert_allclose(curve["pmf"], pmf, atol=1e-14)
        assert_allclose(curve["cdf"], np.cumsum(pmf), atol=1e-14)

    def test_doob_curve_by_enumeration(self):
        """Compare Doob crossings with direct conditional probabilities of all paths."""
        horizon, p0, p1, alpha = 6, 0.4, 0.6, 0.2
        paths = sorted(
            itertools.product((0, 1), repeat=horizon),
            key=lambda x: (sum(x), x),
            reverse=True,
        )
        event = []
        level = 0.0
        for path in paths:
            probability = p0 ** sum(path) * (1 - p0) ** (horizon - sum(path))
            if level + probability > alpha:
                break
            level += probability
            event.append(path)
        pmf = np.zeros(horizon + 1)
        for path in paths:
            mass = p1 ** sum(path) * (1 - p1) ** (horizon - sum(path))
            for t in range(1, horizon + 1):
                conditional = sum(
                    p0 ** sum(tail[t:]) * (1 - p0) ** (horizon - t - sum(tail[t:]))
                    for tail in event
                    if tail[:t] == path[:t]
                )
                if conditional >= level / alpha - 1e-14:
                    pmf[t] += mass
                    break
        curve = bernoulli.exact_doob_curve(horizon, p0, p1, alpha, horizon)
        self.assertAlmostEqual(curve["level"], level)
        assert_allclose(curve["pmf"], pmf, atol=1e-14)

    def test_paper_np_boundaries(self):
        """Check the tied-event sizes reported for Figure 2 in Appendix D.2."""
        for deadline, cutoff, count in ((10, 8, 106), (20, 13, 102809)):
            boundary = bernoulli.choose_np_boundary(deadline, 0.4, 0.05)
            self.assertEqual((boundary["k"], boundary["r"]), (cutoff, count))
            self.assertLessEqual(boundary["level"], 0.05 + 1e-14)

    def test_empty_rejection_event(self):
        """A deadline too short to admit any rejection event must have zero power."""
        curve = bernoulli.exact_doob_curve(1, 0.4, 0.6, 0.05, 3)
        assert_allclose(curve["pmf"], 0.0)
        self.assertEqual(curve["level"], 0.0)

    def test_capped_bets_are_fair_and_do_not_overshoot(self):
        """Check null expectation one and both possible boundary payoffs."""
        p0 = 0.4
        wealth = np.geomspace(1e-8, 1, 100)
        for action in (0.1, 0.4, 0.9):
            q = bellman.cap_bernoulli_action(action, wealth, p0)
            e1, e0 = q / p0, (1 - q) / (1 - p0)
            assert_allclose(p0 * e1 + (1 - p0) * e0, 1.0)
            self.assertTrue(np.all(wealth * e1 <= 1 + 1e-14))
            self.assertTrue(np.all(wealth * e0 <= 1 + 1e-14))

    def test_bernoulli_edo_equations(self):
        """Check the EDO root, null fairness, and theoretical drift identity."""
        p0, p1, time_scale = 0.5, 0.7, 10.0
        result = power.solve_eta_star(p0, p1, time_scale)
        eta, f0, f1 = result["eta_star"], result["f0_star"], result["f1_star"]
        self.assertAlmostEqual(power.log_optimal_eta_moment(p0, p1, eta), 1 / time_scale)
        self.assertAlmostEqual((1 - p0) * f0 + p0 * f1, 1)
        drift = (power.bernoulli_kl(p1, p0) - 1 / time_scale) / (1 - eta)
        self.assertAlmostEqual(power.compute_drift(p1, f0, f1), drift)
        kappa = power.solve_kappa(p1, f0, f1)
        self.assertGreater(kappa, 0)
        self.assertAlmostEqual(power.log_mgf_under_pi1(p1, f0, f1, kappa), 0)
        other = bellman.bernoulli_constant_approximation(p0, p1, time_scale)
        self.assertAlmostEqual(other["q_star"], p0 * f1)

    def test_exact_doob_null_control(self):
        """Check the paper deadlines using independent null probability propagation."""
        for deadline in (10, 20, 25, 100):
            curve = bernoulli.exact_doob_curve(deadline, 0.4, 0.4, 0.05, deadline)
            self.assertLessEqual(curve["power"], 0.05 + 1e-12)
            self.assertTrue(np.all(np.diff(curve["cdf"]) >= 0))

    def test_invalid_simulation_sizes(self):
        """Reject a zero batch size instead of entering an infinite loop."""
        with self.assertRaises(ValueError):
            power.estimate_power_monte_carlo(0.7, 0.05, 0.5, 1.5, 1, 10, 0, 10, 1e-8, 0)

    def test_gaussian_edo_and_quadrature(self):
        """Check Gaussian moments and the closed-form EDO action and exponent."""
        mu, sigma, time_scale = 0.25, 1.0, 14.0
        x, weights = heatmap.gaussian_alt_quadrature(mu, sigma, 41)
        self.assertAlmostEqual(float(weights.sum()), 1)
        self.assertAlmostEqual(float(weights @ x), mu)
        self.assertAlmostEqual(float(weights @ (x - mu) ** 2), sigma**2)
        action = gaussian.gaussian_approx_action(mu, sigma, time_scale)
        eta = gaussian.gaussian_eta_star(mu, sigma, time_scale)
        self.assertAlmostEqual(action, mu / (1 - eta))
        null_x, null_weights = heatmap.gaussian_alt_quadrature(0, sigma, 41)
        multipliers = heatmap.gaussian_e_multiplier(null_x, 0, sigma, np.array([action]))
        self.assertAlmostEqual(float(multipliers[0] @ null_weights), 1)

    def test_one_step_bellman(self):
        """Check one-step Bellman values against the analytic optimum."""
        result = bellman.solve_bernoulli_bellman(0.5, 0.7, 30, 1, 401, 1e-4, 401, 0.5)
        # Away from the discontinuity at wealth 1/2, the analytic optimum is
        # zero below 1/2 and 0.7 times the reward above 1/2.
        wealth = np.array([0.1, 0.25, 0.4, 0.6])
        actual = np.interp(wealth, result["u_grid"], result["values"][0])
        assert_allclose(actual, [0, 0, 0, 0.7 * math.exp(-1 / 30)], atol=1e-14)


if __name__ == "__main__":
    unittest.main()
