"""Exact Bernoulli rejection probabilities and hard-deadline tests."""

from __future__ import annotations

import math
from collections import defaultdict
from math import comb
from typing import Any

import numpy as np

_CMP_TO_INDEX = {-1: 0, 0: 1, 1: 2}
_INDEX_TO_CMP = {0: -1, 1: 0, 2: 1}


def binom_pmf(n: int, p: float, k: int) -> float:
    """Return the probability of exactly k successes in n trials."""
    if k < 0 or k > n:
        return 0.0
    return comb(n, k) * (p**k) * ((1.0 - p) ** (n - k))


def binom_tail(n: int, p: float, k: int) -> float:
    """Return the probability of at least k successes in n trials."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return sum(binom_pmf(n, p, j) for j in range(k, n + 1))


def choose_np_boundary(T: int, p0: float, alpha: float) -> dict[str, Any]:
    """Choose the upper-tail event and number of tied strings to include."""
    if T < 1 or not 0.0 < p0 < 1.0 or not 0.0 < alpha < 1.0:
        raise ValueError("Require T >= 1 and p0, alpha in (0, 1).")
    k = next(k for k in range(T + 2) if binom_tail(T, p0, k) <= alpha)
    m = k - 1

    tail_above = binom_tail(T, p0, k)
    atom0 = (p0**m) * ((1.0 - p0) ** (T - m))
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
    """Return the r-th binary string with m ones, in descending order."""
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
    """Update the prefix comparison: -1 below, 0 equal, or 1 above."""
    if cmp_state != 0:
        return cmp_state
    if bit > cbit:
        return 1
    if bit < cbit:
        return -1
    return 0


def build_conditional_np_prob_table(
    T: int,
    p0: float,
    k: int,
    m: int,
    c_star: str | None,
) -> np.ndarray:
    """Compute null rejection-event probabilities for each prefix state."""
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
    """Compute exact Doob rejection probabilities, including time zero."""
    if horizon < T:
        raise ValueError("horizon must be at least T.")
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

    conditional_prob_threshold = level / alpha if level > 0.0 else np.inf

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


def exact_constant_bernoulli_curve(
    p0: float,
    p1: float,
    alpha: float,
    horizon: int,
    p_action: float,
    label: str,
    kind: str,
) -> dict[str, Any]:
    """Compute exact rejection probabilities for a constant Bernoulli bet."""
    if not (0.0 < p_action < 1.0):
        raise ValueError("p_action must lie in (0,1).")

    log_threshold = math.log(1.0 / alpha)
    log_e1 = math.log(p_action / p0)
    log_e0 = math.log((1.0 - p_action) / (1.0 - p0))

    alive: dict[int, float] = {0: 1.0}
    pmf_full = np.zeros(horizon + 1, dtype=float)

    for t in range(1, horizon + 1):
        new_alive: defaultdict[int, float] = defaultdict(float)

        for s, mass in alive.items():
            for bit, prob in ((0, 1.0 - p1), (1, p1)):
                s_new = s + bit
                logw = s_new * log_e1 + (t - s_new) * log_e0
                mass_new = mass * prob

                if logw >= log_threshold - 1e-14:
                    pmf_full[t] += mass_new
                else:
                    new_alive[s_new] += mass_new

        alive = dict(new_alive)

    cdf_full = np.cumsum(pmf_full)

    return {
        "label": label,
        "kind": kind,
        "pmf": pmf_full[1 : horizon + 1],
        "cdf": cdf_full[1 : horizon + 1],
        "power": float(cdf_full[horizon]),
        "p_action": float(p_action),
    }
