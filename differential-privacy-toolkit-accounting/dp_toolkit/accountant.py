"""Privacy accounting: basic, advanced and Renyi-DP composition."""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np

DEFAULT_ORDERS: Tuple[float, ...] = tuple(
    [1.25, 1.5, 1.75, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64, 128, 256])


class PrivacyBudgetExceeded(RuntimeError):
    """Raised when a spend would exceed the configured total budget."""


class BasicAccountant:
    """Sequential composition: epsilons and deltas add up."""

    def __init__(self, eps_budget: float, delta_budget: float = 0.0):
        self.eps_budget, self.delta_budget = eps_budget, delta_budget
        self.spent: List[Tuple[float, float]] = []

    @property
    def epsilon(self) -> float:
        return sum(e for e, _ in self.spent)

    @property
    def delta(self) -> float:
        return sum(d for _, d in self.spent)

    def spend(self, eps: float, delta: float = 0.0) -> None:
        if (self.epsilon + eps > self.eps_budget + 1e-12 or
                self.delta + delta > self.delta_budget + 1e-15):
            raise PrivacyBudgetExceeded(f"spending ({eps}, {delta}) exceeds budget")
        self.spent.append((eps, delta))

    def remaining(self) -> Tuple[float, float]:
        return self.eps_budget - self.epsilon, self.delta_budget - self.delta


class AdvancedAccountant:
    """Advanced composition (Dwork-Rothblum-Vadhan) for k-fold (eps, delta) mechanisms."""

    @staticmethod
    def compose(eps: float, delta: float, k: int, delta_slack: float) -> Tuple[float, float]:
        """Return the (eps', k*delta + delta_slack) guarantee of k adaptive runs."""
        if k < 1 or delta_slack <= 0:
            raise ValueError("need k>=1 and delta_slack>0")
        e = (math.sqrt(2 * k * math.log(1 / delta_slack)) * eps
             + k * eps * (math.exp(eps) - 1))
        return e, k * delta + delta_slack

    @staticmethod
    def best(eps: float, delta: float, k: int, delta_slack: float) -> Tuple[float, float]:
        """Tighter of basic and advanced composition."""
        adv = AdvancedAccountant.compose(eps, delta, k, delta_slack)
        basic = (k * eps, k * delta)
        return basic if basic[0] <= adv[0] else adv


def gaussian_rdp(orders: Sequence[float], sigma: float) -> np.ndarray:
    """RDP of the Gaussian mechanism with unit sensitivity: alpha/(2 sigma^2)."""
    return np.asarray(orders, dtype=float) / (2.0 * sigma ** 2)


def _log_comb(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _logsumexp(xs: Sequence[float]) -> float:
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def subsampled_gaussian_rdp(orders: Sequence[float], sigma: float, q: float) -> np.ndarray:
    """RDP of the Poisson-subsampled Gaussian mechanism (Mironov et al. 2019).

    Exact for integer orders; fractional orders are rounded up (a valid upper bound
    because RDP is non-decreasing in alpha).
    """
    if not 0 <= q <= 1:
        raise ValueError("q must be in [0,1]")
    out = []
    for a in orders:
        if q == 0:
            out.append(0.0)
            continue
        if q == 1:
            out.append(a / (2 * sigma ** 2))
            continue
        n = int(math.ceil(a))
        if n < 2:
            n = 2
        terms = []
        for k in range(n + 1):
            t = _log_comb(n, k) + (n - k) * math.log1p(-q) + k * math.log(q)
            t += (k * k - k) / (2 * sigma ** 2)
            terms.append(t)
        out.append(_logsumexp(terms) / (n - 1))
    return np.asarray(out)


def rdp_to_dp(orders: Sequence[float], rdp: Sequence[float], delta: float
              ) -> Tuple[float, float]:
    """Convert RDP curve to (eps, delta)-DP using the Balle et al. 2020 bound.

    Returns ``(epsilon, best_order)``.
    """
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0,1)")
    best, best_a = math.inf, float("nan")
    for a, r in zip(orders, rdp):
        if a <= 1:
            continue
        eps = (r + math.log((a - 1) / a) - (math.log(delta) + math.log(a)) / (a - 1))
        if eps < best:
            best, best_a = eps, a
    return max(best, 0.0), best_a


class RDPAccountant:
    """Tracks the cumulative RDP curve of heterogeneous mechanisms."""

    def __init__(self, orders: Sequence[float] = DEFAULT_ORDERS):
        self.orders = tuple(orders)
        self.rdp = np.zeros(len(self.orders))
        self.steps = 0

    def compose_gaussian(self, sigma: float, sensitivity: float = 1.0,
                         steps: int = 1) -> None:
        self.rdp += steps * gaussian_rdp(self.orders, sigma / sensitivity)
        self.steps += steps

    def compose_subsampled_gaussian(self, sigma: float, q: float, steps: int = 1) -> None:
        self.rdp += steps * subsampled_gaussian_rdp(self.orders, sigma, q)
        self.steps += steps

    def compose_laplace(self, epsilon: float, steps: int = 1) -> None:
        """RDP of pure eps-DP (Laplace) via the Mironov formula, with the eps cap."""
        for i, a in enumerate(self.orders):
            if a == 1:
                continue
            val = (1 / (a - 1)) * math.log(
                a / (2 * a - 1) * math.exp((a - 1) * epsilon)
                + (a - 1) / (2 * a - 1) * math.exp(-a * epsilon))
            self.rdp[i] += steps * min(val, epsilon)
        self.steps += steps

    def epsilon(self, delta: float) -> float:
        return rdp_to_dp(self.orders, self.rdp, delta)[0]

    def noise_for_target(self, target_eps: float, delta: float, q: float, steps: int,
                         lo: float = 0.3, hi: float = 100.0) -> float:
        """Smallest noise multiplier achieving ``target_eps`` for DP-SGD style training."""
        def eps_of(sig: float) -> float:
            acc = RDPAccountant(self.orders)
            acc.compose_subsampled_gaussian(sig, q, steps)
            return acc.epsilon(delta)
        if eps_of(hi) > target_eps:
            raise ValueError("target epsilon unreachable with sigma <= hi")
        for _ in range(40):
            mid = math.sqrt(lo * hi)
            if eps_of(mid) > target_eps:
                lo = mid
            else:
                hi = mid
        return hi
