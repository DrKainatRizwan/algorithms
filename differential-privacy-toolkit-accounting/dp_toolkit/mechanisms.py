"""Core noise-adding mechanisms and noise calibration."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def _check(eps: float, sens: float) -> None:
    if eps <= 0:
        raise ValueError("epsilon must be positive")
    if sens < 0:
        raise ValueError("sensitivity must be non-negative")


def laplace_mechanism(value, sensitivity: float, epsilon: float,
                      rng: np.random.Generator):
    """Release ``value + Lap(sensitivity/epsilon)``; satisfies (eps, 0)-DP for L1 sensitivity."""
    _check(epsilon, sensitivity)
    scale = sensitivity / epsilon
    v = np.asarray(value, dtype=float)
    out = v + rng.laplace(0.0, scale, size=v.shape)
    return float(out) if out.ndim == 0 else out


def classical_gaussian_sigma(sensitivity: float, epsilon: float, delta: float) -> float:
    """Classical calibration sigma = sens*sqrt(2 ln(1.25/delta))/eps (valid for eps<1)."""
    _check(epsilon, sensitivity)
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0,1)")
    return sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon


def _phi(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def gaussian_delta(sigma: float, sensitivity: float, epsilon: float) -> float:
    """Exact delta(eps) of the Gaussian mechanism (Balle & Wang 2018)."""
    a = sensitivity / (2 * sigma)
    b = epsilon * sigma / sensitivity
    return _phi(a - b) - math.exp(epsilon) * _phi(-a - b)


def analytic_gaussian_sigma(sensitivity: float, epsilon: float, delta: float,
                            tol: float = 1e-10) -> float:
    """Smallest sigma such that the Gaussian mechanism is (eps, delta)-DP (bisection)."""
    _check(epsilon, sensitivity)
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0,1)")
    if sensitivity == 0:
        return 0.0
    lo, hi = 1e-8 * sensitivity, sensitivity
    while gaussian_delta(hi, sensitivity, epsilon) > delta:
        hi *= 2
    while hi - lo > tol * hi:
        mid = 0.5 * (lo + hi)
        if gaussian_delta(mid, sensitivity, epsilon) > delta:
            lo = mid
        else:
            hi = mid
    return hi


def gaussian_mechanism(value, sensitivity: float, epsilon: float, delta: float,
                       rng: np.random.Generator, analytic: bool = True):
    """Release ``value + N(0, sigma^2)`` for L2 sensitivity, (eps, delta)-DP."""
    sigma = (analytic_gaussian_sigma if analytic else classical_gaussian_sigma)(
        sensitivity, epsilon, delta)
    v = np.asarray(value, dtype=float)
    out = v + rng.normal(0.0, sigma, size=v.shape)
    return float(out) if out.ndim == 0 else out


def geometric_mechanism(value: int, sensitivity: int, epsilon: float,
                        rng: np.random.Generator) -> int:
    """Two-sided geometric (discrete Laplace) noise for integer queries."""
    _check(epsilon, sensitivity)
    alpha = math.exp(-epsilon / max(sensitivity, 1))
    # difference of two geometric variables is two-sided geometric
    p = 1 - alpha
    g1 = rng.geometric(p) - 1
    g2 = rng.geometric(p) - 1
    return int(value) + int(g1 - g2)


def exponential_mechanism(scores: Sequence[float], sensitivity: float, epsilon: float,
                          rng: np.random.Generator) -> int:
    """Sample index i with probability proportional to exp(eps*score_i/(2*sens))."""
    _check(epsilon, sensitivity)
    p = exponential_probabilities(scores, sensitivity, epsilon)
    return int(rng.choice(len(p), p=p))


def exponential_probabilities(scores: Sequence[float], sensitivity: float,
                              epsilon: float) -> np.ndarray:
    """Selection distribution of the exponential mechanism (numerically stable)."""
    s = np.asarray(scores, dtype=float)
    logits = epsilon * s / (2.0 * max(sensitivity, 1e-300))
    logits -= logits.max()
    w = np.exp(logits)
    return w / w.sum()


def randomized_response(bits: np.ndarray, epsilon: float,
                        rng: np.random.Generator) -> np.ndarray:
    """Warner's randomized response: keep each bit w.p. e^eps/(1+e^eps)."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    b = np.asarray(bits).astype(bool)
    keep = rng.random(b.shape) < math.exp(epsilon) / (1 + math.exp(epsilon))
    return np.where(keep, b, ~b).astype(int)


def debias_randomized_response(noisy: np.ndarray, epsilon: float) -> float:
    """Unbiased estimate of the true proportion of ones from randomized responses."""
    p = math.exp(epsilon) / (1 + math.exp(epsilon))
    return float((np.mean(noisy) - (1 - p)) / (2 * p - 1))
