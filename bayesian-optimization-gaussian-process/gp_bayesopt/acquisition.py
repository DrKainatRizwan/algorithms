"""Acquisition functions for minimisation (larger value = more attractive)."""

from __future__ import annotations

import math

import numpy as np

_erf = np.vectorize(math.erf, otypes=[float])


def norm_pdf(z: np.ndarray) -> np.ndarray:
    """Standard normal density."""
    return np.exp(-0.5 * np.asarray(z, float) ** 2) / math.sqrt(2.0 * math.pi)


def norm_cdf(z: np.ndarray) -> np.ndarray:
    """Standard normal distribution function."""
    return 0.5 * (1.0 + _erf(np.asarray(z, float) / math.sqrt(2.0)))


def expected_improvement(mu: np.ndarray, std: np.ndarray, best: float, xi: float = 0.0) -> np.ndarray:
    """E[max(best - xi - f, 0)] for f ~ N(mu, std^2)."""
    mu, std = np.asarray(mu, float), np.asarray(std, float)
    imp = best - xi - mu
    z = imp / np.maximum(std, 1e-12)
    ei = imp * norm_cdf(z) + std * norm_pdf(z)
    return np.where(std > 1e-12, ei, np.maximum(imp, 0.0))


def probability_of_improvement(mu: np.ndarray, std: np.ndarray, best: float, xi: float = 0.0) -> np.ndarray:
    """P(f < best - xi)."""
    z = (best - xi - np.asarray(mu, float)) / np.maximum(np.asarray(std, float), 1e-12)
    return norm_cdf(z)


def upper_confidence_bound(mu: np.ndarray, std: np.ndarray, beta: float = 2.0) -> np.ndarray:
    """Lower-confidence-bound form for minimisation: -(mu - sqrt(beta) * std)."""
    return -(np.asarray(mu, float) - math.sqrt(beta) * np.asarray(std, float))
