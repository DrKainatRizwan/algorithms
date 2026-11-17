"""Cost matrices and input validation helpers for discrete optimal transport."""
from __future__ import annotations

import numpy as np


def as_histogram(a: np.ndarray, name: str = "a") -> np.ndarray:
    """Validate a non-negative vector and normalise it to unit mass."""
    a = np.asarray(a, dtype=float)
    if a.ndim != 1 or a.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D vector")
    if np.any(a < 0) or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be finite and non-negative")
    total = a.sum()
    if total <= 0:
        raise ValueError(f"{name} must have positive mass")
    return a / total


def pairwise_cost(x: np.ndarray, y: np.ndarray, p: float = 2.0) -> np.ndarray:
    """Matrix ``C[i, j] = ||x_i - y_j||_2 ** p`` for point clouds ``x`` and ``y``."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    y = np.atleast_2d(np.asarray(y, dtype=float))
    if x.shape[0] == 1 and x.shape[1] != y.shape[1]:
        x = x.T
    if y.shape[0] == 1 and x.shape[1] != y.shape[1]:
        y = y.T
    if x.shape[1] != y.shape[1]:
        raise ValueError("point clouds must share the same dimension")
    d2 = (x**2).sum(1)[:, None] + (y**2).sum(1)[None, :] - 2.0 * x @ y.T
    d2 = np.maximum(d2, 0.0)
    if p == 2.0:
        return d2
    return d2 ** (p / 2.0)


def grid_cost(n: int, p: float = 2.0, scale: float = 1.0) -> np.ndarray:
    """Cost matrix between the bins of a 1-D histogram on ``n`` equispaced points."""
    t = np.linspace(0.0, scale, n)
    return np.abs(t[:, None] - t[None, :]) ** p


def check_problem(a: np.ndarray, b: np.ndarray, C: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and normalise an OT problem ``(a, b, C)``."""
    a = as_histogram(a, "a")
    b = as_histogram(b, "b")
    C = np.asarray(C, dtype=float)
    if C.shape != (a.size, b.size):
        raise ValueError(f"cost shape {C.shape} does not match marginals ({a.size}, {b.size})")
    return a, b, C


def transport_cost(P: np.ndarray, C: np.ndarray) -> float:
    """Total cost ``<P, C>`` of a transport plan."""
    return float(np.sum(P * C))
