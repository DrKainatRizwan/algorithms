"""Robust aggregation rules.

Every rule maps an ``(n_clients, dim)`` matrix of updates to one ``dim``
vector.  ``f`` is the assumed upper bound on Byzantine clients.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def _check(updates: np.ndarray) -> np.ndarray:
    u = np.asarray(updates, dtype=float)
    if u.ndim != 2 or u.shape[0] == 0:
        raise ValueError("updates must be a non-empty 2-D array")
    return u


def fedavg_mean(updates: np.ndarray, f: int = 0,
                weights: np.ndarray | None = None) -> np.ndarray:
    """Plain (optionally weighted) average -- not Byzantine-robust."""
    u = _check(updates)
    if weights is None:
        return u.mean(axis=0)
    w = np.asarray(weights, dtype=float)
    return (w[:, None] * u).sum(axis=0) / w.sum()


def coordinate_median(updates: np.ndarray, f: int = 0) -> np.ndarray:
    """Coordinate-wise median (Yin et al., 2018)."""
    return np.median(_check(updates), axis=0)


def trimmed_mean(updates: np.ndarray, f: int = 0) -> np.ndarray:
    """Coordinate-wise mean after dropping the ``f`` largest and smallest."""
    u = _check(updates)
    n = u.shape[0]
    if f < 0 or 2 * f >= n:
        raise ValueError("need 0 <= f and 2f < n")
    s = np.sort(u, axis=0)
    return s[f : n - f].mean(axis=0)


def pairwise_sq_distances(u: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance matrix, O(n^2 d)."""
    sq = np.sum(u * u, axis=1)
    d = sq[:, None] + sq[None, :] - 2.0 * (u @ u.T)
    np.fill_diagonal(d, 0.0)
    return np.maximum(d, 0.0)


def krum_scores(updates: np.ndarray, f: int) -> np.ndarray:
    """Krum score: sum of squared distances to the n-f-2 nearest peers."""
    u = _check(updates)
    n = u.shape[0]
    k = n - f - 2
    if f < 0 or k < 1:
        raise ValueError("Krum requires n >= f + 3")
    d = pairwise_sq_distances(u)
    np.fill_diagonal(d, np.inf)
    return np.sort(d, axis=1)[:, :k].sum(axis=1)


def krum(updates: np.ndarray, f: int = 0) -> np.ndarray:
    """Select the single update with the lowest Krum score (Blanchard 2017)."""
    u = _check(updates)
    return u[int(np.argmin(krum_scores(u, f)))].copy()


def multi_krum(updates: np.ndarray, f: int = 0, m: int | None = None) -> np.ndarray:
    """Average the ``m`` lowest-scoring updates (default ``n - f``)."""
    u = _check(updates)
    n = u.shape[0]
    m = n - f if m is None else m
    if not 1 <= m <= n:
        raise ValueError("m must be in [1, n]")
    best = np.argsort(krum_scores(u, f), kind="stable")[:m]
    return u[best].mean(axis=0)


def geometric_median(updates: np.ndarray, f: int = 0, iters: int = 100,
                     tol: float = 1e-8) -> np.ndarray:
    """Weiszfeld iteration for the geometric median."""
    u = _check(updates)
    z = u.mean(axis=0)
    for _ in range(iters):
        dist = np.maximum(np.linalg.norm(u - z, axis=1), 1e-12)
        w = 1.0 / dist
        z_new = (w[:, None] * u).sum(axis=0) / w.sum()
        if np.linalg.norm(z_new - z) < tol:
            return z_new
        z = z_new
    return z


AGGREGATORS: dict[str, Callable[..., np.ndarray]] = {
    "mean": fedavg_mean,
    "median": coordinate_median,
    "trimmed_mean": trimmed_mean,
    "krum": krum,
    "multi_krum": multi_krum,
    "geomed": geometric_median,
}
