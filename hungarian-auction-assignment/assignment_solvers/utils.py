"""Shared helpers: validation, random instances and an exhaustive reference solver."""

from __future__ import annotations

from itertools import permutations

import numpy as np


def validate_cost_matrix(cost: np.ndarray | list) -> np.ndarray:
    """Return ``cost`` as a 2-D float array, rejecting NaN and malformed input.

    ``+inf`` is allowed and marks a forbidden pairing; ``-inf`` is rejected.
    """
    a = np.asarray(cost, dtype=float)
    if a.ndim != 2:
        raise ValueError("cost matrix must be two-dimensional")
    if a.shape[0] == 0 or a.shape[1] == 0:
        raise ValueError("cost matrix must be non-empty")
    if np.isnan(a).any():
        raise ValueError("cost matrix contains NaN")
    if np.isneginf(a).any():
        raise ValueError("cost matrix contains -inf")
    return a


def random_cost_matrix(
    n: int, m: int | None = None, high: int = 100, seed: int = 0
) -> np.ndarray:
    """Random integer-valued cost matrix of shape ``(n, m)`` in ``[0, high)``."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, high, size=(n, m if m is not None else n)).astype(float)


def brute_force(cost: np.ndarray, maximize: bool = False) -> float:
    """Exhaustive optimum for small matrices (every row matched, ``n <= m``)."""
    a = validate_cost_matrix(cost)
    if a.shape[0] > a.shape[1]:
        a = a.T
    n, m = a.shape
    best = -np.inf if maximize else np.inf
    for perm in permutations(range(m), n):
        total = sum(a[i, j] for i, j in enumerate(perm))
        if (maximize and total > best) or (not maximize and total < best):
            best = total
    return float(best)
