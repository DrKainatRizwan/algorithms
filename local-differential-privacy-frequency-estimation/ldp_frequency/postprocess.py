"""Consistency post-processing for noisy frequency estimates.

Raw unbiased estimates can be negative and need not sum to one.  Projecting
them onto the probability simplex is free (it uses no extra privacy budget) and
usually reduces error substantially, at the price of a small bias.
"""
from __future__ import annotations

import numpy as np


def clip_normalize(est: np.ndarray) -> np.ndarray:
    """Clip negatives to zero and rescale to sum to one."""
    c = np.clip(est, 0.0, None)
    s = c.sum()
    return c / s if s > 0 else np.full_like(c, 1.0 / c.size)


def project_simplex(est: np.ndarray) -> np.ndarray:
    """Euclidean projection onto the simplex (sort-based, O(d log d))."""
    v = np.asarray(est, dtype=float)
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1.0
    idx = np.arange(1, v.size + 1)
    rho = np.nonzero(u - css / idx > 0)[0][-1]
    theta = css[rho] / (rho + 1)
    return np.maximum(v - theta, 0.0)


def norm_sub(est: np.ndarray, total: float = 1.0) -> np.ndarray:
    """Norm-Sub: subtract a constant from positives until they sum to ``total``.

    Iteratively zeroes negative entries and redistributes the surplus equally
    over the entries that are still positive (Wang et al., 2020).
    """
    v = np.asarray(est, dtype=float).copy()
    active = np.ones(v.size, dtype=bool)
    for _ in range(v.size + 1):
        delta = (v[active].sum() - total) / max(active.sum(), 1)
        v[active] -= delta
        neg = active & (v < 0)
        if not neg.any():
            break
        v[neg] = 0.0
        active &= ~neg
    return np.maximum(v, 0.0)


def base_cut(est: np.ndarray, n: int, variance: float, z: float = 2.0) -> np.ndarray:
    """Zero out estimates below ``z`` standard deviations of the noise floor.

    Entries that survive are rescaled to sum to one (if any survive).
    """
    thr = z * np.sqrt(variance)
    v = np.where(est > thr, est, 0.0)
    s = v.sum()
    return v / s if s > 0 else v
