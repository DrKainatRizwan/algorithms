"""Proximal operators and projections."""
from __future__ import annotations

import numpy as np


def soft_threshold(v: np.ndarray, t: float) -> np.ndarray:
    """prox of t*||.||_1: sign(v) * max(|v| - t, 0)."""
    return np.sign(v) * np.maximum(np.abs(v) - t, 0.0)


def prox_group_l2(v: np.ndarray, t: float, groups: list[np.ndarray]) -> np.ndarray:
    """Block soft-thresholding: prox of t * sum_g ||v_g||_2 for disjoint groups."""
    out = v.copy()
    for g in groups:
        norm = np.linalg.norm(v[g])
        out[g] = 0.0 if norm <= t else (1 - t / norm) * v[g]
    return out


def prox_elastic_net(v: np.ndarray, t: float, l1: float, l2: float) -> np.ndarray:
    """prox of t*(l1 ||x||_1 + 0.5 l2 ||x||^2)."""
    return soft_threshold(v, t * l1) / (1.0 + t * l2)


def prox_box(v: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Projection onto the box [lo, hi]^n."""
    return np.clip(v, lo, hi)


def project_l1_ball(v: np.ndarray, radius: float) -> np.ndarray:
    """Euclidean projection onto {x : ||x||_1 <= radius} (Duchi et al., 2008).

    Sort-based O(n log n) algorithm.
    """
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if np.abs(v).sum() <= radius:
        return v.copy()
    u = np.sort(np.abs(v))[::-1]
    css = np.cumsum(u)
    ks = np.arange(1, len(v) + 1)
    rho = np.nonzero(u * ks > (css - radius))[0][-1]
    theta = (css[rho] - radius) / (rho + 1)
    return soft_threshold(v, theta)
