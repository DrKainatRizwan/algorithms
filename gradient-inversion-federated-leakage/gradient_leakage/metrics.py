"""Reconstruction quality metrics and sample matching."""
from __future__ import annotations

from itertools import permutations

import numpy as np


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def psnr(a: np.ndarray, b: np.ndarray, peak: float = 1.0) -> float:
    """Peak signal-to-noise ratio in dB (capped at 100 for identical inputs)."""
    m = mse(a, b)
    return 100.0 if m < 1e-10 else float(10 * np.log10(peak ** 2 / m))


def ssim_global(a: np.ndarray, b: np.ndarray, peak: float = 1.0) -> float:
    """Single-window SSIM over the whole vector."""
    c1, c2 = (0.01 * peak) ** 2, (0.03 * peak) ** 2
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = float(((a - ma) * (b - mb)).mean())
    return float(((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma ** 2 + mb ** 2 + c1) * (va + vb + c2)))


def match_reconstructions(truth: np.ndarray, recon: np.ndarray) -> tuple[np.ndarray, float]:
    """Best one-to-one assignment (brute force, B<=7) minimising total MSE.

    Returns the permutation ``perm`` (``recon[perm[i]]`` matches ``truth[i]``)
    and the mean MSE under it.
    """
    B = len(truth)
    cost = np.array([[mse(truth[i], recon[j]) for j in range(B)] for i in range(B)])
    best, best_c = None, np.inf
    for perm in permutations(range(B)):
        c = sum(cost[i, perm[i]] for i in range(B))
        if c < best_c:
            best, best_c = perm, c
    return np.array(best), best_c / B
