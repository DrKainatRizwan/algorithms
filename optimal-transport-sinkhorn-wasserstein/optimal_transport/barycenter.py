"""Wasserstein barycenters of histograms on a shared support."""
from __future__ import annotations

import numpy as np

from .cost import as_histogram


def entropic_barycenter(hists: np.ndarray, C: np.ndarray, weights: np.ndarray | None = None,
                        eps: float = 0.01, max_iter: int = 1000, tol: float = 1e-10) -> np.ndarray:
    """Fixed-support entropic barycenter via iterative Bregman projections.

    ``hists`` has shape ``(k, n)``; ``C`` is the ``(n, n)`` ground cost.
    Implements Benamou et al. (2015), with the scaling iterations in the primal domain.
    """
    H = np.vstack([as_histogram(h, "hist") for h in hists])
    k, n = H.shape
    lam = np.ones(k) / k if weights is None else np.asarray(weights, float) / np.sum(weights)
    K = np.exp(-C / eps)
    v = np.ones((k, n))
    bary = np.ones(n) / n
    for _ in range(max_iter):
        u = H / (v @ K.T)               # u_s = h_s / (K v_s)   (K symmetric)
        KTu = u @ K                      # rows: K^T u_s
        new = np.exp(lam @ np.log(np.maximum(v * KTu, 1e-300)))
        v = new[None, :] / np.maximum(KTu, 1e-300)
        if np.abs(new - bary).sum() < tol:
            bary = new
            break
        bary = new
    return bary / bary.sum()


def displacement_interpolation_1d(x, y, t: float, n_atoms: int = 200) -> np.ndarray:
    """McCann interpolant at time ``t`` between two 1-D samples (quantile mixing)."""
    from .one_dimensional import quantile_function

    q = (np.arange(n_atoms) + 0.5) / n_atoms
    return (1 - t) * quantile_function(x, None, q) + t * quantile_function(y, None, q)


def barycentric_map(P: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Barycentric projection of the source points: ``T(x_i) = sum_j P_ij y_j / sum_j P_ij``."""
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]
    return (P @ Y) / P.sum(axis=1, keepdims=True)
