"""Randomised range finders and low-rank factorisations (Halko-Martinsson-Tropp, Tropp et al.)."""
from __future__ import annotations

import numpy as np


def randomized_range_finder(A: np.ndarray, size: int, n_iter: int, rng: np.random.Generator) -> np.ndarray:
    """Orthonormal Q (m x size) approximating range(A) using ``n_iter`` stabilised power iterations."""
    m, n = A.shape
    size = min(size, m, n)
    Q, _ = np.linalg.qr(A @ rng.standard_normal((n, size)))
    for _ in range(n_iter):
        Z, _ = np.linalg.qr(A.T @ Q)
        Q, _ = np.linalg.qr(A @ Z)
    return Q


def randomized_svd(A: np.ndarray, k: int, oversample: int = 10, n_iter: int = 2,
                   rng: np.random.Generator | None = None):
    """Rank-k randomised SVD. Returns (U, s, Vt). Cost O(mn(k+p)(2q+1)) flops, O((m+n)(k+p)) extra memory."""
    rng = rng or np.random.default_rng(0)
    Q = randomized_range_finder(A, k + oversample, n_iter, rng)
    B = Q.T @ A
    Ub, s, Vt = np.linalg.svd(B, full_matrices=False)
    return (Q @ Ub)[:, :k], s[:k], Vt[:k]


def adaptive_range_finder(A: np.ndarray, tol: float, block: int = 5, max_rank: int | None = None,
                          rng: np.random.Generator | None = None) -> np.ndarray:
    """Grow Q block by block until the relative Frobenius residual ||A - QQ^T A||_F/||A||_F <= tol."""
    rng = rng or np.random.default_rng(0)
    m, n = A.shape
    max_rank = min(max_rank or min(m, n), m, n)
    norm2 = float(np.sum(A * A))
    Q = np.zeros((m, 0))
    while Q.shape[1] < max_rank:
        Y = A @ rng.standard_normal((n, min(block, max_rank - Q.shape[1])))
        for _ in range(2):  # re-orthogonalise twice against current basis
            Y = Y - Q @ (Q.T @ Y)
        Qn, _ = np.linalg.qr(Y)
        Q = np.hstack([Q, Qn])
        resid = norm2 - float(np.sum((Q.T @ A) ** 2))
        if np.sqrt(max(resid, 0.0) / norm2) <= tol:
            break
    return Q


def single_pass_svd(A: np.ndarray, k: int, rng: np.random.Generator | None = None,
                    oversample: int = 10):
    """Streaming-friendly rank-k SVD using only co-range and range sketches (Tropp-Yurtsever-Udell-Cevher).

    A is touched once to form Y = A Omega and W = Psi A; no second pass is needed.
    """
    rng = rng or np.random.default_rng(0)
    m, n = A.shape
    r = min(k + oversample, m, n)
    l = min(2 * r + 1, m)
    Omega = rng.standard_normal((n, r))
    Psi = rng.standard_normal((l, m))
    Y, W = A @ Omega, Psi @ A
    Q, _ = np.linalg.qr(Y)
    X = np.linalg.lstsq(Psi @ Q, W, rcond=None)[0]
    Ux, s, Vt = np.linalg.svd(X, full_matrices=False)
    return (Q @ Ux)[:, :k], s[:k], Vt[:k]


def spectral_error_estimate(A: np.ndarray, Q: np.ndarray, rng: np.random.Generator | None = None,
                            iters: int = 30) -> float:
    """Estimate ||(I - QQ^T)A||_2 by power iteration without forming the residual."""
    rng = rng or np.random.default_rng(0)
    v = rng.standard_normal(A.shape[1])
    v /= np.linalg.norm(v)
    est = 0.0
    for _ in range(iters):
        u = A @ v
        u -= Q @ (Q.T @ u)
        est = np.linalg.norm(u)
        if est == 0:
            return 0.0
        w = A.T @ (u / est)
        # A^T (I-QQ^T) u: project again because (I-P) is idempotent
        wn = np.linalg.norm(w)
        if wn == 0:
            return 0.0
        v = w / wn
    u = A @ v
    u -= Q @ (Q.T @ u)
    return float(np.linalg.norm(u))
