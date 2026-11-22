"""Overdetermined least squares: sketch-and-solve and sketch-preconditioned CGLS (Blendenpik/LSRN style)."""
from __future__ import annotations

import numpy as np

from .sketching import make_sketch


def sketch_and_solve(A: np.ndarray, b: np.ndarray, m: int, kind: str = "srht",
                     rng: np.random.Generator | None = None) -> np.ndarray:
    """Solve min ||SAx - Sb||; gives a (1+eps) residual guarantee for m = O(d/eps)."""
    rng = rng or np.random.default_rng(0)
    S = make_sketch(kind, m, A.shape[0], rng)
    Ab = S.apply(np.column_stack([A, b]))
    return np.linalg.lstsq(Ab[:, :-1], Ab[:, -1], rcond=None)[0]


def cgls(matvec, rmatvec, b: np.ndarray, n: int, tol: float = 1e-12, max_iter: int = 1000):
    """Conjugate gradient on the normal equations. Returns (x, iterations)."""
    x = np.zeros(n)
    r = b.copy()
    s = rmatvec(r)
    p = s.copy()
    gamma = s @ s
    g0 = gamma
    it = 0
    while it < max_iter and gamma > tol ** 2 * g0 and gamma > 0:
        q = matvec(p)
        alpha = gamma / (q @ q)
        x += alpha * p
        r -= alpha * q
        s = rmatvec(r)
        g_new = s @ s
        p = s + (g_new / gamma) * p
        gamma = g_new
        it += 1
    return x, it


def sketch_precondition_cgls(A: np.ndarray, b: np.ndarray, kind: str = "srht", oversample: float = 4.0,
                             tol: float = 1e-10, rng: np.random.Generator | None = None):
    """High-precision LS: QR of SA gives R; CGLS on A R^{-1} converges in O(log 1/tol) iterations.

    Returns (x, iterations, cond(A R^{-1})).
    """
    rng = rng or np.random.default_rng(0)
    n, d = A.shape
    m = min(int(oversample * d), 1 << (n - 1).bit_length()) if kind == "srht" else int(oversample * d)
    S = make_sketch(kind, m, n, rng)
    _, R = np.linalg.qr(S.apply(A))
    Rinv = np.linalg.inv(R)
    y, it = cgls(lambda v: A @ (Rinv @ v), lambda u: Rinv.T @ (A.T @ u), b, d, tol)
    cond = float(np.linalg.cond(A @ Rinv))
    return Rinv @ y, it, cond
