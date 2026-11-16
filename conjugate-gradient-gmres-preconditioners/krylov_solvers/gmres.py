"""Restarted GMRES(m) with Arnoldi (modified Gram-Schmidt) and Givens rotations."""
from __future__ import annotations

import numpy as np

from .cg import SolveResult
from .preconditioners import Identity, Preconditioner
from .sparse import CSRMatrix


def gmres(a: CSRMatrix, b: np.ndarray, x0: np.ndarray | None = None,
          m: Preconditioner | None = None, restart: int = 30, tol: float = 1e-8,
          maxiter: int = 1000) -> SolveResult:
    """Left-preconditioned GMRES; residual is measured in the preconditioned norm.

    `iterations` counts inner iterations (matrix-vector products).
    """
    n = a.shape[0]
    m = m or Identity()
    x = np.zeros(n) if x0 is None else x0.astype(float).copy()
    pb = np.linalg.norm(m.apply(b)) or 1.0
    hist: list[float] = []
    total = 0
    while True:
        r = m.apply(b - a.matvec(x))
        beta = np.linalg.norm(r)
        hist.append(beta / pb)
        if hist[-1] <= tol:
            return SolveResult(x, True, total, hist)
        if total >= maxiter:
            return SolveResult(x, False, total, hist)
        k = min(restart, n)
        V = np.zeros((k + 1, n))
        H = np.zeros((k + 1, k))
        cs, sn = np.zeros(k), np.zeros(k)
        g = np.zeros(k + 1)
        g[0] = beta
        V[0] = r / beta
        j_used = 0
        for j in range(k):
            w = m.apply(a.matvec(V[j]))
            for i in range(j + 1):
                H[i, j] = w @ V[i]
                w = w - H[i, j] * V[i]
            H[j + 1, j] = np.linalg.norm(w)
            if H[j + 1, j] > 1e-14:
                V[j + 1] = w / H[j + 1, j]
            for i in range(j):  # apply previous rotations
                t = cs[i] * H[i, j] + sn[i] * H[i + 1, j]
                H[i + 1, j] = -sn[i] * H[i, j] + cs[i] * H[i + 1, j]
                H[i, j] = t
            h1, h2 = H[j, j], H[j + 1, j]
            rho = np.hypot(h1, h2)
            cs[j], sn[j] = (1.0, 0.0) if rho == 0 else (h1 / rho, h2 / rho)
            H[j, j], H[j + 1, j] = rho, 0.0
            g[j + 1] = -sn[j] * g[j]
            g[j] = cs[j] * g[j]
            total += 1
            j_used = j + 1
            res = abs(g[j + 1]) / pb
            hist.append(res)
            if res <= tol or total >= maxiter:
                break
        y = np.linalg.solve(H[:j_used, :j_used], g[:j_used])
        x = x + V[:j_used].T @ y
