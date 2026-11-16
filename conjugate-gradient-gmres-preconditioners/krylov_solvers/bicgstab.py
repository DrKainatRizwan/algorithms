"""BiCGSTAB for general nonsymmetric systems."""
from __future__ import annotations

import numpy as np

from .cg import SolveResult
from .preconditioners import Identity, Preconditioner
from .sparse import CSRMatrix


def bicgstab(a: CSRMatrix, b: np.ndarray, x0: np.ndarray | None = None,
             m: Preconditioner | None = None, tol: float = 1e-8,
             maxiter: int | None = None) -> SolveResult:
    """Right-preconditioned BiCGSTAB (van der Vorst, 1992)."""
    n = a.shape[0]
    m = m or Identity()
    maxiter = maxiter if maxiter is not None else 10 * n
    x = np.zeros(n) if x0 is None else x0.astype(float).copy()
    bnorm = np.linalg.norm(b) or 1.0
    r = b - a.matvec(x)
    hist = [np.linalg.norm(r) / bnorm]
    if hist[0] <= tol:
        return SolveResult(x, True, 0, hist)
    rhat = r.copy()
    rho = alpha = omega = 1.0
    v = p = np.zeros(n)
    for it in range(1, maxiter + 1):
        rho_new = rhat @ r
        if rho_new == 0:
            break
        beta = (rho_new / rho) * (alpha / omega)
        p = r + beta * (p - omega * v)
        ph = m.apply(p)
        v = a.matvec(ph)
        alpha = rho_new / (rhat @ v)
        s = r - alpha * v
        if np.linalg.norm(s) / bnorm <= tol:
            x += alpha * ph
            hist.append(np.linalg.norm(s) / bnorm)
            return SolveResult(x, True, it, hist)
        sh = m.apply(s)
        t = a.matvec(sh)
        omega = (t @ s) / (t @ t)
        x += alpha * ph + omega * sh
        r = s - omega * t
        rho = rho_new
        hist.append(np.linalg.norm(r) / bnorm)
        if hist[-1] <= tol:
            return SolveResult(x, True, it, hist)
    return SolveResult(x, False, len(hist) - 1, hist)
