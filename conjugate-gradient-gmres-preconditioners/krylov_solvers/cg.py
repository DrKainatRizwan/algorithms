"""(Preconditioned) conjugate gradient for symmetric positive definite systems."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .preconditioners import Identity, Preconditioner
from .sparse import CSRMatrix


@dataclass
class SolveResult:
    x: np.ndarray
    converged: bool
    iterations: int
    residuals: list[float] = field(default_factory=list)  # relative residual history


def conjugate_gradient(a: CSRMatrix, b: np.ndarray, x0: np.ndarray | None = None,
                       m: Preconditioner | None = None, tol: float = 1e-8,
                       maxiter: int | None = None) -> SolveResult:
    """Solve A x = b with PCG; stops when ||r|| <= tol * ||b||."""
    n = a.shape[0]
    m = m or Identity()
    maxiter = maxiter if maxiter is not None else 10 * n
    x = np.zeros(n) if x0 is None else x0.astype(float).copy()
    bnorm = np.linalg.norm(b) or 1.0
    r = b - a.matvec(x)
    hist = [np.linalg.norm(r) / bnorm]
    if hist[0] <= tol:
        return SolveResult(x, True, 0, hist)
    z = m.apply(r)
    p = z.copy()
    rz = r @ z
    for it in range(1, maxiter + 1):
        ap = a.matvec(p)
        alpha = rz / (p @ ap)
        x += alpha * p
        r -= alpha * ap
        hist.append(np.linalg.norm(r) / bnorm)
        if hist[-1] <= tol:
            return SolveResult(x, True, it, hist)
        z = m.apply(r)
        rz_new = r @ z
        p = z + (rz_new / rz) * p
        rz = rz_new
    return SolveResult(x, False, maxiter, hist)
