"""ADMM for the generalised lasso  min 0.5||Xw-y||^2 + lam ||D w||_1."""
from __future__ import annotations

import numpy as np

from .problems import SolverResult
from .proximal import soft_threshold


def difference_matrix(p: int, order: int = 1) -> np.ndarray:
    """Finite-difference operator of given order, shape (p-order, p).

    order=1 gives the fused lasso, order=2 gives l1 trend filtering.
    """
    D = np.eye(p)
    for _ in range(order):
        D = D[1:] - D[:-1]
    return D


def generalized_lasso_admm(X: np.ndarray, y: np.ndarray, lam: float,
                           D: np.ndarray | None = None, rho: float = 1.0,
                           max_iter: int = 5000, abs_tol: float = 1e-9,
                           rel_tol: float = 1e-8, adapt: bool = True) -> SolverResult:
    """Scaled-form ADMM (Boyd et al., 2011, sec. 6.4).

    Splitting z = D w gives the updates
        w <- (X'X + rho D'D)^-1 (X'y + rho D'(z - u))
        z <- S_{lam/rho}(D w + u)
        u <- u + D w - z
    The system matrix is inverted once and refreshed only when ``rho`` changes
    (residual balancing).  Stops on the primal/dual residual criteria.
    """
    p = X.shape[1]
    D = np.eye(p) if D is None else D
    m = D.shape[0]
    Xty = X.T @ y
    XtX = X.T @ X
    DtD = D.T @ D

    def factor(r: float) -> np.ndarray:
        return np.linalg.inv(XtX + r * DtD)

    inv = factor(rho)
    w = np.zeros(p)
    z = np.zeros(m)
    u = np.zeros(m)
    hist: list[float] = []
    converged = False
    it = 0

    def obj(wv: np.ndarray) -> float:
        r = X @ wv - y
        return 0.5 * float(r @ r) + lam * float(np.abs(D @ wv).sum())

    for it in range(1, max_iter + 1):
        w = inv @ (Xty + rho * D.T @ (z - u))
        Dw = D @ w
        z_old = z
        z = soft_threshold(Dw + u, lam / rho)
        u = u + Dw - z
        hist.append(obj(w))
        r_norm = np.linalg.norm(Dw - z)
        s_norm = rho * np.linalg.norm(D.T @ (z - z_old))
        eps_pri = np.sqrt(m) * abs_tol + rel_tol * max(np.linalg.norm(Dw), np.linalg.norm(z))
        eps_dual = np.sqrt(p) * abs_tol + rel_tol * rho * np.linalg.norm(D.T @ u)
        if r_norm <= eps_pri and s_norm <= eps_dual:
            converged = True
            break
        if adapt and it % 10 == 0:
            if r_norm > 10 * s_norm:
                rho *= 2; u = u / 2; inv = factor(rho)
            elif s_norm > 10 * r_norm:
                rho /= 2; u = u * 2; inv = factor(rho)
    return SolverResult(w, hist[-1], it, converged, hist, aux=z)


def lasso_admm(X: np.ndarray, y: np.ndarray, lam: float, rho: float = 1.0,
               **kw) -> SolverResult:
    """Standard lasso via ADMM (D = I).

    The split variable z is exactly sparse, so it is returned as the estimate.
    """
    res = generalized_lasso_admm(X, y, lam, None, rho, **kw)
    z = res.aux
    r = X @ z - y
    obj = 0.5 * float(r @ r) + lam * float(np.abs(z).sum())
    return SolverResult(z, obj, res.iterations, res.converged, res.history, aux=res.w)
