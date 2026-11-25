"""Cyclic coordinate descent for the elastic net with active-set acceleration."""
from __future__ import annotations

import numpy as np

from .problems import SolverResult
from .proximal import soft_threshold


def _en_objective(X, y, w, lam, alpha):
    r = X @ w - y
    return (0.5 * float(r @ r) + lam * alpha * float(np.abs(w).sum())
            + 0.5 * lam * (1 - alpha) * float(w @ w))


def coordinate_descent(X: np.ndarray, y: np.ndarray, lam: float, alpha: float = 1.0,
                       w0: np.ndarray | None = None, active_set: bool = True,
                       max_iter: int = 1000, tol: float = 1e-10) -> SolverResult:
    """Minimise 0.5||Xw-y||^2 + lam*(alpha ||w||_1 + (1-alpha)/2 ||w||^2).

    alpha=1 is the lasso, alpha=0 ridge.  The residual is updated in O(n) per
    coordinate.  With ``active_set`` the solver sweeps only over non-zero
    coordinates until they converge, then does a full sweep to confirm.
    """
    n, p = X.shape
    w = np.zeros(p) if w0 is None else w0.astype(float).copy()
    col_sq = np.einsum("ij,ij->j", X, X)
    r = y - X @ w
    hist = [_en_objective(X, y, w, lam, alpha)]
    denom = col_sq + lam * (1 - alpha)
    l1 = lam * alpha

    def sweep(idx: np.ndarray) -> float:
        biggest = 0.0
        for j in idx:
            if denom[j] == 0:
                continue
            old = w[j]
            rho = X[:, j] @ r + col_sq[j] * old
            new = float(soft_threshold(np.array(rho), l1)) / denom[j]
            if new != old:
                r.__isub__(X[:, j] * (new - old))
                w[j] = new
                biggest = max(biggest, abs(new - old) * np.sqrt(col_sq[j]))
        return biggest

    converged = False
    it = 0
    all_idx = np.arange(p)
    for it in range(1, max_iter + 1):
        if active_set and it > 1 and np.any(w != 0):
            act = np.nonzero(w)[0]
            for _ in range(100):
                if sweep(act) <= tol * max(1.0, np.linalg.norm(w)):
                    break
        delta = sweep(all_idx)
        hist.append(_en_objective(X, y, w, lam, alpha))
        if delta <= tol * max(1.0, np.linalg.norm(w)):
            converged = True
            break
    return SolverResult(w, hist[-1], it, converged, hist)
