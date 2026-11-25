"""Problem generation, objective values and optimality certificates."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SolverResult:
    """Common return type for all solvers."""
    w: np.ndarray
    objective: float
    iterations: int
    converged: bool
    history: list[float] = field(default_factory=list)
    aux: np.ndarray | None = None  # solver-specific extra (e.g. ADMM split variable)


def make_sparse_regression(n: int, p: int, k: int, noise: float = 0.1,
                           corr: float = 0.0, seed: int = 0
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gaussian design with AR(1)-style correlation ``corr`` and k-sparse truth.

    Columns are generated as x_j = sqrt(1-corr) z_j + sqrt(corr) z_0 so that any
    pair of columns has correlation ``corr``.
    """
    if not 0 <= k <= p:
        raise ValueError("need 0 <= k <= p")
    if not 0 <= corr < 1:
        raise ValueError("corr must be in [0, 1)")
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, p))
    common = rng.standard_normal((n, 1))
    X = np.sqrt(1 - corr) * Z + np.sqrt(corr) * common
    w = np.zeros(p)
    idx = rng.choice(p, size=k, replace=False)
    signs = rng.choice([-1.0, 1.0], size=k)
    w[idx] = signs * rng.uniform(1.0, 3.0, size=k)
    y = X @ w + noise * rng.standard_normal(n)
    return X, y, w


def lasso_objective(X: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float) -> float:
    """0.5 ||Xw - y||^2 + lam ||w||_1."""
    r = X @ w - y
    return 0.5 * float(r @ r) + lam * float(np.abs(w).sum())


def lambda_max(X: np.ndarray, y: np.ndarray) -> float:
    """Smallest lambda for which the lasso solution is exactly zero."""
    return float(np.max(np.abs(X.T @ y)))


def kkt_violation(X: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float) -> float:
    """Max violation of the lasso subgradient optimality conditions.

    With g = X^T (Xw - y): active coordinates need g_j = -lam sign(w_j),
    inactive ones need |g_j| <= lam.
    """
    g = X.T @ (X @ w - y)
    active = w != 0
    v = np.where(active, np.abs(g + lam * np.sign(w)), np.maximum(np.abs(g) - lam, 0.0))
    return float(v.max()) if v.size else 0.0


def lipschitz_constant(X: np.ndarray, iters: int = 200, seed: int = 0) -> float:
    """Largest eigenvalue of X^T X by power iteration (gradient Lipschitz constant)."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(X.shape[1])
    v /= np.linalg.norm(v)
    lam = 0.0
    for _ in range(iters):
        u = X.T @ (X @ v)
        nu = np.linalg.norm(u)
        if nu == 0:
            return 0.0
        v = u / nu
        new = float(v @ (X.T @ (X @ v)))
        if abs(new - lam) <= 1e-12 * max(new, 1.0):
            lam = new
            break
        lam = new
    return lam * (1 + 1e-9)
