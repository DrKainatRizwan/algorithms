"""Regularisation paths, cross-validation and support-recovery metrics."""
from __future__ import annotations

import numpy as np

from .coordinate import coordinate_descent
from .problems import lambda_max


def lasso_path(X: np.ndarray, y: np.ndarray, n_lambdas: int = 30, eps: float = 1e-3,
               alpha: float = 1.0, tol: float = 1e-9
               ) -> tuple[np.ndarray, np.ndarray]:
    """Log-spaced lambda grid from lambda_max down to eps*lambda_max, warm-started.

    Returns (lambdas, coefs) with coefs of shape (n_lambdas, p).
    """
    lmax = lambda_max(X, y) / max(alpha, 1e-12)
    lams = np.geomspace(lmax, eps * lmax, n_lambdas)
    coefs = np.zeros((n_lambdas, X.shape[1]))
    w = np.zeros(X.shape[1])
    for i, lam in enumerate(lams):
        w = coordinate_descent(X, y, lam, alpha, w0=w, tol=tol).w
        coefs[i] = w
    return lams, coefs


def cross_validate(X: np.ndarray, y: np.ndarray, lambdas: np.ndarray, folds: int = 5,
                   seed: int = 0) -> tuple[float, np.ndarray]:
    """K-fold CV over a lambda grid; returns (best_lambda, mean_mse per lambda)."""
    n = X.shape[0]
    perm = np.random.default_rng(seed).permutation(n)
    parts = np.array_split(perm, folds)
    errs = np.zeros((folds, len(lambdas)))
    for f, test in enumerate(parts):
        train = np.setdiff1d(perm, test)
        w = np.zeros(X.shape[1])
        for i, lam in enumerate(lambdas):          # descending lambdas assumed
            w = coordinate_descent(X[train], y[train], lam, w0=w, tol=1e-8).w
            errs[f, i] = np.mean((X[test] @ w - y[test]) ** 2)
    mean = errs.mean(axis=0)
    return float(lambdas[int(np.argmin(mean))]), mean


def support_metrics(w_hat: np.ndarray, w_true: np.ndarray, thresh: float = 1e-8
                    ) -> dict[str, float]:
    """Precision, recall and exact-recovery flag of the estimated support."""
    s_hat = np.abs(w_hat) > thresh
    s_true = w_true != 0
    tp = int(np.sum(s_hat & s_true))
    prec = tp / max(int(s_hat.sum()), 1)
    rec = tp / max(int(s_true.sum()), 1)
    return {"precision": prec, "recall": rec, "exact": float(np.array_equal(s_hat, s_true))}
