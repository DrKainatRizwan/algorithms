"""Applications that reduce to LPs and QPs: robust regression, portfolios, SVMs."""
from __future__ import annotations

from typing import Tuple

import numpy as np

from .problem import GeneralProblem
from .solver import solve


def l1_regression(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least-absolute-deviations fit: ``min sum |y - Xw|`` as an LP.

    Variables are ``(w, r)`` with ``-r <= y - Xw <= r``.
    """
    n, d = X.shape
    c = np.concatenate([np.zeros(d), np.ones(n)])
    A_ub = np.block([[X, -np.eye(n)], [-X, -np.eye(n)]])
    b_ub = np.concatenate([y, -y])
    lb = np.concatenate([np.full(d, -np.inf), np.zeros(n)])
    return solve(GeneralProblem(c, A_ub=A_ub, b_ub=b_ub, lb=lb)).x[:d]


def chebyshev_regression(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
    """Minimax fit ``min max_i |y_i - x_i w|``; returns ``(w, max_residual)``."""
    n, d = X.shape
    c = np.concatenate([np.zeros(d), [1.0]])
    ones = np.ones((n, 1))
    A_ub = np.block([[X, -ones], [-X, -ones]])
    b_ub = np.concatenate([y, -y])
    lb = np.concatenate([np.full(d, -np.inf), [0.0]])
    x = solve(GeneralProblem(c, A_ub=A_ub, b_ub=b_ub, lb=lb)).x
    return x[:d], float(x[d])


def min_variance_portfolio(cov: np.ndarray, mu: np.ndarray, target_return: float,
                           max_weight: float = 1.0) -> np.ndarray:
    """Long-only Markowitz portfolio: ``min w'Σw`` s.t. ``1'w=1, mu'w >= target``."""
    n = mu.size
    p = GeneralProblem(c=np.zeros(n), Q=2.0 * cov,
                       A_eq=np.ones((1, n)), b_eq=np.array([1.0]),
                       A_ub=-mu[None, :], b_ub=np.array([-target_return]),
                       lb=np.zeros(n), ub=np.full(n, max_weight))
    return solve(p).x


def svm_dual(X: np.ndarray, y: np.ndarray, C: float = 1.0) -> Tuple[np.ndarray, float]:
    """Soft-margin linear SVM via its dual QP; returns ``(w, bias)``.

    ``min 0.5 a'(yy' * XX')a - 1'a``  s.t. ``y'a = 0, 0 <= a <= C``.
    """
    n = y.size
    K = (X @ X.T) * np.outer(y, y)
    p = GeneralProblem(c=-np.ones(n), Q=K, A_eq=y[None, :].astype(float), b_eq=np.zeros(1),
                       lb=np.zeros(n), ub=np.full(n, C))
    a = solve(p).x
    w = (a * y) @ X
    margin = (a > 1e-6 * C) & (a < C * (1 - 1e-6))
    idx = margin if margin.any() else (a > 1e-6 * C)
    bias = float(np.mean(y[idx] - X[idx] @ w))
    return w, bias
