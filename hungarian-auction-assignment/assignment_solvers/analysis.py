"""Optimality certificates and sensitivity queries for assignment solutions."""

from __future__ import annotations

import numpy as np

from .hungarian import Assignment, hungarian
from .utils import validate_cost_matrix


def reduced_costs(cost: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Matrix of ``cost[i, j] - u[i] - v[j]``; non-negative at a minimisation optimum."""
    a = validate_cost_matrix(cost)
    return a - u[:, None] - v[None, :]


def verify_duals(
    cost: np.ndarray, result: Assignment, tol: float = 1e-9, maximize: bool = False
) -> bool:
    """Check LP-duality optimality: dual feasibility plus complementary slackness.

    For minimisation ``u_i + v_j <= c_ij`` must hold everywhere with equality
    on matched pairs; by weak duality this proves the matching optimal.  ``tol``
    allows slack, e.g. ``n * eps`` for auction outputs.
    """
    a = validate_cost_matrix(cost)
    rc = reduced_costs(a, result.u, result.v)
    if maximize:
        rc = -rc
    if (rc < -tol).any():
        return False
    if (np.abs(rc[result.rows, result.cols]) > tol).any():
        return False
    return True


def forced_pair_cost(cost: np.ndarray, i: int, j: int) -> float:
    """Cost of the best complete assignment that contains the pair ``(i, j)``.

    The difference to the unconstrained optimum is the regret of forcing the
    pair; it is zero exactly when some optimal assignment already uses it.
    """
    a = validate_cost_matrix(cost)
    n, m = a.shape
    if not (0 <= i < n and 0 <= j < m):
        raise IndexError("pair outside matrix")
    if not np.isfinite(a[i, j]):
        return float("inf")
    if n == 1 or m == 1:
        return float(a[i, j])
    minor = np.delete(np.delete(a, i, axis=0), j, axis=1)
    if minor.shape[0] > minor.shape[1] and n <= m:
        return float("inf")
    try:
        sub = hungarian(minor)
    except ValueError:
        return float("inf")
    return float(a[i, j] + sub.cost)
