"""Problem containers and the reduction of general-form programs to standard form.

General form::

    minimise    0.5 x'Qx + c'x
    subject to  A_eq x = b_eq,  A_ub x <= b_ub,  lb <= x <= ub

Standard form (what the primal-dual core solves)::

    minimise    0.5 z'Hz + g'z      subject to  Az = b,  z_i >= 0 (i not free)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class StandardForm:
    """Equality-form convex QP with non-negative variables."""

    Q: np.ndarray
    c: np.ndarray
    A: np.ndarray
    b: np.ndarray
    free: Optional[np.ndarray] = None  # boolean mask of sign-unrestricted variables

    def __post_init__(self) -> None:
        if self.free is None:
            self.free = np.zeros(self.c.size, dtype=bool)

    @property
    def n(self) -> int:
        return self.c.size

    @property
    def m(self) -> int:
        return self.b.size

    def objective(self, z: np.ndarray) -> float:
        return float(0.5 * z @ self.Q @ z + self.c @ z)


@dataclass
class GeneralProblem:
    """General-form convex QP / LP (``Q=None`` means a linear program)."""

    c: np.ndarray
    Q: Optional[np.ndarray] = None
    A_eq: Optional[np.ndarray] = None
    b_eq: Optional[np.ndarray] = None
    A_ub: Optional[np.ndarray] = None
    b_ub: Optional[np.ndarray] = None
    lb: Optional[np.ndarray] = None
    ub: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.c = np.asarray(self.c, dtype=float)
        n = self.c.size
        self.Q = np.zeros((n, n)) if self.Q is None else np.asarray(self.Q, float)
        self.A_eq = np.zeros((0, n)) if self.A_eq is None else np.atleast_2d(np.asarray(self.A_eq, float))
        self.b_eq = np.zeros(0) if self.b_eq is None else np.asarray(self.b_eq, float)
        self.A_ub = np.zeros((0, n)) if self.A_ub is None else np.atleast_2d(np.asarray(self.A_ub, float))
        self.b_ub = np.zeros(0) if self.b_ub is None else np.asarray(self.b_ub, float)
        self.lb = np.zeros(n) if self.lb is None else np.asarray(self.lb, float)
        self.ub = np.full(n, np.inf) if self.ub is None else np.asarray(self.ub, float)
        if self.Q.shape != (n, n):
            raise ValueError("Q must be n x n")
        if self.A_eq.shape != (self.b_eq.size, n) or self.A_ub.shape != (self.b_ub.size, n):
            raise ValueError("constraint dimensions are inconsistent")

    @property
    def n(self) -> int:
        return self.c.size

    def objective(self, x: np.ndarray) -> float:
        return float(0.5 * x @ self.Q @ x + self.c @ x)

    def max_violation(self, x: np.ndarray) -> float:
        """Largest constraint violation of ``x`` (0 if feasible)."""
        v = 0.0
        if self.b_eq.size:
            v = max(v, float(np.max(np.abs(self.A_eq @ x - self.b_eq))))
        if self.b_ub.size:
            v = max(v, float(np.max(self.A_ub @ x - self.b_ub)))
        fin_l = np.isfinite(self.lb)
        fin_u = np.isfinite(self.ub)
        if fin_l.any():
            v = max(v, float(np.max(self.lb[fin_l] - x[fin_l])))
        if fin_u.any():
            v = max(v, float(np.max(x[fin_u] - self.ub[fin_u])))
        return max(v, 0.0)


@dataclass
class Reduction:
    """Affine map ``x = x0 + T z`` together with the standard-form problem."""

    std: StandardForm
    x0: np.ndarray
    T: np.ndarray
    n_ub: int

    def recover(self, z: np.ndarray) -> np.ndarray:
        return self.x0 + self.T @ z[: self.T.shape[1]]


def to_standard_form(p: GeneralProblem) -> Reduction:
    """Reduce a general problem to standard form.

    Variables with a finite lower bound are shifted, upper-bounded-only variables
    are reflected, free variables are kept as sign-unrestricted columns (splitting
    them into positive and negative parts makes the iterates drift to infinity),
    finite upper bounds become inequality rows, and every inequality receives a
    slack variable.
    """
    n = p.n
    cols = []
    free_cols = []
    x0 = np.zeros(n)
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1.0
        if np.isfinite(p.lb[i]):
            x0[i] = p.lb[i]
            cols.append(e)
            free_cols.append(False)
        elif np.isfinite(p.ub[i]):
            x0[i] = p.ub[i]
            cols.append(-e)
            free_cols.append(False)
        else:
            cols.append(e)
            free_cols.append(True)
    T = np.column_stack(cols)
    # finite upper bounds on variables that also have a finite lower bound
    extra_rows, extra_rhs = [], []
    for i in range(n):
        if np.isfinite(p.lb[i]) and np.isfinite(p.ub[i]):
            row = np.zeros(n)
            row[i] = 1.0
            extra_rows.append(row)
            extra_rhs.append(p.ub[i])
    A_ub = np.vstack([p.A_ub] + [np.array(extra_rows)] * bool(extra_rows)) if extra_rows else p.A_ub
    b_ub = np.concatenate([p.b_ub, extra_rhs]) if extra_rows else p.b_ub
    k = T.shape[1]
    m_ub = b_ub.size
    Qz = T.T @ p.Q @ T
    cz = T.T @ (p.c + p.Q @ x0)
    A = np.zeros((p.b_eq.size + m_ub, k + m_ub))
    b = np.zeros(p.b_eq.size + m_ub)
    A[: p.b_eq.size, :k] = p.A_eq @ T
    b[: p.b_eq.size] = p.b_eq - p.A_eq @ x0
    A[p.b_eq.size:, :k] = A_ub @ T
    A[p.b_eq.size:, k:] = np.eye(m_ub)
    b[p.b_eq.size:] = b_ub - A_ub @ x0
    Q = np.zeros((k + m_ub, k + m_ub))
    Q[:k, :k] = Qz
    c = np.concatenate([cz, np.zeros(m_ub)])
    free = np.concatenate([free_cols, np.zeros(m_ub, dtype=bool)]).astype(bool)
    return Reduction(StandardForm(Q, c, A, b, free), x0, T, m_ub)


def random_feasible_lp(m: int, n: int, seed: int = 0) -> GeneralProblem:
    """Random bounded LP ``min c'x, Ax=b, x>=0`` with a known interior point."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, n))
    x_int = rng.uniform(0.5, 2.0, n)
    b = A @ x_int
    y = rng.standard_normal(m)
    s = rng.uniform(0.5, 2.0, n)
    c = A.T @ y + s  # guarantees dual feasibility -> bounded
    return GeneralProblem(c=c, A_eq=A, b_eq=b)


def random_convex_qp(m: int, n: int, seed: int = 0) -> GeneralProblem:
    """Random strictly convex QP with equality constraints and x >= 0."""
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    Q = M @ M.T / n + 0.1 * np.eye(n)
    A = rng.standard_normal((m, n))
    x_int = rng.uniform(0.5, 2.0, n)
    return GeneralProblem(c=rng.standard_normal(n), Q=Q, A_eq=A, b_eq=A @ x_int)
