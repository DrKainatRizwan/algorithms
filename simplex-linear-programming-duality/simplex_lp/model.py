"""Linear-program model and conversion to computational standard form.

A :class:`LinearProgram` is the user-facing description::

    min / max  c^T x
    s.t.       A_ub x <= b_ub
               A_eq x  = b_eq
               lo <= x <= hi

:meth:`LinearProgram.to_standard` rewrites it as ``min c'^T u, A' u = b', u >= 0``
through an affine change of variables ``x = shift + T u`` and slack columns,
and returns a :class:`StandardForm` that knows how to map solutions back.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

INF = float("inf")


def _as_matrix(a, n: int) -> np.ndarray:
    if a is None:
        return np.zeros((0, n))
    arr = np.atleast_2d(np.asarray(a, dtype=float))
    if arr.shape[1] != n:
        raise ValueError(f"constraint matrix has {arr.shape[1]} columns, expected {n}")
    return arr


def _as_vector(v, m: int, name: str) -> np.ndarray:
    if v is None:
        return np.zeros(0)
    arr = np.asarray(v, dtype=float).ravel()
    if arr.size != m:
        raise ValueError(f"{name} has length {arr.size}, expected {m}")
    return arr


@dataclass
class StandardForm:
    """``min c^T u  s.t.  A u = b, u >= 0`` plus the map back to original variables.

    ``x = shift + T @ u[:n_struct]``; the trailing columns of ``u`` are slacks.
    ``n_rows_ub``/``n_rows_eq`` give the number of original inequality and
    equality rows, which occupy the first rows of ``A`` in that order.
    """

    c: np.ndarray
    A: np.ndarray
    b: np.ndarray
    shift: np.ndarray
    T: np.ndarray
    offset: float
    sense: float
    n_rows_ub: int
    n_rows_eq: int
    n_struct: int
    slack_of_row: dict[int, int] = field(default_factory=dict)

    def recover(self, u: np.ndarray) -> np.ndarray:
        """Map a standard-form point back to the original variables."""
        return self.shift + self.T @ u[: self.n_struct]

    def objective(self, u: np.ndarray) -> float:
        """Objective of the *original* problem (respecting max/min)."""
        return self.sense * (float(self.c @ u) + self.offset)

    @property
    def n_orig_rows(self) -> int:
        return self.n_rows_ub + self.n_rows_eq


@dataclass
class LinearProgram:
    """General-form linear program."""

    c: np.ndarray
    A_ub: np.ndarray | None = None
    b_ub: np.ndarray | None = None
    A_eq: np.ndarray | None = None
    b_eq: np.ndarray | None = None
    bounds: list[tuple[float, float]] | None = None
    maximize: bool = False
    name: str = "lp"

    def __post_init__(self) -> None:
        self.c = np.asarray(self.c, dtype=float).ravel()
        n = self.c.size
        self.A_ub = _as_matrix(self.A_ub, n)
        self.A_eq = _as_matrix(self.A_eq, n)
        self.b_ub = _as_vector(self.b_ub, self.A_ub.shape[0], "b_ub")
        self.b_eq = _as_vector(self.b_eq, self.A_eq.shape[0], "b_eq")
        if self.bounds is None:
            self.bounds = [(0.0, INF)] * n
        if len(self.bounds) != n:
            raise ValueError("bounds must have one (lo, hi) pair per variable")
        for lo, hi in self.bounds:
            if lo > hi:
                raise ValueError(f"empty bound interval ({lo}, {hi})")

    @property
    def n_vars(self) -> int:
        return self.c.size

    def objective(self, x: np.ndarray) -> float:
        return float(self.c @ x)

    def max_violation(self, x: np.ndarray) -> float:
        """Largest violation of any constraint or bound at ``x``."""
        v = 0.0
        if self.A_ub.shape[0]:
            v = max(v, float(np.max(self.A_ub @ x - self.b_ub)))
        if self.A_eq.shape[0]:
            v = max(v, float(np.max(np.abs(self.A_eq @ x - self.b_eq))))
        for xj, (lo, hi) in zip(x, self.bounds):
            v = max(v, lo - xj, xj - hi)
        return v

    def is_feasible(self, x: np.ndarray, tol: float = 1e-7) -> bool:
        return self.max_violation(np.asarray(x, dtype=float)) <= tol

    def as_minimisation(self) -> "LinearProgram":
        """Equivalent minimisation problem (negated costs if maximising)."""
        if not self.maximize:
            return self
        return LinearProgram(-self.c, self.A_ub, self.b_ub, self.A_eq, self.b_eq,
                             list(self.bounds), False, self.name + "-min")

    def with_explicit_bounds(self) -> "LinearProgram":
        """Move every bound except ``x >= 0`` / free into explicit ``<=`` rows."""
        rows, rhs, bounds = [], [], []
        for j, (lo, hi) in enumerate(self.bounds):
            e = np.zeros(self.n_vars)
            e[j] = 1.0
            if np.isfinite(hi):
                rows.append(e)
                rhs.append(hi)
            if np.isfinite(lo) and lo != 0.0:
                rows.append(-e)
                rhs.append(-lo)
                bounds.append((-INF, INF))
            else:
                bounds.append((lo, INF) if lo == 0.0 else (-INF, INF))
        A_ub = np.vstack([self.A_ub, rows]) if rows else self.A_ub
        b_ub = np.concatenate([self.b_ub, rhs])
        return LinearProgram(self.c, A_ub, b_ub, self.A_eq, self.b_eq, bounds,
                             self.maximize, self.name)

    def to_standard(self) -> StandardForm:
        """Build the equivalent equality form with non-negative variables."""
        n = self.n_vars
        sense = -1.0 if self.maximize else 1.0
        c = sense * self.c
        cols: list[np.ndarray] = []
        shift = np.zeros(n)
        bound_rows: list[tuple[int, float]] = []  # (struct column, width)
        for j, (lo, hi) in enumerate(self.bounds):
            e = np.zeros(n)
            e[j] = 1.0
            if np.isfinite(lo):
                shift[j] = lo
                cols.append(e)
                if np.isfinite(hi):
                    bound_rows.append((len(cols) - 1, hi - lo))
            elif np.isfinite(hi):
                shift[j] = hi
                cols.append(-e)
            else:
                cols.append(e)
                cols.append(-e)
        T = np.array(cols).T if cols else np.zeros((n, 0))
        k = T.shape[1]
        m_ub, m_eq = self.A_ub.shape[0], self.A_eq.shape[0]
        m = m_ub + m_eq + len(bound_rows)
        A = np.zeros((m, k))
        b = np.zeros(m)
        A[:m_ub] = self.A_ub @ T
        b[:m_ub] = self.b_ub - self.A_ub @ shift
        A[m_ub:m_ub + m_eq] = self.A_eq @ T
        b[m_ub:m_ub + m_eq] = self.b_eq - self.A_eq @ shift
        for r, (col, width) in enumerate(bound_rows):
            A[m_ub + m_eq + r, col] = 1.0
            b[m_ub + m_eq + r] = width
        slack_rows = list(range(m_ub)) + list(range(m_ub + m_eq, m))
        S = np.zeros((m, len(slack_rows)))
        slack_of_row = {}
        for s, row in enumerate(slack_rows):
            S[row, s] = 1.0
            slack_of_row[row] = k + s
        A_std = np.hstack([A, S])
        c_std = np.concatenate([T.T @ c, np.zeros(len(slack_rows))])
        return StandardForm(c_std, A_std, b, shift, T, float(c @ shift), sense,
                            m_ub, m_eq, k, slack_of_row)
