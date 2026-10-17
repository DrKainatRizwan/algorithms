"""Hungarian algorithm (Kuhn-Munkres) with dual potentials, O(n^2 m) time.

The implementation is the shortest-augmenting-path formulation: rows are added
one at a time and a Dijkstra-like search over reduced costs finds the cheapest
alternating path to a free column.  Row potentials ``u`` and column potentials
``v`` are maintained so that ``u[i] + v[j] <= cost[i, j]`` always holds, with
equality on matched pairs, which gives an optimality certificate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .utils import validate_cost_matrix


class InfeasibleError(ValueError):
    """Raised when every complete assignment uses a forbidden (infinite) entry."""


@dataclass
class Assignment:
    """Result of an assignment solve.

    ``rows[k]`` is matched with ``cols[k]``; ``u`` and ``v`` are dual potentials
    for the (possibly transposed, sign-flipped) minimisation problem on the
    original orientation, so ``u[i] + v[j] <= cost[i, j]`` for minimisation.
    """

    rows: np.ndarray
    cols: np.ndarray
    cost: float
    u: np.ndarray
    v: np.ndarray

    def as_dict(self) -> dict[int, int]:
        return {int(r): int(c) for r, c in zip(self.rows, self.cols)}


def _core(a: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve min-cost matching of all ``n`` rows into ``m >= n`` columns.

    Returns ``(col_of_row, u, v)`` with zero-based indices.
    """
    n, m = a.shape
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=int)  # p[j] = row matched to column j (1-based)
    way = np.zeros(m + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            cur = a[i0 - 1] - u[i0] - v[1:]
            free = ~used[1:]
            upd = free & (cur < minv[1:])
            minv[1:][upd] = cur[upd]
            way[1:][upd] = j0
            cand = np.where(free, minv[1:], np.inf)
            j1 = int(np.argmin(cand)) + 1
            delta = cand[j1 - 1]
            u[p[used]] += delta
            v[used] -= delta
            minv[~used] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    col_of_row = np.empty(n, dtype=int)
    for j in range(1, m + 1):
        if p[j]:
            col_of_row[p[j] - 1] = j - 1
    return col_of_row, u[1:], v[1:]


def hungarian(cost: np.ndarray | list, maximize: bool = False) -> Assignment:
    """Optimal assignment for a rectangular cost matrix.

    The smaller side is matched completely.  Entries equal to ``+inf`` are
    forbidden; :class:`InfeasibleError` is raised if no finite solution exists.
    With ``maximize=True`` the total is maximised (``-inf`` then marks a
    forbidden pair is *not* supported; pass ``+inf``-free benefit matrices).
    """
    a = validate_cost_matrix(cost)
    work = -a if maximize else a.copy()
    forbidden = np.isposinf(a)
    if maximize and forbidden.any():
        raise ValueError("forbidden entries are only supported when minimising")
    finite = work[~forbidden]
    big = (np.abs(finite).sum() + 1.0) * 2.0 if finite.size else 1.0
    work[forbidden] = big
    transposed = work.shape[0] > work.shape[1]
    if transposed:
        work = work.T
    col_of_row, u, v = _core(work)
    n = work.shape[0]
    rows = np.arange(n)
    cols = col_of_row
    if transposed:
        rows, cols = cols, rows
        u, v = v, u
    order = np.argsort(rows)
    rows, cols = rows[order], cols[order]
    if forbidden[rows, cols].any():
        raise InfeasibleError("no assignment avoids all forbidden entries")
    total = float(a[rows, cols].sum())
    if maximize:
        u, v = -u, -v
    return Assignment(rows=rows, cols=cols, cost=total, u=u, v=v)
