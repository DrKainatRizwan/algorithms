"""Bertsekas' auction algorithm with epsilon scaling.

Persons (rows) bid for objects (columns).  A bid raises the price of the best
object by ``best - second_best + eps``; the outbid person returns to the pool.
On termination the assignment is *eps-complementary-slack*, so its benefit is
within ``n * eps`` of optimal.  With integer data and ``eps < 1/n`` the result
is exactly optimal.  Epsilon scaling re-runs the auction with geometrically
shrinking ``eps`` while keeping prices, which greatly reduces the number of bids.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hungarian import Assignment
from .utils import validate_cost_matrix


@dataclass
class AuctionStats:
    """Counters describing one auction run."""

    bids: int = 0
    phases: int = 0


def _auction_square(
    benefit: np.ndarray, eps_final: float, scale: float, stats: AuctionStats
) -> tuple[np.ndarray, np.ndarray]:
    """Maximise benefit on a square matrix; returns (object_of_person, prices)."""
    n = benefit.shape[0]
    prices = np.zeros(n)
    if n == 1:
        stats.phases = 1
        return np.zeros(1, dtype=int), prices
    spread = float(benefit.max() - benefit.min())
    eps = max(spread / 2.0, eps_final)
    while True:
        stats.phases += 1
        owner = np.full(n, -1, dtype=int)  # object -> person
        assigned = np.full(n, -1, dtype=int)  # person -> object
        unassigned = list(range(n - 1, -1, -1))
        while unassigned:
            i = unassigned.pop()
            values = benefit[i] - prices
            j = int(np.argmax(values))
            best = values[j]
            values[j] = -np.inf
            second = values.max()
            prices[j] += best - second + eps
            stats.bids += 1
            prev = owner[j]
            if prev >= 0:
                assigned[prev] = -1
                unassigned.append(prev)
            owner[j] = i
            assigned[i] = j
        if eps <= eps_final:
            return assigned, prices
        eps = max(eps / scale, eps_final)


def auction_assignment(
    cost: np.ndarray | list,
    maximize: bool = False,
    eps_final: float | None = None,
    scale: float = 5.0,
    return_stats: bool = False,
):
    """Solve an assignment problem with the auction algorithm.

    Parameters
    ----------
    cost:
        ``n x m`` matrix with finite entries; the smaller side is matched fully.
    eps_final:
        Terminal bid increment.  Defaults to ``1 / (n + 1)`` which is exact for
        integer-valued matrices; for real data the result is within
        ``n * eps_final`` of optimal.
    scale:
        Epsilon reduction factor between phases (must exceed 1).
    """
    a = validate_cost_matrix(cost)
    if not np.isfinite(a).all():
        raise ValueError("auction requires finite costs; use hungarian for forbidden pairs")
    if scale <= 1.0:
        raise ValueError("scale must be greater than 1")
    benefit = a if maximize else -a
    transposed = benefit.shape[0] > benefit.shape[1]
    if transposed:
        benefit = benefit.T
    n, m = benefit.shape
    if eps_final is None:
        eps_final = 1.0 / (m + 1)
    if eps_final <= 0:
        raise ValueError("eps_final must be positive")
    # Pad with dummy persons whose benefit is zero everywhere; they absorb the
    # surplus objects without changing the real persons' optimum.
    padded = np.zeros((m, m))
    padded[:n] = benefit
    stats = AuctionStats()
    assigned, prices = _auction_square(padded, eps_final, scale, stats)
    rows = np.arange(n)
    cols = assigned[:n]
    # Persons' utilities as dual variables for the minimisation formulation.
    util = (padded - prices).max(axis=1)
    u_b, v_b = util[:n], prices
    if transposed:
        rows, cols = cols, rows
    order = np.argsort(rows)
    rows, cols = rows[order], cols[order]
    total = float(a[rows, cols].sum())
    # benefit duals (pi_i + p_j >= b_ij) -> cost duals (u + v <= c)
    if maximize:
        u, v = u_b, v_b
    else:
        u, v = -u_b, -v_b
    if transposed:
        u, v = v, u
    result = Assignment(rows=rows, cols=cols, cost=total, u=u, v=v)
    return (result, stats) if return_stats else result
