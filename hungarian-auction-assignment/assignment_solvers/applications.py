"""Assignment variants: bottleneck (min-max) matching and gated data association."""

from __future__ import annotations

import numpy as np

from .hungarian import Assignment, InfeasibleError, hungarian
from .utils import validate_cost_matrix


def _max_matching(allowed: np.ndarray) -> np.ndarray:
    """Maximum bipartite matching by augmenting paths (Kuhn); returns row of each column."""
    n, m = allowed.shape
    match_col = np.full(m, -1, dtype=int)

    def try_row(r: int, seen: np.ndarray) -> bool:
        for c in np.flatnonzero(allowed[r]):
            if seen[c]:
                continue
            seen[c] = True
            if match_col[c] < 0 or try_row(match_col[c], seen):
                match_col[c] = r
                return True
        return False

    for r in range(n):
        try_row(r, np.zeros(m, dtype=bool))
    return match_col


def bottleneck_assignment(cost: np.ndarray | list, tie_break: bool = True) -> Assignment:
    """Assignment minimising the *largest* matched cost.

    Binary search over the sorted distinct costs finds the smallest threshold
    that admits a perfect matching of the smaller side.  With ``tie_break`` the
    total cost is then minimised among assignments respecting that threshold.
    """
    a = validate_cost_matrix(cost)
    small = min(a.shape)
    values = np.unique(a[np.isfinite(a)])
    lo, hi = 0, len(values) - 1

    def feasible(t: float) -> bool:
        allowed = a <= t
        if a.shape[0] > a.shape[1]:
            allowed = allowed.T
        return int((_max_matching(allowed) >= 0).sum()) == small

    if values.size == 0 or not feasible(values[hi]):
        raise InfeasibleError("no complete assignment exists")
    while lo < hi:
        mid = (lo + hi) // 2
        if feasible(values[mid]):
            hi = mid
        else:
            lo = mid + 1
    threshold = values[lo]
    restricted = np.where(a <= threshold, a if tie_break else 0.0, np.inf)
    res = hungarian(restricted)
    res.cost = float(a[res.rows, res.cols].sum())
    return res


def gated_assignment(
    cost: np.ndarray | list, gate: float
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Track-to-detection association with a distance gate.

    Pairs with cost above ``gate`` are never matched.  Among gated solutions the
    one with the most matches wins, then the one with the least total cost.
    Returns ``(matches, unmatched_rows, unmatched_cols)``.
    """
    a = validate_cost_matrix(cost)
    n, m = a.shape
    ok = a <= gate
    penalty = (np.abs(a[np.isfinite(a)]).sum() + 1.0) * 2.0
    work = np.where(ok, a, penalty)
    res = hungarian(work)
    matches = [(int(r), int(c)) for r, c in zip(res.rows, res.cols) if ok[r, c]]
    used_r = {r for r, _ in matches}
    used_c = {c for _, c in matches}
    return (
        matches,
        [r for r in range(n) if r not in used_r],
        [c for c in range(m) if c not in used_c],
    )
