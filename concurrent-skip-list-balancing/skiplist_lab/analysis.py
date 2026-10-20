"""Probabilistic analysis of skip list balance (Pugh 1990).

With promotion probability p and n keys, let L(n) = log_{1/p}(n).  Then
 * expected maximum height  ~ L(n) + 1/(1-p),
 * expected search cost (steps of the backward path) ~ L(n)/p + 1/(1-p),
 * P(height > k) <= n * p**k  (union bound over towers).
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Sequence

from .levels import GeometricLevels
from .skiplist import SkipList


def expected_height(n: int, p: float = 0.5) -> float:
    """Approximate expected number of levels of an n-key skip list."""
    if n <= 0:
        return 0.0
    return math.log(n, 1.0 / p) + 1.0 / (1.0 - p)


def expected_search_cost(n: int, p: float = 0.5) -> float:
    """Pugh's bound on the expected number of steps of a search path."""
    if n <= 1:
        return 1.0
    return math.log(n, 1.0 / p) / p + 1.0 / (1.0 - p)


def height_tail_bound(n: int, p: float, k: int) -> float:
    """Union bound on P(max level > k)."""
    return min(1.0, n * p ** k)


def expected_pointers_per_node(p: float) -> float:
    """Expected tower height of a node, 1/(1-p)."""
    return 1.0 / (1.0 - p)


def optimal_p(n: int, grid: Sequence[float] | None = None) -> float:
    """p in ``grid`` minimising the expected search cost bound for n keys."""
    candidates = grid or [i / 100 for i in range(5, 96)]
    return min(candidates, key=lambda p: expected_search_cost(n, p))


def level_histogram(heights: Sequence[int]) -> dict[int, int]:
    """Count towers by height."""
    return dict(sorted(Counter(heights).items()))


def geometric_chi_square(heights: Sequence[int], p: float) -> float:
    """Chi-square statistic of tower heights against Geometric(p), sparse tail bins merged."""
    n = len(heights)
    counts = Counter(heights)
    top = max(counts)
    stat, tail_obs, tail_exp = 0.0, 0, 0.0
    for h in range(1, top + 1):
        exp = n * (1 - p) * p ** (h - 1)
        obs = counts.get(h, 0)
        if exp < 5:
            tail_obs += obs
            tail_exp += exp
        else:
            stat += (obs - exp) ** 2 / exp
    if tail_exp > 0:
        stat += (tail_obs - tail_exp) ** 2 / tail_exp
    return stat


def measure_search_cost(n: int, p: float = 0.5, queries: int = 2000, seed: int = 0) -> dict[str, float]:
    """Build an n-key list and report mean comparisons per successful search."""
    sl = SkipList(GeometricLevels(p=p, seed=seed))
    for k in range(n):
        sl.insert(k)
    sl.comparisons = 0
    step = max(1, n // queries)
    probes = list(range(0, n, step))
    for k in probes:
        sl.get(k)
    return {
        "n": n,
        "p": p,
        "height": sl.height,
        "mean_comparisons": sl.comparisons / len(probes),
        "predicted_steps": expected_search_cost(n, p),
        "predicted_height": expected_height(n, p),
    }
