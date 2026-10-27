"""Knapsack family: 0/1, unbounded, bounded (binary splitting) and bitset subset-sum."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import numpy as np


@dataclass
class KnapsackResult:
    """Optimal value together with the multiplicity chosen for each item."""

    value: int
    counts: list[int]

    def weight(self, weights: Sequence[int]) -> int:
        return sum(c * w for c, w in zip(self.counts, weights))


def _check(weights: Sequence[int], values: Sequence[int], capacity: int) -> None:
    if len(weights) != len(values):
        raise ValueError("weights and values must have equal length")
    if capacity < 0 or any(w < 0 for w in weights):
        raise ValueError("capacity and weights must be non-negative")


def knapsack_01(weights: Sequence[int], values: Sequence[int], capacity: int) -> KnapsackResult:
    """Classic 0/1 knapsack, O(nW) time and O(nW) space (boolean take-table for reconstruction).

    The value row is rolled in one dimension; a ``take`` bit-matrix records decisions.
    """
    _check(weights, values, capacity)
    n = len(weights)
    best = np.zeros(capacity + 1, dtype=np.int64)
    take = np.zeros((n, capacity + 1), dtype=bool)
    for i in range(n):
        w, v = weights[i], values[i]
        if w > capacity:
            continue
        if w == 0:
            if v > 0:
                take[i, :] = True
                best += v
            continue
        cand = best[: capacity + 1 - w] + v
        better = cand > best[w:]
        take[i, w:] = better
        best[w:] = np.where(better, cand, best[w:])
    counts = [0] * n
    c = capacity
    for i in range(n - 1, -1, -1):
        if take[i, c]:
            counts[i] = 1
            c -= weights[i]
    return KnapsackResult(int(best[capacity]), counts)


def knapsack_01_bruteforce(weights: Sequence[int], values: Sequence[int], capacity: int) -> int:
    """Exponential reference solver used to validate the DP on small instances."""
    n = len(weights)
    best = 0
    for r in range(n + 1):
        for sub in combinations(range(n), r):
            if sum(weights[i] for i in sub) <= capacity:
                best = max(best, sum(values[i] for i in sub))
    return best


def knapsack_unbounded(weights: Sequence[int], values: Sequence[int], capacity: int) -> KnapsackResult:
    """Unbounded knapsack with reconstruction, O(nW) time, O(W) space.

    Zero-weight items with positive value make the problem unbounded and raise ``ValueError``.
    """
    _check(weights, values, capacity)
    if any(w == 0 and v > 0 for w, v in zip(weights, values)):
        raise ValueError("unbounded: zero-weight item with positive value")
    best = [0] * (capacity + 1)
    choice = [-1] * (capacity + 1)
    for c in range(1, capacity + 1):
        best[c], choice[c] = best[c - 1], -2  # -2: carry over slack from c-1
        for i, (w, v) in enumerate(zip(weights, values)):
            if 0 < w <= c and best[c - w] + v > best[c]:
                best[c], choice[c] = best[c - w] + v, i
    counts = [0] * len(weights)
    c = capacity
    while c > 0:
        if choice[c] == -2:
            c -= 1
        else:
            counts[choice[c]] += 1
            c -= weights[choice[c]]
    return KnapsackResult(best[capacity], counts)


def knapsack_bounded(
    weights: Sequence[int], values: Sequence[int], limits: Sequence[int], capacity: int
) -> KnapsackResult:
    """Bounded knapsack via binary splitting of each multiplicity into O(log k) 0/1 items."""
    _check(weights, values, capacity)
    if len(limits) != len(weights) or any(k < 0 for k in limits):
        raise ValueError("limits must match items and be non-negative")
    pieces: list[tuple[int, int, int]] = []  # (item index, multiplicity, ...)
    for i, k in enumerate(limits):
        m = 1
        while k > 0:
            take = min(m, k)
            pieces.append((i, take, 0))
            k -= take
            m <<= 1
    pw = [weights[i] * t for i, t, _ in pieces]
    pv = [values[i] * t for i, t, _ in pieces]
    res = knapsack_01(pw, pv, capacity)
    counts = [0] * len(weights)
    for flag, (i, t, _) in zip(res.counts, pieces):
        if flag:
            counts[i] += t
    return KnapsackResult(res.value, counts)


def subset_sum_bitset(nums: Sequence[int], target: int) -> list[int] | None:
    """Subset-sum using Python big-int bitsets: O(n * target / 64) word operations.

    Returns indices of a subset summing to ``target`` or ``None``. Reconstruction keeps the
    bitset after each prefix and walks backwards.
    """
    if target < 0 or any(x < 0 for x in nums):
        raise ValueError("non-negative inputs required")
    mask = (1 << (target + 1)) - 1
    states = [1]
    for x in nums:
        states.append((states[-1] | (states[-1] << x)) & mask)
    if not (states[-1] >> target) & 1:
        return None
    out: list[int] = []
    t = target
    for i in range(len(nums) - 1, -1, -1):
        if (states[i] >> t) & 1:
            continue  # reachable without item i
        out.append(i)
        t -= nums[i]
    return sorted(out)
