"""Sequence DP: edit distance with scripts, LCS (full and Hirschberg), LIS, banded edit distance."""

from __future__ import annotations

from bisect import bisect_left
from typing import Sequence

import numpy as np

Script = list[tuple[str, int, str]]  # (op, position in source, char) ; op in {"sub","ins","del"}


def _matrix(a: str, b: str, ci: int, cd: int, cs: int) -> np.ndarray:
    n, m = len(a), len(b)
    d = np.zeros((n + 1, m + 1), dtype=np.int64)
    d[:, 0] = np.arange(n + 1) * cd
    d[0, :] = np.arange(m + 1) * ci
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = d[i - 1, j - 1] + (0 if a[i - 1] == b[j - 1] else cs)
            d[i, j] = min(sub, d[i - 1, j] + cd, d[i, j - 1] + ci)
    return d


def edit_distance(a: str, b: str, ins: int = 1, dele: int = 1, sub: int = 1) -> int:
    """Weighted Levenshtein distance in O(nm) time and O(min(n,m)) space."""
    if len(b) > len(a):
        a, b, ins, dele = b, a, dele, ins
    prev = [j * ins for j in range(len(b) + 1)]
    for i in range(1, len(a) + 1):
        cur = [i * dele] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cur[j] = min(prev[j - 1] + (0 if a[i - 1] == b[j - 1] else sub),
                         prev[j] + dele, cur[j - 1] + ins)
        prev = cur
    return prev[-1]


def edit_script(a: str, b: str) -> Script:
    """Unit-cost optimal edit script transforming ``a`` into ``b`` (traceback of the full table).

    Operations refer to positions in the *original* string ``a``.
    """
    d = _matrix(a, b, 1, 1, 1)
    i, j = len(a), len(b)
    ops: Script = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and a[i - 1] == b[j - 1] and d[i, j] == d[i - 1, j - 1]:
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + 1:
            ops.append(("sub", i - 1, b[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            ops.append(("del", i - 1, a[i - 1]))
            i -= 1
        else:
            ops.append(("ins", i, b[j - 1]))
            j -= 1
    ops.reverse()
    return ops


def apply_script(a: str, script: Script) -> str:
    """Apply an :func:`edit_script` to ``a`` and return the result."""
    out: list[str] = []
    k = 0
    for op, pos, ch in script:
        while k < pos:
            out.append(a[k])
            k += 1
        if op == "sub":
            out.append(ch)
            k += 1
        elif op == "del":
            k += 1
        else:  # ins: insert before a[pos]
            out.append(ch)
    out.extend(a[k:])
    return "".join(out)


def lcs(a: Sequence, b: Sequence) -> list:
    """Longest common subsequence, O(nm) time and space."""
    n, m = len(a), len(b)
    t = np.zeros((n + 1, m + 1), dtype=np.int32)
    for i in range(n):
        for j in range(m):
            t[i + 1, j + 1] = t[i, j] + 1 if a[i] == b[j] else max(t[i, j + 1], t[i + 1, j])
    out = []
    i, j = n, m
    while i and j:
        if a[i - 1] == b[j - 1]:
            out.append(a[i - 1])
            i, j = i - 1, j - 1
        elif t[i - 1, j] >= t[i, j - 1]:
            i -= 1
        else:
            j -= 1
    return out[::-1]


def _lcs_row(a: Sequence, b: Sequence) -> list[int]:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev


def lcs_hirschberg(a: Sequence, b: Sequence) -> list:
    """LCS in O(nm) time but only O(n+m) space via Hirschberg's divide and conquer."""
    if not a or not b:
        return []
    if len(a) == 1:
        return [a[0]] if a[0] in b else []
    mid = len(a) // 2
    left = _lcs_row(a[:mid], b)
    right = _lcs_row(a[mid:][::-1], b[::-1])
    m = len(b)
    k = max(range(m + 1), key=lambda j: left[j] + right[m - j])
    return lcs_hirschberg(a[:mid], b[:k]) + lcs_hirschberg(a[mid:], b[k:])


def lis(seq: Sequence[int]) -> list[int]:
    """Strictly increasing longest subsequence in O(n log n) by patience sorting."""
    tails: list[int] = []
    tail_idx: list[int] = []
    parent = [-1] * len(seq)
    for i, x in enumerate(seq):
        p = bisect_left(tails, x)
        if p == len(tails):
            tails.append(x)
            tail_idx.append(i)
        else:
            tails[p] = x
            tail_idx[p] = i
        parent[i] = tail_idx[p - 1] if p else -1
    out = []
    k = tail_idx[-1] if tail_idx else -1
    while k != -1:
        out.append(seq[k])
        k = parent[k]
    return out[::-1]


def banded_edit_distance(a: str, b: str, k: int) -> int | None:
    """Unit-cost edit distance restricted to a diagonal band; O(k * min(n,m)).

    Returns ``None`` if the distance exceeds ``k``.
    """
    n, m = len(a), len(b)
    if abs(n - m) > k:
        return None
    inf = k + 1
    prev = {j: j for j in range(0, min(m, k) + 1)}
    for i in range(1, n + 1):
        cur: dict[int, int] = {}
        for j in range(max(0, i - k), min(m, i + k) + 1):
            if j == 0:
                cur[j] = i
                continue
            best = prev.get(j - 1, inf) + (0 if a[i - 1] == b[j - 1] else 1)
            best = min(best, prev.get(j, inf) + 1, cur.get(j - 1, inf) + 1)
            cur[j] = min(best, inf)
        prev = cur
    d = prev.get(m, inf)
    return d if d <= k else None
