"""Suffix array construction (prefix doubling with radix sort) and Kasai LCP."""
from __future__ import annotations


def build_suffix_array_naive(s: str) -> list[int]:
    """Reference O(n^2 log n) construction by sorting all suffixes."""
    return sorted(range(len(s)), key=lambda i: s[i:])


def _counting_sort(keys: list[int], order: list[int], k: int) -> list[int]:
    """Stable counting sort of ``order`` by ``keys[i]`` in range [0, k)."""
    count = [0] * (k + 1)
    for i in order:
        count[keys[i] + 1] += 1
    for j in range(1, k + 1):
        count[j] += count[j - 1]
    out = [0] * len(order)
    for i in order:
        out[count[keys[i]]] = i
        count[keys[i]] += 1
    return out


def build_suffix_array(s: str) -> list[int]:
    """Build the suffix array by prefix doubling in O(n log n).

    Each round sorts suffixes by their first ``2k`` characters using two
    stable counting sorts (second key, then first key).
    """
    n = len(s)
    if n == 0:
        return []
    rank = [ord(c) for c in s]
    # compress initial ranks
    vals = sorted(set(rank))
    mapping = {v: i for i, v in enumerate(vals)}
    rank = [mapping[r] for r in rank]
    sa = _counting_sort(rank, list(range(n)), len(vals))
    k = 1
    classes = len(vals)
    while classes < n:
        # sort by second key: suffixes with i+k >= n come first
        second = list(range(max(n - k, 0), n)) + [p - k for p in sa if p >= k]
        sa = _counting_sort(rank, second, classes)
        new_rank = [0] * n
        r = 0
        for idx in range(1, n):
            a, b = sa[idx - 1], sa[idx]
            ka = (rank[a], rank[a + k] if a + k < n else -1)
            kb = (rank[b], rank[b + k] if b + k < n else -1)
            if ka != kb:
                r += 1
            new_rank[b] = r
        new_rank[sa[0]] = 0
        rank = new_rank
        classes = r + 1
        k <<= 1
    return sa


def inverse_suffix_array(sa: list[int]) -> list[int]:
    """Return ``rank`` where ``rank[sa[i]] == i``."""
    rank = [0] * len(sa)
    for i, p in enumerate(sa):
        rank[p] = i
    return rank


def kasai_lcp(s: str, sa: list[int]) -> list[int]:
    """Kasai's linear-time LCP array.

    ``lcp[i]`` is the longest common prefix of suffixes ``sa[i-1]`` and
    ``sa[i]``; ``lcp[0] == 0``.
    """
    n = len(s)
    rank = inverse_suffix_array(sa)
    lcp = [0] * n
    h = 0
    for i in range(n):
        if rank[i] > 0:
            j = sa[rank[i] - 1]
            while i + h < n and j + h < n and s[i + h] == s[j + h]:
                h += 1
            lcp[rank[i]] = h
            if h > 0:
                h -= 1
        else:
            h = 0
    return lcp
