"""String analyses built on the suffix array / LCP and the suffix automaton."""
from __future__ import annotations

from .suffix_array import build_suffix_array, kasai_lcp
from .suffix_automaton import SuffixAutomaton


def count_distinct_substrings(s: str) -> int:
    """Distinct non-empty substrings: n(n+1)/2 - sum(LCP)."""
    n = len(s)
    sa = build_suffix_array(s)
    return n * (n + 1) // 2 - sum(kasai_lcp(s, sa))


def longest_repeated_substring(s: str) -> str:
    """Longest substring occurring at least twice (leftmost among ties by SA order)."""
    if not s:
        return ""
    sa = build_suffix_array(s)
    lcp = kasai_lcp(s, sa)
    i = max(range(len(s)), key=lambda j: lcp[j])
    return s[sa[i]:sa[i] + lcp[i]]


def longest_common_substring(a: str, b: str) -> str:
    """Longest common substring through a suffix automaton of ``a``."""
    return SuffixAutomaton(a).longest_common_substring(b)


def longest_common_substring_sa(a: str, b: str) -> str:
    """Longest common substring via suffix array of ``a + sep + b``."""
    sep = chr(max(map(ord, a + b), default=0) + 1)
    t = a + sep + b
    sa = build_suffix_array(t)
    lcp = kasai_lcp(t, sa)
    la = len(a)
    best, pos = 0, 0
    for i in range(1, len(sa)):
        x, y = sa[i - 1], sa[i]
        if (x < la) != (y < la) and x != la and y != la and lcp[i] > best:
            best, pos = lcp[i], x
    return t[pos:pos + best]


def kth_substring(s: str, k: int) -> str:
    """k-th (1-based) distinct substring in lexicographic order."""
    sa = build_suffix_array(s)
    lcp = kasai_lcp(s, sa)
    for i, p in enumerate(sa):
        new = len(s) - p - lcp[i]
        if k <= new:
            return s[p:p + lcp[i] + k]
        k -= new
    raise IndexError("k exceeds number of distinct substrings")
