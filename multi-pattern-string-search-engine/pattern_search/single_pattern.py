"""Classic single-pattern matchers: naive, KMP and Z-algorithm."""

from __future__ import annotations


def naive_search(text: str, pattern: str) -> list[int]:
    """Return all (possibly overlapping) start offsets of ``pattern`` in ``text``.

    O(n*m) worst case; used as the reference oracle in tests.
    """
    if not pattern:
        return list(range(len(text) + 1))
    m = len(pattern)
    return [i for i in range(len(text) - m + 1) if text[i:i + m] == pattern]


def failure_function(pattern: str) -> list[int]:
    """Prefix function: ``pi[i]`` is the longest proper border of ``pattern[:i+1]``."""
    pi = [0] * len(pattern)
    k = 0
    for i in range(1, len(pattern)):
        while k and pattern[i] != pattern[k]:
            k = pi[k - 1]
        if pattern[i] == pattern[k]:
            k += 1
        pi[i] = k
    return pi


def kmp_search(text: str, pattern: str) -> list[int]:
    """Knuth-Morris-Pratt search in O(n + m) time and O(m) space."""
    if not pattern:
        return list(range(len(text) + 1))
    pi = failure_function(pattern)
    m = len(pattern)
    out: list[int] = []
    k = 0
    for i, ch in enumerate(text):
        while k and ch != pattern[k]:
            k = pi[k - 1]
        if ch == pattern[k]:
            k += 1
        if k == m:
            out.append(i - m + 1)
            k = pi[k - 1]
    return out


def z_array(s: str) -> list[int]:
    """Z-array: ``z[i]`` is the length of the longest common prefix of ``s`` and ``s[i:]``.

    By convention ``z[0] = len(s)``.
    """
    n = len(s)
    z = [0] * n
    if n == 0:
        return z
    z[0] = n
    l = r = 0
    for i in range(1, n):
        if i < r:
            z[i] = min(r - i, z[i - l])
        while i + z[i] < n and s[z[i]] == s[i + z[i]]:
            z[i] += 1
        if i + z[i] > r:
            l, r = i, i + z[i]
    return z


def z_search(text: str, pattern: str) -> list[int]:
    """Pattern matching by Z-array of ``pattern + sep + text`` in O(n + m)."""
    if not pattern:
        return list(range(len(text) + 1))
    m = len(pattern)
    # Use a sentinel outside both alphabets by working on code-point lists.
    seq = [ord(c) for c in pattern] + [-1] + [ord(c) for c in text]
    n = len(seq)
    z = [0] * n
    l = r = 0
    for i in range(1, n):
        if i < r:
            z[i] = min(r - i, z[i - l])
        while i + z[i] < n and seq[z[i]] == seq[i + z[i]]:
            z[i] += 1
        if i + z[i] > r:
            l, r = i, i + z[i]
    return [i - m - 1 for i in range(m + 1, n) if z[i] >= m]
