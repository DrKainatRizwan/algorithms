"""Suffix array (prefix doubling), Kasai LCP and the Burrows-Wheeler transform."""
from __future__ import annotations

SENTINEL = 0  # byte value smaller than any text byte


def suffix_array(s: bytes | list[int]) -> list[int]:
    """Suffix array by Manber-Myers prefix doubling, O(n log^2 n)."""
    n = len(s)
    if n == 0:
        return []
    rank = list(s)
    sa = list(range(n))
    k = 1
    while True:
        key = lambda i: (rank[i], rank[i + k] if i + k < n else -1)
        sa.sort(key=key)
        new = [0] * n
        for j in range(1, n):
            new[sa[j]] = new[sa[j - 1]] + (key(sa[j]) != key(sa[j - 1]))
        rank = new
        if rank[sa[-1]] == n - 1:
            break
        k *= 2
    return sa


def lcp_kasai(s: bytes, sa: list[int]) -> list[int]:
    """LCP array: lcp[i] = common prefix of suffixes sa[i-1] and sa[i] (lcp[0]=0)."""
    n = len(s)
    rank = [0] * n
    for i, p in enumerate(sa):
        rank[p] = i
    lcp = [0] * n
    h = 0
    for i in range(n):
        if rank[i] > 0:
            j = sa[rank[i] - 1]
            while i + h < n and j + h < n and s[i + h] == s[j + h]:
                h += 1
            lcp[rank[i]] = h
            if h:
                h -= 1
        else:
            h = 0
    return lcp


def _terminated(text: bytes) -> bytes:
    if SENTINEL in text:
        raise ValueError("text must not contain the 0x00 byte (reserved sentinel)")
    return text + bytes([SENTINEL])


def bwt_from_sa(t: bytes, sa: list[int]) -> bytes:
    """BWT of sentinel-terminated ``t`` given its suffix array."""
    return bytes(t[i - 1] for i in sa)


def bwt(text: bytes) -> bytes:
    """Burrows-Wheeler transform of ``text`` (a 0x00 sentinel is appended)."""
    t = _terminated(text)
    return bwt_from_sa(t, suffix_array(t))


def inverse_bwt(last: bytes) -> bytes:
    """Invert a BWT via the LF-mapping; returns the original text without sentinel."""
    n = len(last)
    counts = [0] * 256
    for c in last:
        counts[c] += 1
    first = [0] * 256
    tot = 0
    for c in range(256):
        first[c] = tot
        tot += counts[c]
    seen = [0] * 256
    lf = [0] * n
    for i, c in enumerate(last):
        lf[i] = first[c] + seen[c]
        seen[c] += 1
    out = bytearray()
    i = 0  # row 0 is the sentinel-first suffix, i.e. the sentinel itself
    for _ in range(n - 1):
        out.append(last[i])
        i = lf[i]
    out.reverse()
    return bytes(out)
