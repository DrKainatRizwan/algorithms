"""Rabin-Karp matching with a polynomial rolling hash (single and multi-pattern)."""

from __future__ import annotations

from collections import defaultdict

MOD = (1 << 61) - 1
BASE_DEFAULT = 911_382_323


class RollingHash:
    """Polynomial hash over Z_p with prefix hashes for O(1) substring hashes."""

    def __init__(self, s: str, base: int = BASE_DEFAULT, mod: int = MOD) -> None:
        self.base, self.mod = base, mod
        n = len(s)
        self.prefix = [0] * (n + 1)
        self.power = [1] * (n + 1)
        for i, ch in enumerate(s):
            self.prefix[i + 1] = (self.prefix[i] * base + ord(ch)) % mod
            self.power[i + 1] = self.power[i] * base % mod

    def get(self, lo: int, hi: int) -> int:
        """Hash of ``s[lo:hi]``."""
        return (self.prefix[hi] - self.prefix[lo] * self.power[hi - lo]) % self.mod


def _hash(s: str, base: int, mod: int) -> int:
    h = 0
    for ch in s:
        h = (h * base + ord(ch)) % mod
    return h


def rabin_karp(text: str, pattern: str, base: int = BASE_DEFAULT,
               mod: int = MOD, verify: bool = True) -> list[int]:
    """Single-pattern Rabin-Karp; expected O(n + m). Hits are verified by default."""
    m, n = len(pattern), len(text)
    if m == 0:
        return list(range(n + 1))
    if m > n:
        return []
    hp = _hash(pattern, base, mod)
    ht = _hash(text[:m], base, mod)
    top = pow(base, m - 1, mod)
    out: list[int] = []
    for i in range(n - m + 1):
        if ht == hp and (not verify or text[i:i + m] == pattern):
            out.append(i)
        if i + m < n:
            ht = ((ht - ord(text[i]) * top) * base + ord(text[i + m])) % mod
    return out


def rabin_karp_multi(text: str, patterns: list[str], base: int = BASE_DEFAULT,
                     mod: int = MOD) -> dict[str, list[int]]:
    """Multi-pattern Rabin-Karp for patterns of arbitrary lengths.

    Patterns are bucketed by length; one rolling pass is made per distinct
    length, so the cost is O(n * L + total) where L is the number of distinct
    lengths. Empty patterns are ignored.
    """
    result: dict[str, list[int]] = {p: [] for p in patterns if p}
    by_len: dict[int, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))
    for p in result:
        by_len[len(p)][_hash(p, base, mod)].append(p)
    rh = RollingHash(text, base, mod)
    n = len(text)
    for m, table in by_len.items():
        for i in range(n - m + 1):
            cands = table.get(rh.get(i, i + m))
            if cands:
                chunk = text[i:i + m]
                for p in cands:
                    if p == chunk:
                        result[p].append(i)
    return result
