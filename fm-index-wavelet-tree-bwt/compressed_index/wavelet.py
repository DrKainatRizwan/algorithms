"""Balanced wavelet tree over a byte sequence (rank / select / access / quantile)."""
from __future__ import annotations
from .bitvector import BitVector


class _Node:
    __slots__ = ("bv", "left", "right", "lo", "hi")

    def __init__(self, lo: int, hi: int) -> None:
        self.lo, self.hi = lo, hi
        self.bv: BitVector | None = None
        self.left: _Node | None = None
        self.right: _Node | None = None


class WaveletTree:
    """Wavelet tree on the alphabet actually present in the sequence.

    Each node splits its alphabet interval in half and stores one bit per element;
    all queries cost O(log sigma) bit-vector operations.
    """

    def __init__(self, seq: bytes | list[int]) -> None:
        self.n = len(seq)
        self.alphabet = sorted(set(seq))
        self.root = self._build(list(seq), 0, len(self.alphabet) - 1) if self.alphabet else None
        self._index = {c: i for i, c in enumerate(self.alphabet)}

    def _build(self, seq: list[int], lo: int, hi: int) -> _Node:
        node = _Node(lo, hi)
        if lo == hi:
            return node
        mid = (lo + hi) // 2
        pivot = self.alphabet[mid]
        bits = [1 if c > pivot else 0 for c in seq]
        node.bv = BitVector(bits)
        node.left = self._build([c for c in seq if c <= pivot], lo, mid)
        node.right = self._build([c for c in seq if c > pivot], mid + 1, hi)
        return node

    def __len__(self) -> int:
        return self.n

    def access(self, i: int) -> int:
        """Symbol at position ``i``."""
        if not 0 <= i < self.n:
            raise IndexError(i)
        node = self.root
        while node.lo != node.hi:
            b = node.bv.access(i)
            i = node.bv.rank1(i) if b else node.bv.rank0(i)
            node = node.right if b else node.left
        return self.alphabet[node.lo]

    def rank(self, c: int, i: int) -> int:
        """Occurrences of symbol ``c`` in ``seq[:i]``."""
        if c not in self._index:
            return 0
        t = self._index[c]
        node = self.root
        while node.lo != node.hi:
            mid = (node.lo + node.hi) // 2
            if t > mid:
                i = node.bv.rank1(i)
                node = node.right
            else:
                i = node.bv.rank0(i)
                node = node.left
        return i

    def select(self, c: int, k: int) -> int:
        """Position of the ``k``-th (1-based) occurrence of ``c``."""
        if c not in self._index or k < 1 or k > self.rank(c, self.n):
            raise ValueError("select out of range")
        t = self._index[c]
        path = []
        node = self.root
        while node.lo != node.hi:
            mid = (node.lo + node.hi) // 2
            right = t > mid
            path.append((node, right))
            node = node.right if right else node.left
        pos = k - 1
        for nd, right in reversed(path):
            pos = nd.bv.select1(pos + 1) if right else nd.bv.select0(pos + 1)
        return pos

    def quantile(self, l: int, r: int, k: int) -> int:
        """The ``k``-th smallest (0-based) symbol in ``seq[l:r]``."""
        if not (0 <= l < r <= self.n and 0 <= k < r - l):
            raise ValueError("bad quantile arguments")
        node = self.root
        while node.lo != node.hi:
            zl, zr = node.bv.rank0(l), node.bv.rank0(r)
            if k < zr - zl:
                l, r, node = zl, zr, node.left
            else:
                k -= zr - zl
                l, r, node = l - zl, r - zr, node.right
        return self.alphabet[node.lo]
