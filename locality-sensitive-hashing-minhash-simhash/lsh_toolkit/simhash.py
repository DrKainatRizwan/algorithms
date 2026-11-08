"""SimHash fingerprints (Charikar) and exact Hamming-ball lookup by block tables."""
from __future__ import annotations

from collections import defaultdict
from typing import Mapping

import numpy as np

from .shingles import fnv1a_64


def hamming(a: int, b: int) -> int:
    """Hamming distance between two integer fingerprints."""
    return (a ^ b).bit_count()


class SimHasher:
    """Weighted-token SimHash producing ``bits``-bit fingerprints (bits <= 64)."""

    def __init__(self, bits: int = 64) -> None:
        if not 1 <= bits <= 64:
            raise ValueError("bits must be in [1, 64]")
        self.bits = bits

    def fingerprint(self, features: Mapping[str, float] | list[str]) -> int:
        """Fingerprint of a bag of tokens (list) or a token->weight mapping."""
        if not isinstance(features, Mapping):
            counts: dict[str, float] = defaultdict(float)
            for t in features:
                counts[t] += 1.0
            features = counts
        acc = np.zeros(self.bits)
        for token, w in features.items():
            h = fnv1a_64(token)
            # decorrelate low-entropy hashes so every bit position is usable
            h = (h ^ (h >> 33)) * 0xFF51AFD7ED558CCD & ((1 << 64) - 1)
            h ^= h >> 33
            bits = np.array([(h >> i) & 1 for i in range(self.bits)], dtype=float)
            acc += w * (2 * bits - 1)
        fp = 0
        for i in range(self.bits):
            if acc[i] > 0:
                fp |= 1 << i
        return fp

    def text_fingerprint(self, text: str, k: int = 3) -> int:
        """Fingerprint of a document using word k-gram features."""
        words = text.lower().split()
        grams = [" ".join(words[i:i + k]) for i in range(max(1, len(words) - k + 1))]
        return self.fingerprint(grams)

    def similarity(self, a: int, b: int) -> float:
        """Estimated cosine-angle similarity ``1 - hamming/bits``."""
        return 1.0 - hamming(a, b) / self.bits


class SimHashIndex:
    """Find all stored fingerprints within Hamming distance ``k`` (exact).

    The fingerprint is split into ``k+1`` blocks. By the pigeonhole principle,
    any pair differing in at most ``k`` bits agrees exactly on at least one
    block, so per-block hash tables give complete candidate generation.
    """

    def __init__(self, bits: int = 64, k: int = 3) -> None:
        if k < 0 or k + 1 > bits:
            raise ValueError("need 0 <= k < bits")
        self.bits, self.k = bits, k
        edges = np.linspace(0, bits, k + 2).astype(int)
        self._blocks = [(int(edges[i]), int(edges[i + 1])) for i in range(k + 1)]
        self._tables: list[dict[int, list[int]]] = [defaultdict(list) for _ in self._blocks]
        self._fps: list[int] = []

    def _key(self, fp: int, blk: tuple[int, int]) -> int:
        lo, hi = blk
        return (fp >> lo) & ((1 << (hi - lo)) - 1)

    def add(self, fp: int) -> int:
        """Insert a fingerprint; returns its id."""
        idx = len(self._fps)
        self._fps.append(fp)
        for table, blk in zip(self._tables, self._blocks):
            table[self._key(fp, blk)].append(idx)
        return idx

    def query(self, fp: int) -> list[tuple[int, int]]:
        """All ``(id, distance)`` with distance <= k, sorted by distance then id."""
        cand: set[int] = set()
        for table, blk in zip(self._tables, self._blocks):
            cand.update(table.get(self._key(fp, blk), ()))
        out = [(i, hamming(fp, self._fps[i])) for i in cand]
        return sorted((p for p in out if p[1] <= self.k), key=lambda p: (p[1], p[0]))

    def __len__(self) -> int:
        return len(self._fps)
