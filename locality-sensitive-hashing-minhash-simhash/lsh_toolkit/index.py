"""Banded MinHash index and multi-table vector LSH index."""
from __future__ import annotations

from collections import defaultdict
from typing import Callable, Hashable

import numpy as np

from .minhash import MinHasher
from .hyperplane import HyperplaneHasher, PStableHasher


def collision_probability(s: float, bands: int, rows: int) -> float:
    """S-curve: probability two sets of similarity ``s`` share a bucket, ``1-(1-s^r)^b``."""
    return 1.0 - (1.0 - s ** rows) ** bands


def optimal_bands(num_perm: int, threshold: float, fp_weight: float = 0.5,
                  fn_weight: float = 0.5) -> tuple[int, int]:
    """Choose ``(bands, rows)`` with ``bands*rows <= num_perm`` minimising weighted
    false-positive area (below ``threshold``) and false-negative area (above)."""
    xs = np.linspace(0, 1, 501)

    def err(b: int, r: int) -> float:
        p = 1 - (1 - xs ** r) ** b
        lo, hi = xs < threshold, xs >= threshold
        fp = np.trapezoid(p[lo], xs[lo]) if lo.sum() > 1 else 0.0
        fn = np.trapezoid(1 - p[hi], xs[hi]) if hi.sum() > 1 else 0.0
        return fp_weight * fp + fn_weight * fn

    best, best_err = (1, 1), float("inf")
    for b in range(1, num_perm + 1):
        for r in range(1, num_perm // b + 1):
            e = err(b, r)
            if e < best_err - 1e-12:
                best, best_err = (b, r), e
    return best


class MinHashLSHIndex:
    """Jaccard-similarity search with banded MinHash signatures."""

    def __init__(self, hasher: MinHasher, bands: int, rows: int) -> None:
        if bands * rows > hasher.num_perm:
            raise ValueError("bands*rows exceeds number of permutations")
        self.hasher, self.bands, self.rows = hasher, bands, rows
        self._tables: list[dict[bytes, list[Hashable]]] = [defaultdict(list) for _ in range(bands)]
        self._sigs: dict[Hashable, np.ndarray] = {}

    @classmethod
    def for_threshold(cls, hasher: MinHasher, threshold: float) -> "MinHashLSHIndex":
        b, r = optimal_bands(hasher.num_perm, threshold)
        return cls(hasher, b, r)

    def _band_keys(self, sig: np.ndarray) -> list[bytes]:
        return [sig[i * self.rows:(i + 1) * self.rows].tobytes() for i in range(self.bands)]

    def insert(self, key: Hashable, items) -> None:
        if key in self._sigs:
            raise KeyError(f"duplicate key {key!r}")
        sig = self.hasher.signature(items)
        self._sigs[key] = sig
        for table, bk in zip(self._tables, self._band_keys(sig)):
            table[bk].append(key)

    def candidates(self, items) -> set[Hashable]:
        sig = self.hasher.signature(items)
        out: set[Hashable] = set()
        for table, bk in zip(self._tables, self._band_keys(sig)):
            out.update(table.get(bk, ()))
        return out

    def query(self, items, min_similarity: float = 0.0) -> list[tuple[Hashable, float]]:
        """Candidates re-ranked by estimated Jaccard, filtered by ``min_similarity``."""
        sig = self.hasher.signature(items)
        cands: set[Hashable] = set()
        for table, bk in zip(self._tables, self._band_keys(sig)):
            cands.update(table.get(bk, ()))
        scored = [(k, MinHasher.estimate(sig, self._sigs[k])) for k in cands]
        return sorted([p for p in scored if p[1] >= min_similarity], key=lambda p: (-p[1], str(p[0])))

    def __len__(self) -> int:
        return len(self._sigs)


class VectorLSHIndex:
    """Multi-table LSH for vectors with exact re-ranking of candidates.

    ``metric='cosine'`` uses random hyperplanes; ``'l2'`` uses p-stable hashes.
    """

    def __init__(self, dim: int, num_tables: int = 10, hashes_per_table: int = 12,
                 metric: str = "cosine", width: float = 4.0, seed: int = 0) -> None:
        if metric not in ("cosine", "l2"):
            raise ValueError("metric must be 'cosine' or 'l2'")
        self.metric = metric
        self.hashers: list[HyperplaneHasher | PStableHasher] = []
        for t in range(num_tables):
            if metric == "cosine":
                self.hashers.append(HyperplaneHasher(dim, hashes_per_table, seed=seed * 1000 + t))
            else:
                self.hashers.append(PStableHasher(dim, hashes_per_table, width, seed=seed * 1000 + t))
        self._tables: list[dict[bytes, list[int]]] = [defaultdict(list) for _ in self.hashers]
        self._data = np.empty((0, dim))

    def fit(self, data: np.ndarray) -> "VectorLSHIndex":
        """Index all rows of ``data`` (replaces any previous contents)."""
        self._data = np.asarray(data, dtype=float)
        self._tables = [defaultdict(list) for _ in self.hashers]
        for h, table in zip(self.hashers, self._tables):
            for i, k in enumerate(h.keys(self._data)):
                table[k].append(i)
        return self

    def candidates(self, q: np.ndarray) -> np.ndarray:
        out: set[int] = set()
        for h, table in zip(self.hashers, self._tables):
            out.update(table.get(h.keys(q), ()))
        return np.array(sorted(out), dtype=int)

    def _dist(self, rows: np.ndarray, q: np.ndarray) -> np.ndarray:
        if self.metric == "l2":
            return np.linalg.norm(rows - q, axis=1)
        denom = np.linalg.norm(rows, axis=1) * np.linalg.norm(q) + 1e-12
        return 1.0 - rows @ q / denom

    def query(self, q: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray, int]:
        """Return ``(ids, distances, n_candidates)`` of approximate k nearest neighbours."""
        cand = self.candidates(q)
        if cand.size == 0:
            return cand, np.empty(0), 0
        d = self._dist(self._data[cand], q)
        order = np.lexsort((cand, d))[:k]
        return cand[order], d[order], int(cand.size)
