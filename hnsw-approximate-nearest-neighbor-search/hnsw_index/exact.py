"""Exact brute-force k-NN, used as ground truth."""
from __future__ import annotations

import numpy as np

from .distance import get_metric


class BruteForceIndex:
    """Linear-scan index. Query cost O(n d), memory O(n d)."""

    def __init__(self, metric: str = "l2") -> None:
        self.metric_name = metric
        self._dist = get_metric(metric)
        self._data = np.empty((0, 0))

    def add(self, X: np.ndarray) -> None:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2-D")
        self._data = X.copy() if self._data.size == 0 else np.vstack([self._data, X])

    def __len__(self) -> int:
        return len(self._data)

    def query(self, q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (ids, distances) of the k nearest rows, sorted ascending."""
        d = self._dist(np.asarray(q, dtype=float), self._data)
        k = min(k, len(d))
        part = np.argpartition(d, k - 1)[:k]
        order = part[np.lexsort((part, d[part]))]
        return order, d[order]

    def batch_query(self, Q: np.ndarray, k: int) -> np.ndarray:
        return np.array([self.query(q, k)[0] for q in Q])
