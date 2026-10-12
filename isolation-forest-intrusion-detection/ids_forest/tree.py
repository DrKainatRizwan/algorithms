"""Axis-parallel isolation tree (Liu, Ting & Zhou, 2008)."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

EULER_GAMMA = 0.5772156649015329


def average_path_length(n: int | np.ndarray) -> np.ndarray | float:
    """Expected path length of an unsuccessful BST search over ``n`` points.

    c(n) = 2 H(n-1) - 2(n-1)/n with H(i) ~ ln(i) + gamma. c(1)=0, c(2)=1.
    """
    arr = np.asarray(n, dtype=float)
    out = np.zeros_like(arr)
    big = arr > 2
    out = np.where(arr == 2, 1.0, out)
    out = np.where(big, 2.0 * (np.log(np.where(big, arr, 3.0) - 1.0) + EULER_GAMMA)
                   - 2.0 * (np.where(big, arr, 3.0) - 1.0) / np.where(big, arr, 3.0), out)
    return float(out) if out.ndim == 0 else out


@dataclass
class _Node:
    feature: int = -1
    threshold: float = 0.0
    left: "_Node | None" = None
    right: "_Node | None" = None
    size: int = 0  # number of training points at an external node

    @property
    def is_leaf(self) -> bool:
        return self.left is None


class IsolationTree:
    """A single randomly-partitioned tree grown to ``max_depth``."""

    def __init__(self, max_depth: int, rng: np.random.Generator) -> None:
        self.max_depth = max_depth
        self.rng = rng
        self.root: _Node | None = None
        self.n_nodes = 0

    def fit(self, X: np.ndarray) -> "IsolationTree":
        self.n_nodes = 0
        self.root = self._grow(X, 0)
        return self

    def _grow(self, X: np.ndarray, depth: int) -> _Node:
        self.n_nodes += 1
        n = X.shape[0]
        if depth >= self.max_depth or n <= 1:
            return _Node(size=n)
        lo, hi = X.min(axis=0), X.max(axis=0)
        candidates = np.flatnonzero(hi > lo)
        if candidates.size == 0:
            return _Node(size=n)
        f = int(self.rng.choice(candidates))
        t = float(self.rng.uniform(lo[f], hi[f]))
        mask = X[:, f] < t
        if mask.all() or not mask.any():  # numerical edge case
            return _Node(size=n)
        node = _Node(feature=f, threshold=t, size=n)
        node.left = self._grow(X[mask], depth + 1)
        node.right = self._grow(X[~mask], depth + 1)
        return node

    def path_lengths(self, X: np.ndarray) -> np.ndarray:
        """Depth reached plus c(size) adjustment for every row of ``X``."""
        assert self.root is not None, "tree not fitted"
        out = np.zeros(X.shape[0])
        stack = [(self.root, np.arange(X.shape[0]), 0)]
        while stack:
            node, idx, depth = stack.pop()
            if idx.size == 0:
                continue
            if node.is_leaf:
                out[idx] = depth + average_path_length(node.size)
                continue
            go_left = X[idx, node.feature] < node.threshold
            stack.append((node.left, idx[go_left], depth + 1))
            stack.append((node.right, idx[~go_left], depth + 1))
        return out
