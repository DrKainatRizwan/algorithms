"""Extended Isolation Forest (Hariri, Kind & Brunner, 2019): random hyperplane cuts."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .forest import IsolationForest
from .tree import average_path_length


@dataclass
class _HNode:
    normal: np.ndarray | None = None
    intercept: float = 0.0
    left: "_HNode | None" = None
    right: "_HNode | None" = None
    size: int = 0


class ExtendedIsolationTree:
    """Isolation tree whose splits are ``(x - p) . n < 0`` with random slope n and point p."""

    def __init__(self, max_depth: int, rng: np.random.Generator, extension_level: int | None = None) -> None:
        self.max_depth = max_depth
        self.rng = rng
        self.extension_level = extension_level
        self.root: _HNode | None = None

    def fit(self, X: np.ndarray) -> "ExtendedIsolationTree":
        d = X.shape[1]
        if self.extension_level is None:
            self.extension_level = d - 1
        if not 0 <= self.extension_level <= d - 1:
            raise ValueError("extension_level must be in [0, d-1]")
        self.root = self._grow(X, 0)
        return self

    def _grow(self, X: np.ndarray, depth: int) -> _HNode:
        n, d = X.shape
        if depth >= self.max_depth or n <= 1:
            return _HNode(size=n)
        lo, hi = X.min(axis=0), X.max(axis=0)
        if not (hi > lo).any():
            return _HNode(size=n)
        normal = self.rng.normal(size=d)
        # zero out d - 1 - extension_level coordinates
        n_zero = d - 1 - self.extension_level
        if n_zero > 0:
            normal[self.rng.choice(d, size=n_zero, replace=False)] = 0.0
        point = self.rng.uniform(lo, hi)
        proj = (X - point) @ normal
        mask = proj < 0
        if mask.all() or not mask.any():
            return _HNode(size=n)
        node = _HNode(normal=normal, intercept=float(point @ normal), size=n)
        node.left = self._grow(X[mask], depth + 1)
        node.right = self._grow(X[~mask], depth + 1)
        return node

    def path_lengths(self, X: np.ndarray) -> np.ndarray:
        out = np.zeros(X.shape[0])
        stack = [(self.root, np.arange(X.shape[0]), 0)]
        while stack:
            node, idx, depth = stack.pop()
            if idx.size == 0:
                continue
            if node.left is None:
                out[idx] = depth + average_path_length(node.size)
                continue
            go_left = X[idx] @ node.normal < node.intercept
            stack.append((node.left, idx[go_left], depth + 1))
            stack.append((node.right, idx[~go_left], depth + 1))
        return out


class ExtendedIsolationForest(IsolationForest):
    """Isolation forest using oblique splits; removes the axis-aligned score artefacts."""

    def __init__(self, n_trees: int = 100, max_samples: int = 256, contamination: float = 0.05,
                 seed: int = 0, extension_level: int | None = None) -> None:
        super().__init__(n_trees, max_samples, contamination, seed)
        self.extension_level = extension_level

    def _make_tree(self, max_depth: int, rng: np.random.Generator):
        return ExtendedIsolationTree(max_depth, rng, self.extension_level)

    def feature_importance(self, X: np.ndarray) -> np.ndarray:
        """Mean absolute normalised hyperplane coefficient, weighted by 1/(depth+1)."""
        d = np.asarray(X).shape[1]
        imp = np.zeros(d)
        for tree in self.trees:
            stack = [(tree.root, 0)]
            while stack:
                node, depth = stack.pop()
                if node is None or node.left is None:
                    continue
                imp += np.abs(node.normal) / (np.linalg.norm(node.normal) + 1e-12) / (depth + 1)
                stack += [(node.left, depth + 1), (node.right, depth + 1)]
        s = imp.sum()
        return imp / s if s > 0 else imp
