"""Isolation Forest ensemble with anomaly scoring."""

from __future__ import annotations

import numpy as np

from .tree import IsolationTree, average_path_length


class IsolationForest:
    """Isolation Forest.

    Parameters
    ----------
    n_trees: number of isolation trees.
    max_samples: sub-sample size psi per tree (capped at n).
    contamination: expected anomaly fraction, used to set ``threshold_``.
    seed: seed of the random generator.
    """

    tree_class = IsolationTree

    def __init__(self, n_trees: int = 100, max_samples: int = 256,
                 contamination: float = 0.05, seed: int = 0) -> None:
        if n_trees < 1 or max_samples < 2:
            raise ValueError("n_trees >= 1 and max_samples >= 2 required")
        if not 0.0 < contamination < 0.5:
            raise ValueError("contamination must be in (0, 0.5)")
        self.n_trees = n_trees
        self.max_samples = max_samples
        self.contamination = contamination
        self.seed = seed
        self.trees: list = []
        self.psi_ = 0
        self.threshold_ = 0.5

    def _make_tree(self, max_depth: int, rng: np.random.Generator):
        return IsolationTree(max_depth, rng)

    def fit(self, X: np.ndarray) -> "IsolationForest":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] < 2:
            raise ValueError("X must be 2-D with at least two rows")
        rng = np.random.default_rng(self.seed)
        n = X.shape[0]
        self.psi_ = min(self.max_samples, n)
        max_depth = int(np.ceil(np.log2(self.psi_)))
        self.trees = []
        for _ in range(self.n_trees):
            idx = rng.choice(n, size=self.psi_, replace=False)
            self.trees.append(self._make_tree(max_depth, rng).fit(X[idx]))
        self.threshold_ = float(np.quantile(self.score_samples(X), 1.0 - self.contamination))
        return self

    def mean_path_length(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        total = np.zeros(X.shape[0])
        for tree in self.trees:
            total += tree.path_lengths(X)
        return total / len(self.trees)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Anomaly score s = 2^(-E[h]/c(psi)) in (0, 1]; higher is more anomalous."""
        if not self.trees:
            raise RuntimeError("forest not fitted")
        return 2.0 ** (-self.mean_path_length(X) / average_path_length(self.psi_))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """1 for anomaly, 0 for normal."""
        return (self.score_samples(X) > self.threshold_).astype(int)

    def feature_importance(self, X: np.ndarray) -> np.ndarray:
        """Normalised split-frequency weighted by 1/(depth+1) for axis-parallel trees."""
        d = np.asarray(X).shape[1]
        imp = np.zeros(d)
        for tree in self.trees:
            stack = [(tree.root, 0)]
            while stack:
                node, depth = stack.pop()
                if node is None or node.is_leaf:
                    continue
                imp[node.feature] += 1.0 / (depth + 1)
                stack += [(node.left, depth + 1), (node.right, depth + 1)]
        s = imp.sum()
        return imp / s if s > 0 else imp
