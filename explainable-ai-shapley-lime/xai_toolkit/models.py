"""Small models to explain: logistic regression, MLP, regression tree, random forest.

Every model exposes ``predict(X) -> (n,)`` returning a scalar score (probability
for classifiers) so explainers can treat them as black boxes.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * z))


class LogisticModel:
    """L2-regularised logistic regression trained by full-batch gradient descent."""

    def __init__(self, l2: float = 1e-2, lr: float = 0.5, epochs: int = 500):
        self.l2, self.lr, self.epochs = l2, lr, epochs
        self.w = np.zeros(0)
        self.b = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticModel":
        n, d = X.shape
        self.w, self.b = np.zeros(d), 0.0
        for _ in range(self.epochs):
            err = _sigmoid(X @ self.w + self.b) - y
            self.w -= self.lr * (X.T @ err / n + self.l2 * self.w)
            self.b -= self.lr * err.mean()
        return self

    def decision(self, X: np.ndarray) -> np.ndarray:
        return np.atleast_2d(X) @ self.w + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        return _sigmoid(self.decision(X))


class MLPClassifier:
    """One-hidden-layer tanh network with sigmoid output, trained with Adam."""

    def __init__(self, hidden: int = 16, lr: float = 0.02, epochs: int = 300,
                 batch: int = 64, l2: float = 1e-4, seed: int = 0):
        self.hidden, self.lr, self.epochs, self.batch, self.l2 = hidden, lr, epochs, batch, l2
        self.seed = seed
        self.W1 = self.b1 = self.W2 = self.b2 = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPClassifier":
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        self.W1 = rng.normal(0, 1 / np.sqrt(d), (d, self.hidden))
        self.b1 = np.zeros(self.hidden)
        self.W2 = rng.normal(0, 1 / np.sqrt(self.hidden), self.hidden)
        self.b2 = 0.0
        params = [self.W1, self.b1, self.W2, np.array([self.b2])]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        t = 0
        for _ in range(self.epochs):
            perm = rng.permutation(n)
            for s in range(0, n, self.batch):
                i = perm[s:s + self.batch]
                xb, yb = X[i], y[i]
                h = np.tanh(xb @ self.W1 + self.b1)
                p = _sigmoid(h @ self.W2 + self.b2)
                e = (p - yb) / len(i)
                gW2 = h.T @ e + self.l2 * self.W2
                gb2 = np.array([e.sum()])
                dh = np.outer(e, self.W2) * (1 - h ** 2)
                gW1 = xb.T @ dh + self.l2 * self.W1
                gb1 = dh.sum(axis=0)
                t += 1
                for k, (p_, g) in enumerate(zip([self.W1, self.b1, self.W2, params[3]],
                                                [gW1, gb1, gW2, gb2])):
                    m[k] = 0.9 * m[k] + 0.1 * g
                    v[k] = 0.999 * v[k] + 0.001 * g * g
                    p_ -= self.lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v[k] / (1 - 0.999 ** t)) + 1e-8)
        self.b2 = float(params[3][0])
        return self

    def decision(self, X: np.ndarray) -> np.ndarray:
        h = np.tanh(np.atleast_2d(X) @ self.W1 + self.b1)
        return h @ self.W2 + self.b2

    def predict(self, X: np.ndarray) -> np.ndarray:
        return _sigmoid(self.decision(X))

    def gradient(self, X: np.ndarray) -> np.ndarray:
        """d predict / d x for each row, shape (n, d)."""
        X = np.atleast_2d(X)
        h = np.tanh(X @ self.W1 + self.b1)
        p = _sigmoid(h @ self.W2 + self.b2)
        dz = (((1 - h ** 2) * self.W2) @ self.W1.T)
        return dz * (p * (1 - p))[:, None]


@dataclass
class _Node:
    feature: int = -1
    thr: float = 0.0
    left: "._Node | None" = None
    right: "._Node | None" = None
    value: float = 0.0


class RegressionTree:
    """CART regression tree (variance reduction, exhaustive threshold search)."""

    def __init__(self, max_depth: int = 5, min_leaf: int = 5, max_features: int | None = None,
                 seed: int = 0):
        self.max_depth, self.min_leaf, self.max_features = max_depth, min_leaf, max_features
        self.rng = np.random.default_rng(seed)
        self.root: _Node | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RegressionTree":
        self.root = self._grow(X, y, 0)
        return self

    def _grow(self, X: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        node = _Node(value=float(y.mean()))
        if depth >= self.max_depth or len(y) < 2 * self.min_leaf or np.ptp(y) == 0:
            return node
        d = X.shape[1]
        feats = np.arange(d)
        if self.max_features and self.max_features < d:
            feats = self.rng.choice(d, self.max_features, replace=False)
        best = (0.0, -1, 0.0)
        total = y.sum(); n = len(y)
        for f in feats:
            order = np.argsort(X[:, f], kind="stable")
            xs, ys = X[order, f], y[order]
            cs = np.cumsum(ys)
            k = np.arange(1, n)
            gain = cs[:-1] ** 2 / k + (total - cs[:-1]) ** 2 / (n - k) - total ** 2 / n
            ok = (k >= self.min_leaf) & (n - k >= self.min_leaf) & (xs[1:] > xs[:-1])
            if not ok.any():
                continue
            gain = np.where(ok, gain, -np.inf)
            j = int(np.argmax(gain))
            if gain[j] > best[0] + 1e-12:
                best = (float(gain[j]), int(f), float((xs[j] + xs[j + 1]) / 2))
        if best[1] < 0:
            return node
        _, node.feature, node.thr = best
        mask = X[:, node.feature] <= node.thr
        node.left = self._grow(X[mask], y[mask], depth + 1)
        node.right = self._grow(X[~mask], y[~mask], depth + 1)
        return node

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        out = np.empty(len(X))
        for i, x in enumerate(X):
            nd = self.root
            while nd.left is not None:
                nd = nd.left if x[nd.feature] <= nd.thr else nd.right
            out[i] = nd.value
        return out


class RandomForest:
    """Bagged regression trees with feature subsampling."""

    def __init__(self, n_trees: int = 25, max_depth: int = 6, min_leaf: int = 5, seed: int = 0):
        self.n_trees, self.max_depth, self.min_leaf, self.seed = n_trees, max_depth, min_leaf, seed
        self.trees: list[RegressionTree] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForest":
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        mf = max(1, int(np.sqrt(d)) + 1)
        self.trees = []
        for t in range(self.n_trees):
            idx = rng.integers(0, n, n)
            tr = RegressionTree(self.max_depth, self.min_leaf, mf, seed=self.seed * 1000 + t)
            self.trees.append(tr.fit(X[idx], y[idx]))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.mean([t.predict(X) for t in self.trees], axis=0)
