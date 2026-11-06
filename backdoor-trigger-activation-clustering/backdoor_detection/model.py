"""A small fully-connected network with manual backprop (NumPy only)."""
from __future__ import annotations

import numpy as np


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class MLP:
    """ReLU MLP. ``activations`` exposes the last hidden layer, which is the
    representation inspected by backdoor detectors."""

    def __init__(self, sizes: list[int], seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W = [rng.normal(0, np.sqrt(2.0 / a), size=(a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]

    def _forward(self, X: np.ndarray) -> list[np.ndarray]:
        hs = [np.asarray(X, float)]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = hs[-1] @ W + b
            hs.append(z if i == len(self.W) - 1 else np.maximum(z, 0.0))
        return hs

    def logits(self, X: np.ndarray) -> np.ndarray:
        return self._forward(X)[-1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.logits(X).argmax(axis=1)

    def activations(self, X: np.ndarray) -> np.ndarray:
        """Activations of the last hidden layer."""
        return self._forward(X)[-2]

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        p = _softmax(self.logits(X))
        return float(-np.log(p[np.arange(len(y)), y] + 1e-12).mean())

    def step(self, X: np.ndarray, y: np.ndarray, lr: float, wd: float = 0.0) -> None:
        hs = self._forward(X)
        d = _softmax(hs[-1])
        d[np.arange(len(y)), y] -= 1.0
        d /= len(y)
        for i in range(len(self.W) - 1, -1, -1):
            gW = hs[i].T @ d + wd * self.W[i]
            gb = d.sum(axis=0)
            if i > 0:
                d = (d @ self.W[i].T) * (hs[i] > 0)
            self.W[i] -= lr * gW
            self.b[i] -= lr * gb


def train(X: np.ndarray, y: np.ndarray, n_classes: int, hidden: tuple[int, ...] = (32, 16),
          epochs: int = 40, lr: float = 0.05, batch: int = 64, seed: int = 0,
          wd: float = 1e-4) -> MLP:
    """Mini-batch SGD training; deterministic given ``seed``."""
    model = MLP([X.shape[1], *hidden, n_classes], seed=seed)
    rng = np.random.default_rng(seed + 1)
    n = len(y)
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            sel = order[s:s + batch]
            model.step(X[sel], y[sel], lr, wd)
    return model
