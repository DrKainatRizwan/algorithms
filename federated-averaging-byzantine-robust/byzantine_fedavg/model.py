"""Multinomial logistic regression with a flat parameter vector."""

from __future__ import annotations

import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax."""
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class SoftmaxModel:
    """Softmax classifier whose weights live in one flat vector.

    A flat representation lets aggregation rules treat each client update
    as a point in R^d.
    """

    def __init__(self, n_features: int, n_classes: int) -> None:
        self.n_features = n_features
        self.n_classes = n_classes

    @property
    def dim(self) -> int:
        return (self.n_features + 1) * self.n_classes

    def init_params(self) -> np.ndarray:
        return np.zeros(self.dim)

    def _unpack(self, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        k = self.n_features * self.n_classes
        return w[:k].reshape(self.n_features, self.n_classes), w[k:]

    def predict_proba(self, w: np.ndarray, X: np.ndarray) -> np.ndarray:
        W, b = self._unpack(w)
        return softmax(X @ W + b)

    def predict(self, w: np.ndarray, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(w, X).argmax(axis=1)

    def accuracy(self, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(w, X) == y))

    def loss_and_grad(
        self, w: np.ndarray, X: np.ndarray, y: np.ndarray, l2: float = 0.0
    ) -> tuple[float, np.ndarray]:
        """Mean cross-entropy (plus L2 on weights) and its gradient."""
        n = X.shape[0]
        W, _ = self._unpack(w)
        P = self.predict_proba(w, X)
        loss = -np.log(P[np.arange(n), y] + 1e-12).mean() + 0.5 * l2 * np.sum(W * W)
        D = P.copy()
        D[np.arange(n), y] -= 1.0
        D /= n
        gW = X.T @ D + l2 * W
        gb = D.sum(axis=0)
        return float(loss), np.concatenate([gW.ravel(), gb])

    def local_train(
        self,
        w: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int,
        lr: float,
        batch_size: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Run minibatch SGD from ``w`` and return the new parameters."""
        w = w.copy()
        n = X.shape[0]
        for _ in range(epochs):
            order = rng.permutation(n)
            for s in range(0, n, batch_size):
                idx = order[s : s + batch_size]
                _, g = self.loss_and_grad(w, X[idx], y[idx])
                w -= lr * g
        return w
