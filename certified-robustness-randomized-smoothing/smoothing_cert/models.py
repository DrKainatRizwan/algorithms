"""Base classifiers (softmax regression, one-hidden-layer MLP) and synthetic datasets."""

from __future__ import annotations

import numpy as np


def make_blobs(n: int, dim: int = 10, classes: int = 3, spread: float = 1.0,
               sep: float = 3.0, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian blobs with class means at ``sep`` along random unit directions."""
    rng = np.random.default_rng(1234)  # fixed class geometry
    means = rng.normal(size=(classes, dim))
    means = sep * means / np.linalg.norm(means, axis=1, keepdims=True)
    rng = np.random.default_rng(seed)
    y = rng.integers(0, classes, size=n)
    X = means[y] + spread * rng.normal(size=(n, dim))
    return X, y


def make_rings(n: int, seed: int = 0, noise: float = 0.08) -> tuple[np.ndarray, np.ndarray]:
    """Two concentric rings in 2-D (not linearly separable)."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    r = np.where(y == 0, 1.0, 2.2) + noise * rng.normal(size=n)
    th = rng.uniform(0, 2 * np.pi, size=n)
    return np.stack([r * np.cos(th), r * np.sin(th)], axis=1), y


def accuracy(model, X: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(model.predict(X) == y))


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class _Base:
    num_classes: int

    def logits(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Hard labels for a batch (or a single vector)."""
        X = np.atleast_2d(X)
        return np.argmax(self.logits(X), axis=1)

    def proba(self, X: np.ndarray) -> np.ndarray:
        return _softmax(self.logits(np.atleast_2d(X)))

    def input_gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 60, lr: float = 0.05,
            batch: int = 64, noise_sigma: float = 0.0, seed: int = 0) -> list[float]:
        """Minibatch Adam with optional Gaussian noise augmentation. Returns epoch losses."""
        rng = np.random.default_rng(seed)
        params = self._params()
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        t = 0
        losses = []
        for _ in range(epochs):
            order = rng.permutation(len(X))
            tot = 0.0
            for s in range(0, len(X), batch):
                idx = order[s:s + batch]
                xb = X[idx]
                if noise_sigma > 0:
                    xb = xb + noise_sigma * rng.normal(size=xb.shape)
                loss, grads = self._loss_grads(xb, y[idx])
                tot += loss * len(idx)
                t += 1
                for i, (p, g) in enumerate(zip(params, grads)):
                    m[i] = 0.9 * m[i] + 0.1 * g
                    v[i] = 0.999 * v[i] + 0.001 * g * g
                    mh = m[i] / (1 - 0.9 ** t)
                    vh = v[i] / (1 - 0.999 ** t)
                    p -= lr * mh / (np.sqrt(vh) + 1e-8)
            losses.append(tot / len(X))
        return losses

    def _params(self) -> list[np.ndarray]:  # pragma: no cover
        raise NotImplementedError

    def _loss_grads(self, X, y):  # pragma: no cover
        raise NotImplementedError


class LinearSoftmax(_Base):
    """Multinomial logistic regression."""

    def __init__(self, dim: int, num_classes: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W = 0.01 * rng.normal(size=(dim, num_classes))
        self.b = np.zeros(num_classes)
        self.num_classes = num_classes

    def logits(self, X):
        return np.atleast_2d(X) @ self.W + self.b

    def _params(self):
        return [self.W, self.b]

    def _loss_grads(self, X, y):
        P = _softmax(self.logits(X))
        n = len(X)
        loss = float(-np.mean(np.log(P[np.arange(n), y] + 1e-12)))
        G = P.copy()
        G[np.arange(n), y] -= 1
        G /= n
        return loss, [X.T @ G, G.sum(axis=0)]

    def input_gradient(self, X, y):
        """Gradient of the cross-entropy loss w.r.t. each input row."""
        X = np.atleast_2d(X)
        P = _softmax(self.logits(X))
        P[np.arange(len(X)), y] -= 1
        return P @ self.W.T


class MLPClassifier(_Base):
    """One-hidden-layer ReLU network with hand-written backprop."""

    def __init__(self, dim: int, hidden: int, num_classes: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(size=(dim, hidden)) * np.sqrt(2.0 / dim)
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(size=(hidden, num_classes)) * np.sqrt(2.0 / hidden)
        self.b2 = np.zeros(num_classes)
        self.num_classes = num_classes

    def logits(self, X):
        H = np.maximum(np.atleast_2d(X) @ self.W1 + self.b1, 0)
        return H @ self.W2 + self.b2

    def _params(self):
        return [self.W1, self.b1, self.W2, self.b2]

    def _loss_grads(self, X, y):
        Z1 = X @ self.W1 + self.b1
        H = np.maximum(Z1, 0)
        P = _softmax(H @ self.W2 + self.b2)
        n = len(X)
        loss = float(-np.mean(np.log(P[np.arange(n), y] + 1e-12)))
        G = P.copy()
        G[np.arange(n), y] -= 1
        G /= n
        dH = (G @ self.W2.T) * (Z1 > 0)
        return loss, [X.T @ dH, dH.sum(axis=0), H.T @ G, G.sum(axis=0)]

    def input_gradient(self, X, y):
        X = np.atleast_2d(X)
        Z1 = X @ self.W1 + self.b1
        H = np.maximum(Z1, 0)
        P = _softmax(H @ self.W2 + self.b2)
        P[np.arange(len(X)), y] -= 1
        return ((P @ self.W2.T) * (Z1 > 0)) @ self.W1.T
