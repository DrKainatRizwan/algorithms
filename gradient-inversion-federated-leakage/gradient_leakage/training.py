"""Plain minibatch SGD used to move the victim away from its random init."""
from __future__ import annotations

import numpy as np

from .model import MLP


def train_sgd(model: MLP, X: np.ndarray, y: np.ndarray, epochs: int, lr: float,
              batch: int, rng: np.random.Generator) -> list[float]:
    """Train in place; return the per-epoch loss history."""
    hist = []
    n = len(y)
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            g = model.gradients(X[idx], y[idx])
            model.W1 -= lr * g["W1"]; model.b1 -= lr * g["b1"]
            model.W2 -= lr * g["W2"]; model.b2 -= lr * g["b2"]
        hist.append(model.loss(X, y))
    return hist
