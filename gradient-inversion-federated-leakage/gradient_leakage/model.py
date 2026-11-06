"""Two-layer tanh MLP with explicit gradients (the victim model)."""
from __future__ import annotations

import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    """Row-wise stable softmax for 1-D or 2-D input."""
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class MLP:
    """``logits = W2 tanh(W1 x + b1) + b2`` with cross-entropy loss.

    Parameters are stored as ``W1 (H,d), b1 (H), W2 (K,H), b2 (K)``.
    """

    def __init__(self, d: int, h: int, k: int, rng: np.random.Generator, scale: float = 1.0):
        self.W1 = rng.standard_normal((h, d)) * scale / np.sqrt(d)
        self.b1 = np.zeros(h)
        self.W2 = rng.standard_normal((k, h)) * scale / np.sqrt(h)
        self.b2 = np.zeros(k)

    @property
    def sizes(self) -> tuple[int, int, int]:
        return self.W1.shape[1], self.W1.shape[0], self.W2.shape[0]

    def copy(self) -> "MLP":
        m = object.__new__(MLP)
        m.W1, m.b1, m.W2, m.b2 = (a.copy() for a in (self.W1, self.b1, self.W2, self.b2))
        return m

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return hidden activations ``(n,H)`` and probabilities ``(n,K)``."""
        h = np.tanh(X @ self.W1.T + self.b1)
        return h, softmax(h @ self.W2.T + self.b2)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)[1].argmax(axis=1)

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        _, p = self.forward(X)
        return float(-np.mean(np.log(p[np.arange(len(y)), y] + 1e-12)))

    def gradients(self, X: np.ndarray, y: np.ndarray, reduce: str = "mean"
                  ) -> dict[str, np.ndarray]:
        """Gradient of the cross-entropy loss, averaged (or summed) over the batch."""
        h, p = self.forward(X)
        dz = p.copy()
        dz[np.arange(len(y)), y] -= 1.0
        da = (dz @ self.W2) * (1.0 - h ** 2)
        g = {"W1": da.T @ X, "b1": da.sum(0), "W2": dz.T @ h, "b2": dz.sum(0)}
        if reduce == "mean":
            g = {k: v / len(y) for k, v in g.items()}
        return g
