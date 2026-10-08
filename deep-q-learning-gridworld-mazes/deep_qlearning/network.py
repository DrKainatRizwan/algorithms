"""A fully-connected network with hand-written backprop, plus the Adam optimiser."""
from __future__ import annotations

import numpy as np


class MLP:
    """ReLU multilayer perceptron with a linear output layer."""

    def __init__(self, sizes: list[int], seed: int = 0) -> None:
        if len(sizes) < 2:
            raise ValueError("need at least input and output sizes")
        rng = np.random.default_rng(seed)
        self.W = [rng.normal(0, np.sqrt(2.0 / m), size=(m, n)) for m, n in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(n) for n in sizes[1:]]
        self._cache: list[np.ndarray] = []

    @property
    def params(self) -> list[np.ndarray]:
        return self.W + self.b

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        self._cache = [x]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            x = x @ W + b
            if i < len(self.W) - 1:
                x = np.maximum(x, 0.0)
            self._cache.append(x)
        return x

    def backward(self, grad_out: np.ndarray) -> list[np.ndarray]:
        """Backpropagate dL/d(output); returns grads aligned with ``params``."""
        gW: list[np.ndarray] = [None] * len(self.W)  # type: ignore[list-item]
        gb: list[np.ndarray] = [None] * len(self.W)  # type: ignore[list-item]
        g = grad_out
        for i in reversed(range(len(self.W))):
            gW[i] = self._cache[i].T @ g
            gb[i] = g.sum(axis=0)
            if i > 0:
                g = (g @ self.W[i].T) * (self._cache[i] > 0)
        return gW + gb

    def copy_from(self, other: "MLP") -> None:
        for p, q in zip(self.params, other.params):
            p[...] = q

    def soft_update(self, other: "MLP", tau: float) -> None:
        for p, q in zip(self.params, other.params):
            p *= 1.0 - tau
            p += tau * q


class Adam:
    """Adam optimiser operating in place on a list of parameter arrays."""

    def __init__(self, params: list[np.ndarray], lr: float = 1e-3, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8, clip: float | None = None) -> None:
        self.params, self.lr, self.b1, self.b2, self.eps, self.clip = params, lr, beta1, beta2, eps, clip
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads: list[np.ndarray]) -> None:
        if self.clip is not None:
            norm = np.sqrt(sum(float((g ** 2).sum()) for g in grads))
            if norm > self.clip:
                grads = [g * (self.clip / norm) for g in grads]
        self.t += 1
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            mh = m / (1 - self.b1 ** self.t)
            vh = v / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)
