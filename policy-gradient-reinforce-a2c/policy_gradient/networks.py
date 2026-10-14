"""Minimal neural-network toolkit: tanh MLP with manual backprop and Adam."""

from __future__ import annotations

import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along the last axis."""
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def log_softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable log-softmax along the last axis."""
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


class MLP:
    """Fully connected network with tanh hidden units and a linear output layer.

    Parameters are stored as a flat list ``[W0, b0, W1, b1, ...]`` so that
    optimisers and gradient checks can treat them uniformly.
    """

    def __init__(self, sizes: list[int], rng: np.random.Generator,
                 out_scale: float = 1.0) -> None:
        self.sizes = list(sizes)
        self.params: list[np.ndarray] = []
        n_layers = len(sizes) - 1
        for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            gain = np.sqrt(2.0 / (n_in + n_out)) * (out_scale if i == n_layers - 1 else 1.0)
            self.params.append(rng.normal(0.0, gain, size=(n_in, n_out)))
            self.params.append(np.zeros(n_out))

    def forward(self, x: np.ndarray, cache: bool = False):
        """Evaluate the network; with ``cache=True`` also return activations."""
        acts = [x]
        h = x
        n_layers = len(self.params) // 2
        for i in range(n_layers):
            h = h @ self.params[2 * i] + self.params[2 * i + 1]
            if i < n_layers - 1:
                h = np.tanh(h)
            acts.append(h)
        return (h, acts) if cache else h

    def backward(self, acts: list[np.ndarray], grad_out: np.ndarray) -> list[np.ndarray]:
        """Backpropagate ``dL/d(output)``; returns gradients aligned with ``params``."""
        n_layers = len(self.params) // 2
        grads: list[np.ndarray] = [None] * len(self.params)  # type: ignore[list-item]
        g = grad_out
        for i in reversed(range(n_layers)):
            grads[2 * i] = acts[i].T @ g
            grads[2 * i + 1] = g.sum(axis=0)
            if i > 0:
                g = (g @ self.params[2 * i].T) * (1.0 - acts[i] ** 2)
        return grads


class Adam:
    """Adam optimiser (Kingma & Ba, 2015) with optional global-norm clipping."""

    def __init__(self, params: list[np.ndarray], lr: float = 1e-2, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8, clip_norm: float | None = None) -> None:
        self.params, self.lr, self.b1, self.b2, self.eps = params, lr, beta1, beta2, eps
        self.clip_norm = clip_norm
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads: list[np.ndarray]) -> None:
        if self.clip_norm is not None:
            norm = np.sqrt(sum(float((g ** 2).sum()) for g in grads))
            if norm > self.clip_norm:
                grads = [g * (self.clip_norm / (norm + 1e-12)) for g in grads]
        self.t += 1
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            m_hat = m / (1 - self.b1 ** self.t)
            v_hat = v / (1 - self.b2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
