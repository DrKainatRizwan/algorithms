"""Stacked GNN with ReLU, dropout, softmax cross-entropy and Adam."""
from __future__ import annotations

import numpy as np

from .layers import GATLayer, GCNLayer, Layer, SAGELayer

KINDS = ("gcn", "sage", "gat", "mlp")


def softmax(z: np.ndarray) -> np.ndarray:
    """Row-wise numerically stable softmax."""
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(logits: np.ndarray, y: np.ndarray, mask: np.ndarray
                  ) -> tuple[float, np.ndarray]:
    """Mean cross-entropy over masked nodes and its gradient w.r.t. logits."""
    p = softmax(logits)
    m = int(mask.sum())
    idx = np.flatnonzero(mask)
    loss = -np.log(p[idx, y[idx]] + 1e-12).mean()
    grad = np.zeros_like(p)
    grad[idx] = p[idx]
    grad[idx, y[idx]] -= 1.0
    return float(loss), grad / m


class GNN:
    """A multi-layer graph network.

    Args:
        kind: ``"gcn"``, ``"sage"``, ``"gat"`` or ``"mlp"`` (no propagation).
        adj: Adjacency matrix of the graph the model will run on.
        dims: Layer widths ``[d_in, h1, ..., n_classes]``.
        dropout: Dropout probability applied to every layer input in training.
        seed: Seed for initialisation and dropout masks.
    """

    def __init__(self, kind: str, adj: np.ndarray, dims: list[int],
                 dropout: float = 0.5, seed: int = 0, heads: int = 2) -> None:
        if kind not in KINDS:
            raise ValueError(f"unknown kind {kind!r}")
        self.rng = np.random.default_rng(seed)
        self.dropout = dropout
        self.layers: list[Layer] = []
        for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
            if kind == "gcn":
                layer: Layer = GCNLayer(adj, a, b, self.rng)
            elif kind == "mlp":
                layer = GCNLayer(adj, a, b, self.rng, identity=True)
            elif kind == "sage":
                layer = SAGELayer(adj, a, b, self.rng)
            else:
                last = i == len(dims) - 2
                layer = GATLayer(adj, a, b, self.rng, heads=1 if last else heads)
            self.layers.append(layer)
        self._cache: list[tuple[np.ndarray | None, np.ndarray | None]] = []

    def forward(self, x: np.ndarray, train: bool = False) -> np.ndarray:
        """Return class logits; dropout is active only if ``train``."""
        self._cache = []
        h = x
        for i, layer in enumerate(self.layers):
            drop = None
            if train and self.dropout > 0:
                keep = 1.0 - self.dropout
                drop = (self.rng.random(h.shape) < keep) / keep
                h = h * drop
            h = layer.forward(h)
            relu = None
            if i < len(self.layers) - 1:
                relu = h > 0
                h = h * relu
            self._cache.append((drop, relu))
        return h

    def backward(self, grad: np.ndarray) -> None:
        """Backpropagate ``dL/dlogits``; fills ``layer.grads``."""
        for layer, (drop, relu) in zip(reversed(self.layers), reversed(self._cache)):
            if relu is not None:
                grad = grad * relu
            grad = layer.backward(grad)
            if drop is not None:
                grad = grad * drop

    def embeddings(self, x: np.ndarray) -> list[np.ndarray]:
        """Hidden representations after each layer (evaluation mode)."""
        outs, h = [], x
        for i, layer in enumerate(self.layers):
            h = layer.forward(h)
            if i < len(self.layers) - 1:
                h = np.maximum(h, 0.0)
            outs.append(h)
        return outs

    def n_params(self) -> int:
        return sum(p.size for l in self.layers for p in l.params.values())


class Adam:
    """Adam optimiser with decoupled-from-loss L2 weight decay on weights."""

    def __init__(self, layers: list[Layer], lr: float = 0.01,
                 weight_decay: float = 5e-4, b1: float = 0.9, b2: float = 0.999) -> None:
        self.layers, self.lr, self.wd = layers, lr, weight_decay
        self.b1, self.b2, self.t = b1, b2, 0
        self.m = [{k: np.zeros_like(v) for k, v in l.params.items()} for l in layers]
        self.v = [{k: np.zeros_like(v) for k, v in l.params.items()} for l in layers]

    def step(self) -> None:
        self.t += 1
        for l, m, v in zip(self.layers, self.m, self.v):
            for k, p in l.params.items():
                g = l.grads[k]
                if k != "b":
                    g = g + self.wd * p
                m[k] = self.b1 * m[k] + (1 - self.b1) * g
                v[k] = self.b2 * v[k] + (1 - self.b2) * g * g
                mh = m[k] / (1 - self.b1 ** self.t)
                vh = v[k] / (1 - self.b2 ** self.t)
                p -= self.lr * mh / (np.sqrt(vh) + 1e-8)
