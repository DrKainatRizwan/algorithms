"""Graph containers, synthetic benchmark generators and propagation matrices."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Graph:
    """A undirected attributed graph with node labels.

    Attributes:
        adj: Dense symmetric 0/1 adjacency matrix without self loops, ``(n, n)``.
        x: Node feature matrix, ``(n, d)``.
        y: Integer node labels, ``(n,)``.
    """

    adj: np.ndarray
    x: np.ndarray
    y: np.ndarray

    @property
    def n_nodes(self) -> int:
        return self.adj.shape[0]

    @property
    def n_classes(self) -> int:
        return int(self.y.max()) + 1

    @property
    def n_edges(self) -> int:
        return int(self.adj.sum() // 2)


def stochastic_block_model(
    sizes: list[int],
    p_in: float,
    p_out: float,
    n_features: int = 16,
    signal: float = 0.6,
    noise: float = 1.0,
    seed: int = 0,
) -> Graph:
    """Sample a stochastic block model with class-conditional Gaussian features.

    Each block is one class. Node features are ``signal * mu_c + noise * eps``
    where ``mu_c`` is a random unit-scale class prototype, so features alone
    are only weakly informative and the graph structure helps.
    """
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(len(sizes)), sizes)
    n = y.size
    same = y[:, None] == y[None, :]
    probs = np.where(same, p_in, p_out)
    upper = np.triu(rng.random((n, n)) < probs, k=1)
    adj = (upper | upper.T).astype(float)
    protos = rng.normal(size=(len(sizes), n_features))
    x = signal * protos[y] + noise * rng.normal(size=(n, n_features))
    return Graph(adj=adj, x=x, y=y)


def add_self_loops(adj: np.ndarray) -> np.ndarray:
    """Return ``adj`` with a unit diagonal."""
    out = adj.copy()
    np.fill_diagonal(out, 1.0)
    return out


def gcn_normalize(adj: np.ndarray) -> np.ndarray:
    """Symmetric normalisation ``D^-1/2 (A + I) D^-1/2`` of Kipf & Welling."""
    a = add_self_loops(adj)
    d = a.sum(axis=1)
    inv = 1.0 / np.sqrt(d)
    return a * inv[:, None] * inv[None, :]


def mean_normalize(adj: np.ndarray) -> np.ndarray:
    """Row-normalised adjacency (neighbour mean); isolated nodes get a zero row."""
    d = adj.sum(axis=1)
    inv = np.divide(1.0, d, out=np.zeros_like(d), where=d > 0)
    return adj * inv[:, None]


def split_masks(
    y: np.ndarray, per_class_train: int, n_val: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Planetoid-style split: few labelled nodes per class for training.

    Returns boolean ``(train, val, test)`` masks; the remainder is test.
    """
    rng = np.random.default_rng(seed)
    n = y.size
    train = np.zeros(n, dtype=bool)
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))[:per_class_train]
        train[idx] = True
    rest = rng.permutation(np.flatnonzero(~train))
    val = np.zeros(n, dtype=bool)
    val[rest[:n_val]] = True
    test = ~(train | val)
    return train, val, test


def edge_homophily(g: Graph) -> float:
    """Fraction of edges joining nodes with the same label."""
    i, j = np.nonzero(np.triu(g.adj, 1))
    if i.size == 0:
        return 0.0
    return float(np.mean(g.y[i] == g.y[j]))
