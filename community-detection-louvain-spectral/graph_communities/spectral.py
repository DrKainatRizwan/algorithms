"""Spectral clustering on the normalised adjacency (Ng-Jordan-Weiss)."""
from __future__ import annotations

import numpy as np

from .graph import Graph
from .metrics import relabel


def normalized_laplacian(g: Graph) -> np.ndarray:
    """``L = I - D^{-1/2} A D^{-1/2}`` (isolated nodes get a zero row)."""
    a = g.adjacency_matrix()
    d = a.sum(1)
    inv = np.zeros_like(d)
    inv[d > 0] = 1.0 / np.sqrt(d[d > 0])
    return np.eye(g.n) - inv[:, None] * a * inv[None, :]


def eigengap_k(g: Graph, max_k: int = 10) -> int:
    """Estimate the number of clusters from the largest gap in the Laplacian spectrum."""
    vals = np.linalg.eigvalsh(normalized_laplacian(g))
    max_k = min(max_k, g.n - 1)
    gaps = np.diff(vals[: max_k + 1])
    return int(np.argmax(gaps)) + 1


def kmeans(x: np.ndarray, k: int, rng: np.random.Generator, n_init: int = 10,
           max_iter: int = 100) -> tuple[np.ndarray, float]:
    """Lloyd's algorithm with k-means++ seeding; returns best labels and inertia."""
    best_labels, best_inertia = None, np.inf
    for _ in range(n_init):
        centers = [x[rng.integers(len(x))]]
        for _ in range(1, k):
            d2 = np.min(((x[:, None, :] - np.array(centers)[None]) ** 2).sum(2), axis=1)
            total = d2.sum()
            p = d2 / total if total > 0 else np.full(len(x), 1.0 / len(x))
            centers.append(x[rng.choice(len(x), p=p)])
        c = np.array(centers)
        labels = np.zeros(len(x), dtype=int)
        for _ in range(max_iter):
            d = ((x[:, None, :] - c[None]) ** 2).sum(2)
            new = d.argmin(1)
            for j in range(k):
                if np.any(new == j):
                    c[j] = x[new == j].mean(0)
            if np.array_equal(new, labels):
                break
            labels = new
        inertia = float(((x - c[labels]) ** 2).sum())
        if inertia < best_inertia:
            best_inertia, best_labels = inertia, labels.copy()
    return best_labels, best_inertia


def spectral_clustering(g: Graph, k: int | None = None, seed: int = 0) -> np.ndarray:
    """Cluster nodes into ``k`` groups (``k`` chosen by eigengap if omitted)."""
    if k is None:
        k = eigengap_k(g)
    if k < 1:
        raise ValueError("k must be >= 1")
    if k == 1:
        return np.zeros(g.n, dtype=int)
    vals, vecs = np.linalg.eigh(normalized_laplacian(g))
    emb = vecs[:, :k]
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.where(norms > 0, norms, 1.0)
    labels, _ = kmeans(emb, k, np.random.default_rng(seed))
    return relabel(labels)
