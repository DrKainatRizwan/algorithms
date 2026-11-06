"""Dimensionality reduction and clustering primitives."""
from __future__ import annotations

import numpy as np


def pca(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project onto the top ``k`` principal components.

    Returns (projection, components (k, d), explained_variance_ratio)."""
    Xc = X - X.mean(axis=0)
    _, s, vt = np.linalg.svd(Xc, full_matrices=False)
    k = min(k, vt.shape[0])
    var = s ** 2
    ratio = var[:k] / max(var.sum(), 1e-300)
    return Xc @ vt[:k].T, vt[:k], ratio


def kmeans(X: np.ndarray, k: int, seed: int = 0, n_init: int = 5,
           max_iter: int = 100) -> tuple[np.ndarray, np.ndarray, float]:
    """Lloyd's algorithm with k-means++ seeding; best of ``n_init`` runs.

    Returns (labels, centers, inertia)."""
    rng = np.random.default_rng(seed)
    n = len(X)
    if k > n:
        raise ValueError("k larger than number of points")
    best: tuple[np.ndarray, np.ndarray, float] | None = None
    for _ in range(n_init):
        centers = [X[rng.integers(n)]]
        for _ in range(1, k):
            d2 = np.min([((X - c) ** 2).sum(1) for c in centers], axis=0)
            tot = d2.sum()
            probs = d2 / tot if tot > 0 else np.full(n, 1.0 / n)
            centers.append(X[rng.choice(n, p=probs)])
        C = np.array(centers)
        labels = np.zeros(n, int)
        for it in range(max_iter):
            d = ((X[:, None, :] - C[None]) ** 2).sum(2)
            new = d.argmin(1)
            if it > 0 and np.array_equal(new, labels):
                break
            labels = new
            for j in range(k):
                m = labels == j
                if m.any():
                    C[j] = X[m].mean(0)
                else:  # re-seed empty cluster at farthest point
                    C[j] = X[d.min(1).argmax()]
        inertia = float(((X - C[labels]) ** 2).sum())
        if best is None or inertia < best[2]:
            best = (labels.copy(), C.copy(), inertia)
    assert best is not None
    return best


def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette coefficient (exact, O(n^2)). Returns 0 for one cluster."""
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return 0.0
    D = np.sqrt(np.maximum(((X[:, None] - X[None]) ** 2).sum(2), 0.0))
    n = len(X)
    s = np.zeros(n)
    for i in range(n):
        own = labels == labels[i]
        if own.sum() <= 1:
            continue
        a = D[i, own].sum() / (own.sum() - 1)
        b = min(D[i, labels == u].mean() for u in uniq if u != labels[i])
        s[i] = (b - a) / max(a, b, 1e-300)
    return float(s.mean())
