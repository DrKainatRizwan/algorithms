"""Synthetic data generators and exact baselines for evaluation."""
from __future__ import annotations

import numpy as np

_VOCAB = [f"w{i}" for i in range(2000)]


def make_documents(n_base: int, dups_per_doc: int, edit_rate: float, length: int = 120,
                   seed: int = 0) -> tuple[list[str], list[int]]:
    """Random documents plus mutated near-copies.

    Returns ``(docs, group)`` where ``group[i]`` is the id of the source document;
    each copy replaces a fraction ``edit_rate`` of the words.
    """
    rng = np.random.default_rng(seed)
    docs: list[str] = []
    group: list[int] = []
    for g in range(n_base):
        words = list(rng.choice(_VOCAB, size=length))
        docs.append(" ".join(words))
        group.append(g)
        for _ in range(dups_per_doc):
            w = list(words)
            for pos in np.nonzero(rng.random(length) < edit_rate)[0]:
                w[pos] = str(rng.choice(_VOCAB))
            docs.append(" ".join(w))
            group.append(g)
    return docs, group


def make_clusters(n: int, dim: int, n_clusters: int = 20, spread: float = 0.5,
                  seed: int = 0) -> np.ndarray:
    """Gaussian-mixture vectors, shape (n, dim)."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, dim)) * 3
    labels = rng.integers(0, n_clusters, size=n)
    return centers[labels] + spread * rng.standard_normal((n, dim))


def exact_knn(data: np.ndarray, q: np.ndarray, k: int, metric: str = "cosine") -> np.ndarray:
    """Brute-force k nearest neighbour ids (ties broken by id)."""
    if metric == "l2":
        d = np.linalg.norm(data - q, axis=1)
    else:
        d = 1.0 - data @ q / (np.linalg.norm(data, axis=1) * np.linalg.norm(q) + 1e-12)
    return np.lexsort((np.arange(len(d)), d))[:k]


def recall_at_k(found: np.ndarray, truth: np.ndarray) -> float:
    """Fraction of true neighbours present in ``found``."""
    if len(truth) == 0:
        return 1.0
    return len(set(map(int, found)) & set(map(int, truth))) / len(truth)
