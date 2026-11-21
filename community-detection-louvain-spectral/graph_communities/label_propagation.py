"""Asynchronous label propagation (Raghavan, Albert, Kumara, 2007)."""
from __future__ import annotations

import numpy as np

from .graph import Graph
from .metrics import relabel


def label_propagation(g: Graph, seed: int = 0, max_iter: int = 100) -> np.ndarray:
    """Each node repeatedly adopts the label with the largest incident weight.

    Ties are broken uniformly at random (keeping the current label if it is
    among the maxima, which guarantees termination on the stopping rule).
    """
    rng = np.random.default_rng(seed)
    labels = np.arange(g.n)
    for _ in range(max_iter):
        changed = False
        for u in rng.permutation(g.n):
            weights: dict[int, float] = {}
            for v, w in g.adj[u].items():
                if v != u:
                    weights[labels[v]] = weights.get(labels[v], 0.0) + w
            if not weights:
                continue
            top = max(weights.values())
            best = sorted(l for l, w in weights.items() if abs(w - top) < 1e-12)
            if labels[u] in best:
                continue
            labels[u] = best[int(rng.integers(len(best)))]
            changed = True
        if not changed:
            break
    return relabel(labels)
