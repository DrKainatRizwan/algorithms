"""Diagnostics: over-smoothing measures and depth sweeps."""
from __future__ import annotations

import numpy as np

from .graph import Graph, gcn_normalize


def dirichlet_energy(h: np.ndarray, adj: np.ndarray) -> float:
    """Mean squared feature difference across edges, normalised by feature norm."""
    i, j = np.nonzero(np.triu(adj, 1))
    if i.size == 0:
        return 0.0
    num = ((h[i] - h[j]) ** 2).sum(axis=1).mean()
    den = (h ** 2).sum(axis=1).mean() + 1e-12
    return float(num / den)


def mean_pairwise_distance(h: np.ndarray) -> float:
    """Average Euclidean distance of node embeddings to their centroid."""
    return float(np.linalg.norm(h - h.mean(axis=0), axis=1).mean())


def smoothing_curve(g: Graph, max_hops: int) -> list[float]:
    """Dirichlet energy of ``A_hat^k X`` for ``k = 0..max_hops`` (pure propagation)."""
    a = gcn_normalize(g.adj)
    h = g.x.copy()
    out = [dirichlet_energy(h, g.adj)]
    for _ in range(max_hops):
        h = a @ h
        out.append(dirichlet_energy(h, g.adj))
    return out


def label_agreement_after_hops(g: Graph, hops: int) -> float:
    """Nearest-class-centroid accuracy of ``A_hat^hops X`` (training-free probe)."""
    a = gcn_normalize(g.adj)
    h = g.x.copy()
    for _ in range(hops):
        h = a @ h
    cents = np.stack([h[g.y == c].mean(axis=0) for c in range(g.n_classes)])
    d = ((h[:, None, :] - cents[None]) ** 2).sum(axis=2)
    return float((d.argmin(axis=1) == g.y).mean())
