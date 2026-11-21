"""Graph diagnostics for HNSW indexes."""
from __future__ import annotations

import numpy as np

from .hnsw import HNSWIndex


def degree_stats(index: HNSWIndex) -> dict[int, dict[str, float]]:
    """Per-layer node count and min / mean / max out-degree."""
    out: dict[int, dict[str, float]] = {}
    for layer in range(index.max_level + 1):
        deg = [len(l[layer]) for l in index._links if len(l) > layer]
        out[layer] = {"nodes": len(deg), "min": min(deg),
                      "mean": float(np.mean(deg)), "max": max(deg)}
    return out


def greedy_hops(index: HNSWIndex, q: np.ndarray) -> int:
    """Number of greedy moves taken descending from the entry point to layer 0."""
    q = np.asarray(q, dtype=float)
    cur = index.entry
    hops = 0
    for layer in range(index.max_level, -1, -1):
        improved = True
        while improved:
            improved = False
            nb = index._links[cur][layer]
            if not nb:
                break
            ids = [cur] + nb
            d = index._dist(q, index._vecs[ids])
            j = int(np.argmin(d))
            if j != 0:
                cur = ids[j]
                hops += 1
                improved = True
    return hops


def reciprocal_fraction(index: HNSWIndex, layer: int = 0) -> float:
    """Fraction of directed edges whose reverse edge also exists."""
    total = rec = 0
    for a, links in enumerate(index._links):
        if len(links) <= layer:
            continue
        for b in links[layer]:
            total += 1
            rec += a in index._links[b][layer]
    return rec / total if total else 1.0
