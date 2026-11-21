"""Partition quality and partition-similarity measures."""
from __future__ import annotations

import numpy as np

from .graph import Graph


def relabel(labels) -> np.ndarray:
    """Map arbitrary labels to consecutive integers ``0..k-1`` (first-seen order)."""
    mapping: dict = {}
    out = np.empty(len(labels), dtype=int)
    for i, l in enumerate(labels):
        out[i] = mapping.setdefault(l, len(mapping))
    return out


def modularity(g: Graph, labels, resolution: float = 1.0) -> float:
    """Newman-Girvan modularity ``Q = sum_c [ in_c/m - gamma (tot_c/2m)^2 ]``."""
    m = g.total_weight()
    if m == 0:
        return 0.0
    lab = relabel(labels)
    k = int(lab.max()) + 1 if lab.size else 0
    inside = np.zeros(k)
    tot = np.zeros(k)
    for u in range(g.n):
        tot[lab[u]] += g.degree(u)
        for v, w in g.adj[u].items():
            if u <= v and lab[u] == lab[v]:
                inside[lab[u]] += w
    return float(np.sum(inside / m - resolution * (tot / (2.0 * m)) ** 2))


def conductance(g: Graph, nodes) -> float:
    """Cut weight divided by the smaller of the two volumes."""
    s = set(int(x) for x in nodes)
    cut = vol_s = 0.0
    for u in range(g.n):
        d = g.degree(u)
        if u in s:
            vol_s += d
            cut += sum(w for v, w in g.adj[u].items() if v not in s)
    vol_t = 2.0 * g.total_weight() - vol_s
    denom = min(vol_s, vol_t)
    return cut / denom if denom > 0 else 0.0


def _contingency(a, b) -> np.ndarray:
    a, b = relabel(a), relabel(b)
    table = np.zeros((a.max() + 1, b.max() + 1))
    for x, y in zip(a, b):
        table[x, y] += 1
    return table


def nmi(a, b) -> float:
    """Normalised mutual information (arithmetic-mean normalisation)."""
    t = _contingency(a, b)
    n = t.sum()
    pa, pb = t.sum(1) / n, t.sum(0) / n
    ha = -np.sum(pa[pa > 0] * np.log(pa[pa > 0]))
    hb = -np.sum(pb[pb > 0] * np.log(pb[pb > 0]))
    if ha == 0 and hb == 0:
        return 1.0
    pij = t / n
    nz = pij > 0
    mi = np.sum(pij[nz] * np.log(pij[nz] / np.outer(pa, pb)[nz]))
    return float(mi / (0.5 * (ha + hb)))


def adjusted_rand_index(a, b) -> float:
    """Hubert-Arabie adjusted Rand index."""
    t = _contingency(a, b)
    n = t.sum()
    c2 = lambda x: x * (x - 1) / 2.0
    sum_ij = c2(t).sum()
    sum_a = c2(t.sum(1)).sum()
    sum_b = c2(t.sum(0)).sum()
    expected = sum_a * sum_b / c2(n) if n > 1 else 0.0
    maximum = 0.5 * (sum_a + sum_b)
    if maximum == expected:
        return 1.0
    return float((sum_ij - expected) / (maximum - expected))


def disconnected_communities(g: Graph, labels) -> list[int]:
    """Community ids whose induced subgraph is not connected."""
    lab = relabel(labels)
    bad = []
    for c in range(int(lab.max()) + 1 if lab.size else 0):
        members = [u for u in range(g.n) if lab[u] == c]
        mset = set(members)
        seen = {members[0]}
        stack = [members[0]]
        while stack:
            u = stack.pop()
            for v in g.adj[u]:
                if v in mset and v not in seen:
                    seen.add(v)
                    stack.append(v)
        if len(seen) != len(members):
            bad.append(c)
    return bad
