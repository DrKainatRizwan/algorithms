"""Girvan-Newman divisive clustering via Brandes edge betweenness."""
from __future__ import annotations

from collections import deque

import numpy as np

from .graph import Graph
from .metrics import modularity


def edge_betweenness(g: Graph) -> dict[tuple[int, int], float]:
    """Unweighted edge betweenness for every edge ``(u, v)`` with ``u < v``."""
    eb: dict[tuple[int, int], float] = {(u, v): 0.0 for u, v, _ in g.edges() if u != v}
    for s in range(g.n):
        order: list[int] = []
        preds: list[list[int]] = [[] for _ in range(g.n)]
        sigma = np.zeros(g.n)
        sigma[s] = 1.0
        dist = [-1] * g.n
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            order.append(u)
            for v in g.adj[u]:
                if v == u:
                    continue
                if dist[v] < 0:
                    dist[v] = dist[u] + 1
                    q.append(v)
                if dist[v] == dist[u] + 1:
                    sigma[v] += sigma[u]
                    preds[v].append(u)
        delta = np.zeros(g.n)
        for w in reversed(order):
            for v in preds[w]:
                c = sigma[v] / sigma[w] * (1.0 + delta[w])
                eb[(min(v, w), max(v, w))] += c
                delta[v] += c
    return {e: b / 2.0 for e, b in eb.items()}


def _labels_from_components(g: Graph) -> np.ndarray:
    labels = np.zeros(g.n, dtype=int)
    for i, comp in enumerate(g.connected_components()):
        labels[comp] = i
    return labels


def girvan_newman(g: Graph) -> tuple[np.ndarray, float, list[tuple[int, float]]]:
    """Remove highest-betweenness edges until none remain.

    Returns ``(best_labels, best_modularity, trace)`` where ``trace`` lists
    ``(number_of_components, modularity)`` each time the split count grows.
    """
    work = g.copy()
    best_labels = _labels_from_components(work)
    best_q = modularity(g, best_labels)
    n_comp = int(best_labels.max()) + 1
    trace = [(n_comp, best_q)]
    while any(u != v for u, v, _ in work.edges()):
        eb = edge_betweenness(work)
        # deterministic tie-break: smallest edge index among maxima
        top = max(eb.values())
        u, v = min(e for e, b in eb.items() if b >= top - 1e-9)
        work.remove_edge(u, v)
        labels = _labels_from_components(work)
        k = int(labels.max()) + 1
        if k > n_comp:
            n_comp = k
            q = modularity(g, labels)
            trace.append((k, q))
            if q > best_q + 1e-12:
                best_q, best_labels = q, labels
    return best_labels, best_q, trace
