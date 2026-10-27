"""Closed-form p=1 results, landscapes and diagnostics."""
from __future__ import annotations

import numpy as np

from .graphs import Graph
from .qaoa import QAOA


def p1_edge_expectation(g: Graph, u: int, v: int, gamma: float, beta: float) -> float:
    """Exact p=1 expectation of (1 - Z_u Z_v)/2 on an unweighted graph (Wang et al. 2018)."""
    d = len(g.neighbours(u)) - 1
    e = len(g.neighbours(v)) - 1
    f = g.triangles_on_edge(u, v)
    cg = np.cos(gamma)
    term1 = 0.25 * np.sin(4 * beta) * np.sin(gamma) * (cg ** d + cg ** e)
    term2 = 0.25 * np.sin(2 * beta) ** 2 * cg ** (d + e - 2 * f) * (1 - np.cos(2 * gamma) ** f)
    return float(0.5 + term1 - term2)


def p1_expectation(g: Graph, gamma: float, beta: float) -> float:
    """Exact p=1 expected cut of an unweighted graph without simulation."""
    if not g.is_unweighted():
        raise ValueError("closed form requires unit weights")
    return sum(p1_edge_expectation(g, u, v, gamma, beta) for u, v, _ in g.edges)


def approximation_ratio(value: float, optimum: float) -> float:
    return value / optimum


def landscape_p1(q: QAOA, gammas: np.ndarray, betas: np.ndarray) -> np.ndarray:
    """Energy grid over (gamma, beta); rows index gamma."""
    if q.p != 1:
        raise ValueError("landscape is for p=1")
    return np.array([[q.energy(np.array([g, b])) for b in betas] for g in gammas])


def best_bitstring_stats(q: QAOA, params: np.ndarray, top: int = 4) -> list[tuple[str, float, float]]:
    """Most probable bitstrings as (bits, probability, cut value)."""
    from .statevector import probabilities
    pr = probabilities(q.state(params))
    out = []
    for i in np.argsort(-pr)[:top]:
        out.append((format(int(i), f"0{q.n}b"), float(pr[i]), float(q.diag[i])))
    return out
