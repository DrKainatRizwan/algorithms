"""Push-sum gossip averaging (Kempe, Dobra & Gehrke, 2003)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .network import Graph


@dataclass
class PushSumResult:
    estimates: np.ndarray
    rounds: int
    max_error_history: list[float] = field(default_factory=list)
    converged: bool = False
    mass_s: float = 0.0
    mass_w: float = 0.0


def push_sum(graph: Graph, values: np.ndarray, rng: np.random.Generator,
             tol: float = 1e-6, max_rounds: int = 500) -> PushSumResult:
    """Estimate the mean of ``values`` using the push-sum protocol.

    Each node holds a pair (s, w) initialised to (x_i, 1). Each round it keeps
    half and sends half to a random neighbour. The ratio s/w converges to the
    global mean at every node; total s and total w are conserved exactly.
    """
    x = np.asarray(values, dtype=float)
    if x.shape != (graph.n,):
        raise ValueError("values must have one entry per node")
    s = x.copy()
    w = np.ones(graph.n)
    target = float(x.mean())
    scale = max(1.0, abs(target))
    hist: list[float] = []
    converged = False
    rounds = 0
    for rounds in range(1, max_rounds + 1):
        ns = s / 2.0
        nw = w / 2.0
        for u in range(graph.n):
            v = graph.random_neighbor(u, rng)
            if v < 0:
                ns[u] += s[u] / 2.0
                nw[u] += w[u] / 2.0
            else:
                ns[v] += s[u] / 2.0
                nw[v] += w[u] / 2.0
        s, w = ns, nw
        err = float(np.max(np.abs(s / w - target)) / scale)
        hist.append(err)
        if err < tol:
            converged = True
            break
    return PushSumResult(s / w, rounds, hist, converged, float(s.sum()), float(w.sum()))
