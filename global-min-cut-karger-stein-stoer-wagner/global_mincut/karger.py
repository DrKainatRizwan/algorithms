"""Karger's random contraction and the recursive Karger-Stein algorithm."""
from __future__ import annotations

import math

import numpy as np

from .graph import CutResult, Graph
from .stoer_wagner import stoer_wagner


class _DSU:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.p[ra] = rb
        return True


def _contract_matrix(w: np.ndarray, target: int,
                     rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Randomly contract the weighted graph ``w`` down to ``target`` super-vertices.

    Sampling an edge with probability proportional to weight, repeatedly, is
    equivalent to giving each edge an Exponential(weight) key and running
    Kruskal on the keys; this gives an O(m log m) contraction.

    Returns the contracted weight matrix and the label (0..target-1) of each
    original vertex.
    """
    n = w.shape[0]
    us, vs = np.nonzero(np.triu(w, 1))
    weights = w[us, vs]
    keys = rng.exponential(size=len(us)) / weights
    dsu = _DSU(n)
    comps = n
    for idx in np.argsort(keys, kind="stable"):
        if comps <= target:
            break
        if dsu.union(int(us[idx]), int(vs[idx])):
            comps -= 1
    roots = {}
    labels = np.empty(n, dtype=int)
    for v in range(n):
        labels[v] = roots.setdefault(dsu.find(v), len(roots))
    k = len(roots)
    onehot = np.zeros((n, k))
    onehot[np.arange(n), labels] = 1.0
    cw = onehot.T @ w @ onehot
    np.fill_diagonal(cw, 0.0)
    return cw, labels


def karger_contract(g: Graph, rng: np.random.Generator) -> CutResult:
    """One run of Karger's algorithm: contract to two super-vertices."""
    if g.n < 2:
        raise ValueError("need at least two vertices")
    cw, labels = _contract_matrix(g.w, 2, rng)
    side = tuple(int(v) for v in np.nonzero(labels == 0)[0])
    if cw.shape[0] < 2:  # disconnected graph contracted to one component
        return CutResult(0.0, side[: max(1, len(side) // 2)])
    return CutResult(float(cw[0, 1]), side)


def karger(g: Graph, trials: int, seed: int = 0) -> CutResult:
    """Best of ``trials`` independent contraction runs."""
    rng = np.random.default_rng(seed)
    best = CutResult(float("inf"), ())
    for _ in range(trials):
        r = karger_contract(g, rng)
        if r.value < best.value:
            best = r
    return best


def recommended_trials(n: int, failure: float = 0.01) -> int:
    """Runs needed so that failure probability <= ``failure``.

    One run succeeds with probability >= 2/(n(n-1)).
    """
    p = 2.0 / (n * (n - 1))
    return int(math.ceil(math.log(failure) / math.log1p(-p)))


def _ks_rec(w: np.ndarray, members: list[list[int]], rng: np.random.Generator,
            base: int) -> tuple[float, list[int]]:
    n = w.shape[0]
    if n <= base:
        sub = Graph(n)
        sub.w = w.copy()
        r = stoer_wagner(sub)
        side = [v for i in r.side for v in members[i]]
        return r.value, side
    target = int(math.ceil(1 + n / math.sqrt(2)))
    best = (float("inf"), [])
    for _ in range(2):
        cw, labels = _contract_matrix(w, target, rng)
        k = cw.shape[0]
        new_members = [[] for _ in range(k)]
        for v in range(n):
            new_members[labels[v]].extend(members[v])
        if k < 2:  # disconnected: any component split has weight 0
            return 0.0, list(new_members[0][: max(1, len(new_members[0]) // 2)])
        res = _ks_rec(cw, new_members, rng, base)
        if res[0] < best[0]:
            best = res
    return best


def karger_stein(g: Graph, seed: int = 0, repeats: int | None = None,
                 base: int = 12) -> CutResult:
    """Karger-Stein recursive contraction.

    Contracts to ``ceil(1 + n/sqrt(2))`` vertices and recurses twice, giving
    success probability Omega(1/log n) per call at O(n^2 log n) cost. The
    call is repeated ``repeats`` times (default ``ceil(log2(n)/2)``, at least 2) to push the failure probability down.
    """
    if g.n < 2:
        raise ValueError("need at least two vertices")
    rng = np.random.default_rng(seed)
    if repeats is None:
        repeats = max(2, int(math.ceil(math.log2(max(g.n, 2)))) // 2)
    best = CutResult(float("inf"), ())
    for _ in range(repeats):
        val, side = _ks_rec(g.w, [[i] for i in range(g.n)], rng, base)
        if val < best.value:
            best = CutResult(float(val), tuple(sorted(side)))
    return best


def count_min_cuts_estimate(g: Graph, trials: int, seed: int = 0) -> dict[tuple[int, ...], int]:
    """Histogram of distinct minimum cuts (canonical side containing vertex 0).

    A graph has at most n(n-1)/2 minimum cuts; contraction finds each with
    probability >= 2/(n(n-1)). The histogram is built from ``trials`` runs
    that achieve the best value seen.
    """
    rng = np.random.default_rng(seed)
    runs = [karger_contract(g, rng) for _ in range(trials)]
    best = min(r.value for r in runs)
    hist: dict[tuple[int, ...], int] = {}
    for r in runs:
        if abs(r.value - best) < 1e-9:
            side = set(r.side)
            if 0 not in side:
                side = set(range(g.n)) - side
            key = tuple(sorted(side))
            hist[key] = hist.get(key, 0) + 1
    return hist
