"""Timing and operation-count benchmarks."""
from __future__ import annotations

import time
from typing import Callable

from .algorithms import ALGORITHMS, MSTResult, kruskal
from .graph import Graph, random_connected_graph
from .unionfind import COMPRESSION_RULES, UNION_RULES, UnionFind


def time_call(fn: Callable[[], MSTResult], repeat: int = 3) -> tuple[float, MSTResult]:
    """Best-of-``repeat`` wall time in milliseconds and the last result."""
    best = float("inf")
    res = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        res = fn()
        best = min(best, time.perf_counter() - t0)
    assert res is not None
    return best * 1e3, res


def compare_algorithms(g: Graph, repeat: int = 3) -> dict[str, tuple[float, float]]:
    """Map algorithm name -> (milliseconds, tree weight)."""
    out = {}
    for name, fn in ALGORITHMS.items():
        ms, r = time_call(lambda fn=fn: fn(g), repeat)
        out[name] = (ms, r.weight)
    return out


def density_sweep(n: int, factors: list[int], seed: int = 0) -> list[tuple[int, dict[str, tuple[float, float]]]]:
    """Run all algorithms for m = factor * n on random graphs."""
    return [(f * n, compare_algorithms(random_connected_graph(n, f * n, seed), repeat=2)) for f in factors]


def unionfind_steps(g: Graph) -> dict[tuple[str, str], int]:
    """Parent-pointer steps used by Kruskal for each heuristic combination."""
    out = {}
    for u in UNION_RULES:
        for c in COMPRESSION_RULES:
            out[(u, c)] = kruskal(g, u, c).stats["uf_steps"]
    return out


def adversarial_chain_steps(n: int, union: str, compression: str) -> int:
    """Steps for a worst-case-ish workload: link a chain then query every node."""
    uf = UnionFind(n, union, compression)
    for i in range(n - 1):
        uf.union(i + 1, i)
    for _ in range(3):
        for i in range(n):
            uf.find(i)
    return uf.steps
