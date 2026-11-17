"""Query benchmarking helpers shared by the demo and tests."""
from __future__ import annotations

import random
import time
from typing import Callable

from .dijkstra import QueryResult
from .graph import Graph


def random_pairs(g: Graph, k: int, seed: int = 0) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    return [(rng.randrange(g.n), rng.randrange(g.n)) for _ in range(k)]


def run_queries(fn: Callable[[int, int], QueryResult], pairs: list[tuple[int, int]]) -> dict:
    """Run ``fn`` on all pairs; return distances, mean settled count and ms/query."""
    t0 = time.perf_counter()
    res = [fn(s, t) for s, t in pairs]
    el = time.perf_counter() - t0
    return {"dists": [r.dist for r in res],
            "settled": sum(r.settled for r in res) / len(res),
            "ms": 1000 * el / len(res)}
