"""Small timing harness comparing structures on identical workloads."""
from __future__ import annotations

import random
import time
from typing import Callable


def timed(fn: Callable[[], object]) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def random_ranges(n: int, q: int, rng: random.Random) -> list[tuple[int, int]]:
    out = []
    for _ in range(q):
        a, b = rng.randrange(n), rng.randrange(n)
        lo, hi = min(a, b), max(a, b) + 1
        out.append((lo, hi))
    return out


def bench_sum(n: int, q: int, seed: int = 0) -> dict[str, float]:
    """Interleaved point updates and range sums on several structures."""
    from .fenwick import Fenwick
    from .naive import NaiveArray
    from .sqrt_decomp import SqrtDecomposition

    rng = random.Random(seed)
    data = [rng.randrange(1000) for _ in range(n)]
    ops = [(rng.random() < 0.5, rng.randrange(n), rng.randrange(1000), r)
           for r in random_ranges(n, q, rng)]

    def run(s) -> None:
        for is_upd, i, v, (lo, hi) in ops:
            if is_upd:
                s.set(i, v) if hasattr(s, "set") else None
            else:
                s.sum(lo, hi)

    return {
        "naive": timed(lambda: run(NaiveArray(data))),
        "sqrt": timed(lambda: run(SqrtDecomposition(data))),
        "fenwick": timed(lambda: run(Fenwick(data))),
    }


def bench_static_min(n: int, q: int, seed: int = 0) -> dict[str, float]:
    """Static range-minimum: build time and total query time."""
    from .sparse_table import SparseTable
    from .sqrt_decomp import SqrtDecomposition

    rng = random.Random(seed)
    data = [rng.randrange(10**6) for _ in range(n)]
    qs = random_ranges(n, q, rng)
    res: dict[str, float] = {}
    st: list[SparseTable] = []
    res["sparse build"] = timed(lambda: st.append(SparseTable(data)))
    res["sparse query"] = timed(lambda: [st[0].query(lo, hi) for lo, hi in qs])
    sq = SqrtDecomposition(data)
    res["sqrt query"] = timed(lambda: [sq.min(lo, hi) for lo, hi in qs])
    res["naive query"] = timed(lambda: [min(data[lo:hi]) for lo, hi in qs])
    return res
