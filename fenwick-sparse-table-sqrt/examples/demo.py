"""Correctness cross-check and timing of range-query structures."""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from range_queries import (DisjointSparseTable, Fenwick, RangeFenwick, SparseTable,
                           SqrtRangeAdd, mo_distinct_counts)
from range_queries.bench import bench_static_min, bench_sum, random_ranges


def main() -> None:
    rng = random.Random(42)
    n = 5000
    a = [rng.randrange(1000) for _ in range(n)]
    qs = random_ranges(n, 2000, rng)

    f, rf, sp = Fenwick(a), RangeFenwick(a), SparseTable(a)
    ds, sq = DisjointSparseTable(a), SqrtRangeAdd(a)
    for lo, hi in qs:
        assert f.sum(lo, hi) == ds.query(lo, hi) == rf.sum(lo, hi) == sq.sum(lo, hi)
        assert sp.query(lo, hi) == min(a[lo:hi])
    print(f"cross-checked 2000 random queries on n={n}: all structures agree")

    for lo, hi, d in [(100, 900, 5), (400, 4000, -3)]:
        rf.range_add(lo, hi, d)
        sq.range_add(lo, hi, d)
    print("after range updates, sum[0,5000):", rf.sum(0, n), "==", sq.sum(0, n))

    b = [rng.randrange(50) for _ in range(n)]
    print("distinct counts (Mo):", mo_distinct_counts(b, qs[:5]),
          "expected", [len(set(b[lo:hi])) for lo, hi in qs[:5]])

    print("\nUpdate+sum workload, n=20000, 4000 ops (seconds):")
    for k, v in bench_sum(20000, 4000).items():
        print(f"  {k:<8}{v:.4f}")
    print("Static range-min, n=100000, 5000 queries (seconds):")
    for k, v in bench_static_min(100000, 5000).items():
        print(f"  {k:<13}{v:.4f}")


if __name__ == "__main__":
    main()
