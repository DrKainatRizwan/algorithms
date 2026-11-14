"""Compare linear probing, Robin Hood, cuckoo and hopscotch hash tables."""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from open_addressing import (  # noqa: E402
    CuckooTable,
    HopscotchTable,
    IdentityHash,
    LinearProbingTable,
    MultiplyShift,
    RobinHoodTable,
    TabulationHash,
)
from open_addressing.benchmark import (  # noqa: E402
    churn,
    load_sweep,
    make_workload,
    probe_profile,
)

CAP = 4096
SCHEMES = {
    "linear": lambda c: LinearProbingTable(c, max_load=0.99, seed=1),
    "robin-hood": lambda c: RobinHoodTable(c, max_load=0.99, seed=1),
    "cuckoo b=1": lambda c: CuckooTable(c, bucket_size=1, max_load=0.99, seed=1),
    "cuckoo b=4": lambda c: CuckooTable(c, bucket_size=4, max_load=0.99, seed=1),
    "hopscotch": lambda c: HopscotchTable(c, neighborhood=32, max_load=0.99, seed=1),
}


def basics() -> None:
    print("== Basic use ==")
    t = RobinHoodTable()
    for i, w in enumerate("the quick brown fox jumps over the lazy dog".split()):
        t[w] = i
    print(t, "->", dict(sorted(t.items())))
    del t["fox"]
    print("after delete:", len(t), "keys; 'fox' in t ->", "fox" in t)


def sweep() -> None:
    loads = [0.3, 0.5, 0.7, 0.85, 0.95]
    res = load_sweep(SCHEMES, loads, capacity=CAP, seed=0)
    print(f"\n== Probes per lookup, capacity {CAP}, random keys ==")
    print("(hit mean/max | miss mean; '--' = table had to grow or rehash)")
    print(f"{'scheme':<12}" + "".join(f"{l:>20.0%}" for l in loads))
    for name, row in res.items():
        cells = []
        for l in loads:
            p = row[l]
            cells.append("--" if p is None else f"{p.mean_hit:.2f}/{p.max_hit:<3d}| {p.mean_miss:.2f}")
        print(f"{name:<12}" + "".join(f"{c:>20}" for c in cells))


def hash_quality() -> None:
    print("\n== Hash function quality on strided keys (multiples of 1024), n=1500 ==")
    present, absent = make_workload("strided", 1500)
    for label, fam in [("identity", IdentityHash), ("multiply-shift", MultiplyShift), ("tabulation", TabulationHash)]:
        t = RobinHoodTable(4096, hash_factory=fam, seed=3)
        for k in present:
            t[k] = k
        p = probe_profile(t, present, absent)
        print(f"{label:<15} mean hit {p.mean_hit:8.2f}  max {p.max_hit:5d}  mean miss {p.mean_miss:8.2f}")


def churn_test() -> None:
    print("\n== Steady-state churn: 2000 live keys, 20000 delete+insert rounds ==")
    for name, cls in [("linear", LinearProbingTable), ("robin-hood", RobinHoodTable)]:
        hit, miss = churn(cls(4096, max_load=0.8, seed=2), 2000, 20000, seed=2)
        print(f"{name:<12} mean hit {hit:.2f}  mean miss {miss:.2f}")


def timing() -> None:
    print("\n== Wall-clock: 20000 inserts + 20000 hits + 20000 misses ==")
    present, absent = make_workload("random", 20000, seed=5)
    for name, make in [
        ("linear", lambda: LinearProbingTable(seed=1)),
        ("robin-hood", lambda: RobinHoodTable(seed=1)),
        ("cuckoo b=4", lambda: CuckooTable(bucket_size=4, seed=1)),
        ("hopscotch", lambda: HopscotchTable(seed=1)),
        ("dict", dict),
    ]:
        t = make()
        t0 = time.perf_counter()
        for k in present:
            t[k] = k
        for k in present:
            t[k]
        for k in absent:
            k in t
        print(f"{name:<12} {time.perf_counter() - t0:6.3f} s")


if __name__ == "__main__":
    basics()
    sweep()
    hash_quality()
    churn_test()
    timing()
