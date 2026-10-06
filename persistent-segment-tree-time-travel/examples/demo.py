"""Benchmark and walkthrough for the persistent segment tree toolkit."""
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from persistent_segtree import (MAX, PersistentLazyTree, PersistentSegmentTree,
                                RangeKth, SnapshotArray, TimeTravelArray)


def main() -> None:
    rng = random.Random(2024)

    print("== Time-travel sensor log ==")
    tt = TimeTravelArray([0] * 8)
    for t, (i, v) in enumerate([(0, 5), (3, 9), (3, 2), (7, 11), (0, 1)], start=1):
        tt.set(t * 10, i, v)
    for t in (0, 15, 35, 50):
        print(f"t={t:>2}: total={tt.query(t, 0, 8):>3}  state={[tt.get(t, i) for i in range(8)]}")
    print("history of cell 3:", tt.history(3))

    print("\n== Range k-th smallest ==")
    a = [rng.randint(0, 1000) for _ in range(20000)]
    q = RangeKth(a)
    l, r = 1000, 15000
    print("median of a[1000:15000]:", q.kth(l, r, (r - l + 1) // 2),
          "| brute force:", sorted(a[l:r])[(r - l + 1) // 2 - 1])

    print("\n== Lazy persistent range add ==")
    lt = PersistentLazyTree([0] * 1000)
    v = lt.add(0, 100, 900, 3)
    v = lt.add(v, 0, 500, -1)
    print("sum v0:", lt.sum(0, 0, 1000), " sum latest:", lt.sum(v, 0, 1000))

    print("\n== Persistent tree vs full snapshots (n=20000, 2000 updates) ==")
    n, ups = 20000, 2000
    base = [rng.randint(0, 99) for _ in range(n)]
    tree = PersistentSegmentTree(base, MAX)
    snap = SnapshotArray(base, MAX)
    ops = [(rng.randrange(n), rng.randint(0, 999)) for _ in range(ups)]
    t0 = time.perf_counter()
    v = 0
    for i, x in ops:
        v = tree.update(v, i, x)
    t_tree = time.perf_counter() - t0
    t0 = time.perf_counter()
    s = 0
    for i, x in ops:
        s = snap.update(s, i, x)
    t_snap = time.perf_counter() - t0
    print(f"update time  persistent: {t_tree:.3f}s   snapshots: {t_snap:.3f}s")
    extra = tree.nodes_created - (2 * n - 1)
    print(f"extra nodes: {extra} (~{extra / ups:.1f}/update) vs {n * ups} cells for snapshots")
    qs = [(rng.randrange(ups + 1), rng.randrange(n // 2)) for _ in range(300)]
    ok = all(tree.query(ver, l, l + n // 2) == snap.query(ver, l, l + n // 2) for ver, l in qs)
    print("all 300 historical range-max queries agree:", ok)


if __name__ == "__main__":
    main()
