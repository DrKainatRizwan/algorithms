"""Correctness cross-check and timing of interval/segment-tree geometry queries."""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from geometry_trees import (CenteredIntervalTree, DynamicIntervalTree, RectangleIndex,
                            max_overlap_point, rect_intersections, skyline, timeit,
                            union_area)
from geometry_trees import naive
from geometry_trees.bench import random_intervals, random_rects


def main() -> None:
    rng = random.Random(7)

    n, q = 20000, 2000
    ivs = random_intervals(n, rng)
    pts = [rng.uniform(0, 1000) for _ in range(q)]
    tree = CenteredIntervalTree(ivs)
    dyn = DynamicIntervalTree(seed=1)
    for lo, hi in ivs:
        dyn.insert(lo, hi)
    assert all(tree.stab(p) == naive.stab(ivs, p) == dyn.stab(p) for p in pts[:50])
    print(f"Stabbing queries: {n} intervals, {q} queries")
    print(f"  centred tree   {timeit(lambda: [tree.stab(p) for p in pts]) * 1e3:8.1f} ms")
    print(f"  dynamic treap  {timeit(lambda: [dyn.stab(p) for p in pts]) * 1e3:8.1f} ms")
    print(f"  brute force    {timeit(lambda: [naive.stab(ivs, p) for p in pts[:200]]) * 1e4:8.1f} ms (est.)")
    print(f"  treap depth {dyn.depth()}, centred height {tree.height()}")

    depth, x = max_overlap_point(ivs)
    print(f"Max overlap depth {depth} at x={x:.2f} (brute force: {naive.max_overlap(ivs)})")

    m = 3000
    rects = random_rects(m, rng, span=1000, max_side=30)
    area = union_area(rects)
    pairs = rect_intersections(rects)
    t_sweep = timeit(lambda: rect_intersections(rects))
    t_naive = timeit(lambda: naive.rect_intersections(rects))
    print(f"Rectangles: {m}; union area {area:.1f}; intersecting pairs {len(pairs)}")
    print(f"  sweep {t_sweep * 1e3:.1f} ms vs brute force {t_naive * 1e3:.1f} ms; "
          f"agree: {pairs == naive.rect_intersections(rects)}")
    small = rects[:40]
    print(f"  union area check (40 rects): {union_area(small):.2f} vs "
          f"{naive.rect_union_area(small):.2f}")

    idx = RectangleIndex(rects)
    wins = [(x, y, x + 50, y + 50) for x, y in
            ((rng.uniform(0, 950), rng.uniform(0, 950)) for _ in range(300))]
    ok = all(idx.query(w) == naive.window_query(rects, w) for w in wins[:30])
    print(f"Window queries: index {timeit(lambda: [idx.query(w) for w in wins]) * 1e3:.1f} ms "
          f"for {len(wins)} windows; correct: {ok}")

    print("Skyline:", skyline([(2, 9, 10), (3, 7, 15), (5, 12, 12), (15, 20, 10), (19, 24, 8)]))


if __name__ == "__main__":
    main()
