"""Compare global min-cut algorithms on random and planted-cut graphs."""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from global_mincut import (GomoryHuTree, cut_value, karger_contract, karger_stein,
                           planted_cut_graph, random_graph, stoer_wagner,
                           stoer_wagner_heap)


def timed(fn, *a, **k):
    t = time.perf_counter()
    r = fn(*a, **k)
    return r, (time.perf_counter() - t) * 1000


print("== Exact vs randomised on planted-cut graphs ==")
print(f"{'n':>4} {'m':>6} {'planted':>8} {'SW':>5} {'SWheap':>7} {'KS':>5} "
      f"{'SW ms':>8} {'heap ms':>8} {'KS ms':>8}")
for n in (20, 40, 60, 80):
    g, planted = planted_cut_graph(n // 2, n // 2, 0.4, 3, seed=n)
    a, ta = timed(stoer_wagner, g)
    b, tb = timed(stoer_wagner_heap, g)
    c, tc = timed(karger_stein, g, 1)
    print(f"{n:>4} {g.num_edges():>6} {planted:>8.0f} {a.value:>5.0f} {b.value:>7.0f} "
          f"{c.value:>5.0f} {ta:>8.1f} {tb:>8.1f} {tc:>8.1f}")

print("\n== Single-run Karger success rate (n=16, 400 runs) ==")
g = random_graph(16, 0.3, 5, 1, 3)
opt = stoer_wagner(g).value
rng = np.random.default_rng(0)
hits = sum(karger_contract(g, rng).value == opt for _ in range(400))
print(f"optimum={opt:.0f}  empirical success={hits/400:.3f}  "
      f"theory lower bound={2/(16*15):.4f}")

print("\n== Gomory-Hu tree (n=12) ==")
g = random_graph(12, 0.3, 2, 1, 5)
tree = GomoryHuTree(g)
print("tree edges (v, parent, weight):", [(v, p, int(w)) for v, p, w in tree.tree_edges()][:6], "...")
print("min cut(0, 11) =", tree.min_cut_value(0, 11), " global =", tree.global_min_cut().value,
      " SW =", stoer_wagner(g).value)
