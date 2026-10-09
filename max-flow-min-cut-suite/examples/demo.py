"""Benchmark the three max-flow solvers and show a min-cut certificate."""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flow_suite import (SOLVERS, random_network, layered_network, bipartite_matching,
                        min_cut, verify_certificate)


def bench(label, make):
    row = []
    for name, fn in SOLVERS.items():
        g, s, t = make()
        t0 = time.perf_counter()
        v = fn(g, s, t)
        row.append((name, v, time.perf_counter() - t0))
    assert len({v for _, v, _ in row}) == 1
    print(f"{label:<28} flow={row[0][1]:<7}" + "  ".join(f"{n}={d*1000:8.1f}ms" for n, _, d in row))


bench("random n=400 m=2400", lambda: random_network(400, 2400, 100, 1))
bench("random n=1000 m=6000", lambda: random_network(1000, 6000, 100, 2))
bench("layered 6x12", lambda: layered_network(6, 12, 50, 3))
bench("layered 10x20", lambda: layered_network(10, 20, 50, 4))
import random
rng = random.Random(5)
pairs = list({(rng.randrange(150), rng.randrange(150)) for _ in range(600)})
bench("matching 150x150", lambda: bipartite_matching(150, 150, pairs))

g, s, t = random_network(30, 90, 25, 7)
cut = min_cut(g, s, t)
print("\nmin-cut value:", cut.value, "| source side size:", len(cut.source_side),
      "| cut edges:", len(cut.cut_edges), "| certificate valid:", verify_certificate(g, s, t, cut))
