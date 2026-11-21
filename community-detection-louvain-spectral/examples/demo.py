"""Compare community detection algorithms on synthetic benchmarks."""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from graph_communities import (girvan_newman, label_propagation, louvain, modularity,
                               nmi, planted_partition, ring_of_cliques,
                               spectral_clustering, disconnected_communities)


def timed(f):
    t = time.perf_counter()
    r = f()
    return r, (time.perf_counter() - t) * 1000


print("== Planted partition: 4 blocks x 30 nodes, p_in=0.4, varying p_out ==")
print(f"{'p_out':>6} {'method':<16} {'Q':>7} {'NMI':>6} {'ms':>8}")
for p_out in (0.02, 0.06, 0.10):
    g, truth = planted_partition([30] * 4, 0.4, p_out, seed=11)
    runs = {
        "louvain": lambda: louvain(g, seed=0).labels,
        "label-prop": lambda: label_propagation(g, seed=0),
        "spectral(k=4)": lambda: spectral_clustering(g, 4, seed=0),
    }
    for name, f in runs.items():
        lab, ms = timed(f)
        if lab is None:
            continue
        print(f"{p_out:>6.2f} {name:<16} {modularity(g, lab):>7.3f} {nmi(lab, truth):>6.3f} {ms:>8.1f}")

print("\n== Girvan-Newman: 3 blocks x 12 nodes (O(m^2 n), so kept small) ==")
g, truth = planted_partition([12] * 3, 0.6, 0.03, seed=2)
(lab, q, trace), ms = timed(lambda: girvan_newman(g))
print(f"best Q={q:.3f} NMI={nmi(lab, truth):.3f} communities={len(set(lab))} time={ms:.0f} ms")
print(f"Louvain on same graph: Q={louvain(g, seed=0).modularity:.3f}")

print("\n== Resolution limit: ring of 30 cliques of size 5 ==")
g, truth = ring_of_cliques(30, 5)
for gamma in (1.0, 2.0, 3.0):
    res = louvain(g, resolution=gamma, seed=0)
    print(f"gamma={gamma}: communities={len(set(res.labels)):>2}  NMI vs cliques={nmi(res.labels, truth):.3f}"
          f"  Q(gamma=1)={modularity(g, res.labels):.3f}")
print(f"Q of the true clique partition (gamma=1): {modularity(g, truth):.3f}")

print("\n== Louvain on a 600-node graph ==")
g, truth = planted_partition([100] * 6, 0.15, 0.01, seed=5)
res, ms = timed(lambda: louvain(g, seed=0))
print(f"edges={g.num_edges()} levels={len(res.levels)} communities={len(set(res.labels))} "
      f"Q={res.modularity:.3f} NMI={nmi(res.labels, truth):.3f} time={ms:.0f} ms "
      f"disconnected={len(disconnected_communities(g, res.labels))}")
