"""Build an HNSW index on clustered data and compare it with exact search."""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from hnsw_index import HNSWIndex, benchmark_index, gaussian_clusters
from hnsw_index.analysis import degree_stats, greedy_hops

N, DIM, K = 2000, 24, 10
X = gaussian_clusters(N, DIM, clusters=20, spread=0.8, seed=0)
Q = gaussian_clusters(100, DIM, clusters=20, spread=0.8, seed=0) + 0.1

for name, kw in [("HNSW (heuristic)", {}), ("HNSW (simple select)", {"heuristic": False}),
                 ("flat NSW", {"single_layer": True})]:
    idx = HNSWIndex(DIM, M=12, ef_construction=80, seed=1, **kw)
    t0 = time.perf_counter(); idx.add_batch(X)
    bt = time.perf_counter() - t0
    hops = np.mean([greedy_hops(idx, q) for q in Q])
    print(f"\n{name}: build {bt:.2f}s, levels {idx.level_histogram()}, "
          f"mean greedy hops {hops:.1f}, connected={idx.is_connected_layer0()}")
    print(f"{'ef':>5}{'recall@10':>11}{'ms/query':>10}{'dist evals':>12}")
    for r in benchmark_index(idx, X, Q, K, [10, 20, 50, 100]):
        print(f"{r['ef']:>5}{r['recall']:>11.3f}{r['ms_per_query']:>10.2f}{r['dist_evals']:>12.0f}")
    if not kw:
        print("brute force ms/query: %.2f (n=%d dist evals/query)" % (r["brute_ms"], N))
        print("layer stats:", {l: (s["nodes"], round(s["mean"], 1)) for l, s in degree_stats(idx).items()})
