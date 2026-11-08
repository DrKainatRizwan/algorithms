"""Demo: near-duplicate detection and approximate nearest-neighbour search with LSH."""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lsh_toolkit import (MinHasher, MinHashLSHIndex, SimHashIndex, SimHasher, VectorLSHIndex,
                         char_shingles, collision_probability, exact_knn, hamming, jaccard,
                         make_clusters, make_documents, optimal_bands, recall_at_k)


def main() -> None:
    print("== MinHash accuracy (true Jaccard 0.333) ==")
    a, b = set(range(0, 600)), set(range(300, 900))
    for m in (16, 64, 256, 1024):
        mh = MinHasher(m, seed=1)
        est = mh.estimate(mh.signature(a), mh.signature(b))
        print(f"  perms={m:5d} estimate={est:.3f} abs_err={abs(est - jaccard(a, b)):.3f}")

    print("\n== Banded index: S-curve for threshold 0.7, 128 perms ==")
    bands, rows = optimal_bands(128, 0.7)
    print(f"  chosen bands={bands} rows={rows}")
    for s in (0.3, 0.5, 0.7, 0.9):
        print(f"  P[candidate | J={s}] = {collision_probability(s, bands, rows):.3f}")

    print("\n== Near-duplicate search over 600 documents ==")
    docs, group = make_documents(100, 5, 0.05, seed=3)
    sh = [char_shingles(d, 5) for d in docs]
    idx = MinHashLSHIndex.for_threshold(MinHasher(128, seed=0), 0.5)
    t0 = time.perf_counter()
    for i, s in enumerate(sh):
        idx.insert(i, s)
    t_idx = time.perf_counter() - t0
    found = total = cands = 0
    for i, g in enumerate(group):
        res = {k for k, _ in idx.query(sh[i], 0.4)}
        mates = {j for j, gj in enumerate(group) if gj == g}
        found += len(res & mates)
        total += len(mates)
        cands += len(idx.candidates(sh[i]))
    print(f"  indexing {t_idx:.2f}s, recall={found / total:.3f}, "
          f"avg candidates={cands / len(docs):.1f} of {len(docs)}")

    print("\n== SimHash Hamming-ball lookup (k=3) ==")
    sim = SimHasher(64)
    fps = [sim.text_fingerprint(d) for d in docs]
    sidx = SimHashIndex(64, k=3)
    for f in fps:
        sidx.add(f)
    hits = sidx.query(fps[0])
    print(f"  doc 0 matches: {[(i, d) for i, d in hits]}  (group-mates: "
          f"{[j for j, g in enumerate(group) if g == group[0]]})")
    brute = sum(1 for f in fps if hamming(f, fps[0]) <= 3)
    print(f"  exactness check: index={len(hits)} brute-force={brute}")

    print("\n== Vector ANN: 4000 points, dim 32 ==")
    data = make_clusters(4000, 32, seed=4)
    rng = np.random.default_rng(5)
    queries = [data[i] + 0.05 * rng.standard_normal(32) for i in rng.choice(4000, 50, replace=False)]
    for metric, kw, h in (("cosine", {}, 8), ("l2", {"width": 6.0}, 4)):
        for tables in (4, 12):
            v = VectorLSHIndex(32, tables, h, metric=metric, seed=1, **kw).fit(data)
            rec, nc = [], []
            t0 = time.perf_counter()
            for q in queries:
                ids, _, n = v.query(q, 10)
                rec.append(recall_at_k(ids, exact_knn(data, q, 10, metric)))
                nc.append(n)
            print(f"  {metric:6s} tables={tables:2d} recall@10={np.mean(rec):.3f} "
                  f"candidates={np.mean(nc):7.1f}/4000 time={time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    main()
