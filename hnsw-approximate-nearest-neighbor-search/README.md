# HNSW Approximate Nearest-Neighbour Search from Scratch

## Overview
A NumPy implementation of Hierarchical Navigable Small World (HNSW) graphs for approximate k-nearest-neighbour search: incremental insertion, layered greedy/beam search, the diversity-based neighbour-selection heuristic, tombstone deletion, persistence, and graph diagnostics. An exact brute-force index provides ground truth, and a single-layer (flat NSW) variant and a simple closest-M selection variant act as baselines.

## How it works
- Each node draws a top layer `l = floor(-ln(U) / ln M)`, giving exponentially thinner upper layers.
- **Insert:** greedy descent (ef=1) from the entry point to layer `l+1`; on each layer from `min(l, L)` down to 0 run a beam search with width `ef_construction`, link the node to `M` selected neighbours, add reverse links and shrink any neighbour list exceeding its cap (`M` above layer 0, `2M` on layer 0).
- **Neighbour selection heuristic (Alg. 4):** scan candidates by distance and keep one only if it is closer to the new point than to every already kept neighbour; this preserves links between clusters. Pruned candidates fill remaining slots.
- **Query:** greedy descent through upper layers, then a beam search of width `ef` on layer 0.
- **Delete:** tombstones; deleted nodes still route traffic but are filtered from results.

## Complexity
- Query: expected O(log n) hops, roughly O(ef · M · d · log n) work.
- Insert: O(ef_construction · M · d · log n) expected.
- Memory: O(n · d) for vectors plus O(n · M) for links.

## Usage
```python
from hnsw_index import HNSWIndex, gaussian_clusters

X = gaussian_clusters(2000, 24, clusters=20, seed=0)
index = HNSWIndex(24, metric="l2", M=12, ef_construction=80, seed=1)
index.add_batch(X)
ids, dists = index.query(X[0] + 0.1, k=10, ef=50)
```
Metrics: `l2` (squared), `cosine`, `ip` (maximum inner product).

## Results
`python examples/demo.py` (2000 points, 24 dims, 20 Gaussian clusters, 100 queries, k=10, M=12, ef_construction=80):

| Variant | ef | recall@10 | dist evals / query |
|---|---|---|---|
| HNSW (heuristic) | 10 | 0.994 | 120 |
| HNSW (heuristic) | 50 | 1.000 | 152 |
| HNSW (simple select) | 10 | 0.832 | 111 |
| HNSW (simple select) | 100 | 0.980 | 190 |
| Flat NSW | 10 | 0.983 | 170 |
| Flat NSW | 50 | 1.000 | 199 |

Brute force needs 2000 distance evaluations per query. The simple selection variant produced a layer-0 graph that was not fully connected, which explains its recall ceiling; the heuristic graph was connected. At this small scale NumPy overhead means wall-clock time does not beat vectorised brute force, but the distance-evaluation count shows the sublinear behaviour.

## Tests
`python -m pytest -q` runs 16 seeded tests covering metrics, exact search, recall, monotonicity in `ef`, degree bounds, connectivity, determinism, deletion, persistence and the selection heuristic.

## References
- Y. Malkov, D. Yashunin, "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs", IEEE TPAMI 2020 (arXiv:1603.09320).
- Y. Malkov et al., "Approximate nearest neighbor algorithm based on navigable small world graphs", Information Systems 2014.

---

**Author:** Dr. Kainat Rizwan
