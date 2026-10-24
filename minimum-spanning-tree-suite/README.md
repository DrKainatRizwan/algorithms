# Minimum Spanning Tree Suite

Kruskal, Prim (heap and dense), Borůvka and reverse-delete, built on a family of union-find variants, with post-hoc tools: optimality certificates, bottleneck paths, second-best MST and edge sensitivity.

## Overview

Package `mst_suite`:

- `graph.py` – edge-list multigraph and generators (random connected, grid, Euclidean complete, disconnected).
- `unionfind.py` – disjoint sets with naive/size/rank union and none/full/halving/splitting compression, step counters, and a rollback variant.
- `algorithms.py` – the MST/forest algorithms. Ties are broken by edge index, so all algorithms return the identical edge set even with repeated weights.
- `analysis.py` – binary-lifting path maxima, cycle-property certificate, bottleneck (minimax) paths, second-best MST, replacement edges and sensitivity.
- `benchmark.py` – timing and operation-count helpers.

## How it works

- **Kruskal** sorts edges and adds each one that joins two components (cut property).
- **Prim** grows one tree from a seed, always taking the lightest crossing edge; the heap version is lazy, the dense version scans an array.
- **Borůvka** lets every component pick its cheapest outgoing edge per round; components at least halve, so there are at most log₂ n rounds.
- **Reverse-delete** removes the heaviest edge of any cycle (cycle property); used as an independent cross-check.
- **Certificate**: a spanning forest is minimum iff no non-tree edge is lighter than the heaviest edge on its tree path.
- **Sensitivity**: non-tree edges are processed by weight and paint the unpainted tree edges on their path (skipping via union-find), giving each tree edge's cheapest replacement.

## Complexity

| Algorithm | Time | Space |
|-----------|------|-------|
| Kruskal | O(m log m) | O(n + m) |
| Prim (binary heap) | O(m log n) | O(n + m) |
| Prim (dense) | O(n²) | O(n²) |
| Borůvka | O(m log n) | O(n + m) |
| Reverse-delete | O(m (n + m)) | O(n + m) |
| Path-max preprocessing / query | O(n log n) / O(log n) | O(n log n) |
| Union-find (rank + compression) | O(α(n)) amortised | O(n) |

## Usage

```python
from mst_suite import kruskal, prim, random_connected_graph, is_minimum_spanning_forest, sensitivity

g = random_connected_graph(2000, 10000, seed=1)
t = kruskal(g)
print(t.weight, is_minimum_spanning_forest(g, t.edges))
print(prim(g).edges == t.edges)
sens = sensitivity(g, t.edges)   # tree edge -> max weight before it leaves the MST
```

Run `python examples/demo.py` for the full demo and `python -m pytest -q` for the tests.

## Results

From `examples/demo.py` (n=2000, m=10000; best of 3, in ms; all weights equal 237.069233):

| Algorithm | Time (ms) |
|-----------|-----------|
| Kruskal | 5.85 |
| Prim (heap) | 13.14 |
| Prim (dense) | 67.49 |
| Borůvka | 36.87 |

Density sweep (n=1000): at m=2000 Kruskal takes 1.59 ms vs 22.62 ms for dense Prim; at m=32000 the gap narrows to 9.77 ms vs 48.23 ms (dense Prim is roughly on par with heap Prim, 49.72 ms).

Union-find parent-pointer steps inside Kruskal: naive/no compression 2,515,461; rank/no compression 22,457; rank/halving 11,865. On a chain workload (n=2000) naive/no compression needs 5,997,000 steps versus 7,994 for union by rank.

## References

- J. B. Kruskal, "On the shortest spanning subtree of a graph and the traveling salesman problem", Proc. AMS, 1956.
- R. C. Prim, "Shortest connection networks and some generalizations", Bell System Technical Journal, 1957.
- O. Borůvka, "O jistém problému minimálním", Práce Mor. Přírodověd. Spol., 1926.
- R. E. Tarjan, "Efficiency of a good but not linear set union algorithm", JACM, 1975.
- R. E. Tarjan, "Applications of path compression on balanced trees", JACM, 1979.
- T. H. Cormen et al., *Introduction to Algorithms*, chapters 21 and 23.

---

**Author:** Dr. Kainat Rizwan
