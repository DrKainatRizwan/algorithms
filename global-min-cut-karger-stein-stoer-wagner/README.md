# Global Minimum Cut: Stoer-Wagner, Karger, Karger-Stein and Gomory-Hu Trees

## Overview

From-scratch NumPy implementations of the main global minimum cut algorithms for weighted undirected graphs:

- **Stoer-Wagner** (dense `O(n^3)` and lazy-heap variants), deterministic and exact.
- **Karger's random contraction**, implemented as Kruskal over exponential edge keys so weighted sampling costs `O(m log m)` per run.
- **Karger-Stein** recursive contraction with a Stoer-Wagner base case.
- **Gomory-Hu cut tree** (Gusfield's algorithm over a Dinic max-flow), giving all-pairs min cuts and a global cut.
- Brute-force verification and planted-cut / random graph generators.

## How it works

*Stoer-Wagner* runs `n-1` phases. Each phase builds a maximum-adjacency ordering; the last vertex `t` and its predecessor `s` satisfy: the cut `({t}, rest)` is a minimum `s`-`t` cut. The best phase cut is recorded and `s`, `t` are merged.

*Karger* repeatedly picks an edge with probability proportional to its weight and contracts it until two super-vertices remain. A fixed minimum cut survives with probability at least `2/(n(n-1))`. Giving each edge an `Exp(1)/w` key and merging in key order samples the same distribution.

*Karger-Stein* contracts only to `ceil(1 + n/sqrt 2)` vertices (where survival is still ≥ 1/2) and recurses twice, lifting single-call success to `Omega(1/log n)`.

*Gomory-Hu* (Gusfield) performs `n-1` max-flow computations; the min `s`-`t` cut is the lightest edge on the tree path.

## Complexity

| Algorithm | Time | Space |
|---|---|---|
| Stoer-Wagner (dense) | `O(n^3)` | `O(n^2)` |
| Stoer-Wagner (heap) | `O(n m log n)` | `O(n + m)` |
| Karger (one run) | `O(m log m)` here | `O(m)` |
| Karger-Stein (one call) | `O(n^2 log n)` per contraction level, `O(n^2 log^2 n)`-style total | `O(n^2)` |
| Gomory-Hu (Gusfield) | `n-1` max flows | `O(n^2)` |

## Usage

```python
from global_mincut import planted_cut_graph, stoer_wagner, karger_stein, GomoryHuTree

g, planted = planted_cut_graph(20, 20, 0.4, cross_edges=3, seed=1)
print(stoer_wagner(g).value, karger_stein(g, seed=1).value)  # 3.0 3.0
print(GomoryHuTree(g).global_min_cut().value)
```

Run `python -m pytest -q` for the tests and `python examples/demo.py` for the benchmark.

## Results

Planted-cut graphs (two dense clusters, 3 crossing edges), timings from `examples/demo.py`:

| n | m | Stoer-Wagner ms | SW heap ms | Karger-Stein ms | cut found |
|---|---|---|---|---|---|
| 20 | 58 | 1.3 | 0.8 | 7.4 | 3 (all agree) |
| 40 | 185 | 7.0 | 3.4 | 32.0 | 3 |
| 60 | 399 | 14.3 | 7.7 | 72.1 | 3 |
| 80 | 686 | 31.4 | 18.5 | 126.7 | 3 |

A single Karger contraction run on a random 16-vertex graph (optimum 4) succeeded in 28.5% of 400 runs, well above the worst-case bound 2/(n(n-1)) ≈ 0.8%. On small instances the exact Stoer-Wagner solver is faster than the randomised methods in pure Python; the randomised algorithms are mainly of interest for their analysis and for enumerating near-minimum cuts.

## References

- M. Stoer and F. Wagner, "A simple min-cut algorithm", J. ACM 44(4), 1997.
- D. Karger, "Global min-cuts in RNC, and other ramifications of a simple min-cut algorithm", SODA 1993.
- D. Karger and C. Stein, "A new approach to the minimum cut problem", J. ACM 43(4), 1996.
- R. Gomory and T. Hu, "Multi-terminal network flows", SIAM J. 9(4), 1961.
- D. Gusfield, "Very simple methods for all pairs network flow analysis", SIAM J. Comput. 19(1), 1990.

---

**Author:** Dr. Kainat Rizwan
