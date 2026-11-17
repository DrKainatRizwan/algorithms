# Road-Network Shortest Paths: Bidirectional Dijkstra, ALT Landmarks and Contraction Hierarchies

## Overview
From-scratch implementations of four exact point-to-point shortest-path algorithms for static road-like graphs, using only the standard library: plain Dijkstra, bidirectional Dijkstra, ALT (A* with landmarks and triangle-inequality bounds) and contraction hierarchies (CH) with witness searches and path unpacking. A synthetic road generator (perturbed grid with fast "highway" edges and asymmetric congestion) provides test networks.

## How it works
- **Dijkstra / bidirectional Dijkstra** – the baseline; the bidirectional variant grows a forward and a backward ball and stops once `top_f + top_b >= mu`, where `mu` is the best meeting distance.
- **ALT** – pick landmarks by farthest-point selection, precompute `d(L,·)` and `d(·,L)`; the heuristic `max_L max(d(L,t)-d(L,v), d(v,L)-d(t,L))` is admissible and consistent, so A* stays exact.
- **Contraction hierarchies** – vertices are contracted in order of a lazily updated priority (edge difference + deleted neighbours). Contracting `v` inserts shortcut `u→w` only if a bounded witness search finds no path `u→w` avoiding `v` that is at least as short. Queries run two upward-only Dijkstra searches and meet at the highest-ranked vertex; shortcuts are unpacked recursively into original edges.

## Complexity
| Algorithm | Preprocessing | Space | Query |
|---|---|---|---|
| Dijkstra | – | O(m) | O(m + n log n) |
| Bidirectional | – | O(m) | same worst case, ~half the search space |
| ALT (k landmarks) | k Dijkstra runs, O(k(m + n log n)) | O(kn) | A*, graph dependent |
| CH | witness searches, graph dependent | O(m + shortcuts) | upward search spaces, typically polylogarithmic on road graphs |

## Usage
```python
from road_routing import grid_road_network, largest_scc_subgraph, ContractionHierarchy, ALT

g = largest_scc_subgraph(grid_road_network(40, 40, seed=1))
ch = ContractionHierarchy(g)
r = ch.query(1309, 228)
print(r.dist, r.path)
print(ALT(g, k=8).query(1309, 228).dist)
```
Run `python examples/demo.py` for the benchmark and `python -m pytest -q` for the tests.

## Results
1600-vertex network (5780 edges), 200 random queries, all algorithms agree with Dijkstra:

| Algorithm | Settled / query | ms / query |
|---|---|---|
| Dijkstra | 784.0 | 0.687 |
| Bidirectional | 406.0 | 0.490 |
| ALT (8 landmarks) | 62.8 | 0.485 |
| CH | 110.2 | 0.212 |

CH preprocessing took about 2.4 s and added 10385 shortcuts; ALT preprocessing took 0.03 s.

## References
- Goldberg, A. V., Harrelson, C. "Computing the Shortest Path: A* Search Meets Graph Theory", SODA 2005.
- Geisberger, R., Sanders, P., Schultes, D., Delling, D. "Contraction Hierarchies: Faster and Simpler Hierarchical Routing in Road Networks", WEA 2008.
- Pohl, I. "Bi-directional Search", Machine Intelligence 6, 1971.

---

**Author:** Dr. Kainat Rizwan
