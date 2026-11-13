# Graph Matching Toolkit: Hopcroft-Karp, Edmonds' Blossom and Stable Marriage

## Overview

From-scratch implementations of the classic matching algorithms in pure Python:

- **Bipartite**: Kuhn's augmenting paths, Hopcroft-Karp, König minimum vertex cover and a Hall's-theorem violator certificate.
- **General graphs**: Edmonds' blossom algorithm for maximum cardinality matching, plus a greedy baseline.
- **Stable matching**: Gale-Shapley deferred acceptance, the hospitals/residents generalisation, a blocking-pair checker and brute-force enumeration of all stable matchings.
- **Verification**: matching validity checks and an exact bitmask-DP reference solver used to cross-check the blossom code.

## How it works

*Hopcroft-Karp* alternates a BFS, which layers the graph by shortest augmenting-path length from free left vertices, with a DFS that extracts a maximal set of vertex-disjoint shortest augmenting paths. Path lengths strictly increase between phases, so only O(√V) phases are needed.

*Edmonds' blossom* grows an alternating forest from each exposed vertex. When an edge joins two even vertices, an odd cycle (blossom) is found via a lowest-common-ancestor query and contracted implicitly by redirecting a `base` array; the search continues on the contracted graph until an exposed vertex is reached and the path is flipped.

*König's theorem* turns a maximum bipartite matching into a minimum vertex cover: take the vertices reachable from free left vertices by alternating paths, and cover the unreachable left vertices and reachable right vertices.

*Gale-Shapley* has free proposers propose in preference order; receivers keep their best offer so far. The result is stable and optimal for the proposing side.

## Complexity

| Algorithm | Time | Space |
|-----------|------|-------|
| Kuhn | O(VE) | O(V) |
| Hopcroft-Karp | O(E√V) | O(V) |
| Edmonds blossom | O(V³) | O(V) |
| Gale-Shapley | O(n²) | O(n²) |

## Usage

```python
from graph_matching import hopcroft_karp, edmonds_blossom, gale_shapley, random_preferences

adj = [[0, 1], [0], [1, 2]]            # left vertex -> right neighbours
print(hopcroft_karp(adj, 3))           # [1, 0, 2]

triangle_plus_edge = [[1, 2], [0, 2], [0, 1, 3], [2]]
print(edmonds_blossom(triangle_plus_edge))   # perfect matching on 4 vertices

men, women = random_preferences(5, seed=0)
print(gale_shapley(men, women))
```

Run the tests with `python -m pytest -q` and the demo with `python examples/demo.py`.

## Results

Output of `examples/demo.py` (single-threaded CPython, best of 2):

| Instance | Matching size | Hopcroft-Karp | Kuhn |
|----------|---------------|---------------|------|
| 1500+1500 vertices, 4,510 edges | 1380 | 5.3 ms | 17.8 ms |
| 1500+1500 vertices, 11,290 edges | 1499 | 6.0 ms | 96.7 ms |

Blossom on a random 400-vertex graph (p = 0.01) finds a matching of 196 edges in 1.7 ms versus 166 for greedy. On 16-vertex graphs it equals the exact optimum in all demo cases (7, 7, 8). Gale-Shapley on 200×200 random preferences yields 0 blocking pairs, with a mean proposer rank of 8.12, in 2.1 ms.

## References

- J. Hopcroft, R. Karp, "An n^{5/2} algorithm for maximum matchings in bipartite graphs", SIAM J. Comput., 1973.
- J. Edmonds, "Paths, trees, and flowers", Canadian J. Math., 1965.
- D. Gale, L. Shapley, "College admissions and the stability of marriage", Amer. Math. Monthly, 1962.
- D. König, "Gráfok és mátrixok", Matematikai és Fizikai Lapok, 1931.
- D. Gusfield, R. Irving, *The Stable Marriage Problem: Structure and Algorithms*, MIT Press, 1989.

---

**Author:** Dr. Kainat Rizwan
