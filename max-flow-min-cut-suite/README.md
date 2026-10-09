# Max-Flow Min-Cut Suite

From-scratch implementations of three classical maximum-flow algorithms — Edmonds-Karp, Dinic and FIFO push-relabel — on a shared residual-graph structure, plus minimum-cut extraction with certificate checking, and generators for random, layered and bipartite-matching networks.

## How it works

- **Residual network** (`graph.py`): arcs are stored in pairs (`i`, `i^1`), so pushing `d` units is `cap[i] -= d; cap[i^1] += d`. Flow validity (capacity and conservation) can be checked independently.
- **Edmonds-Karp** (`edmonds_karp.py`): repeatedly augments along a shortest BFS path; at most `VE/2` augmentations.
- **Dinic** (`dinic.py`): builds BFS level graphs and saturates them with an iterative blocking-flow DFS using current-arc pointers and dead-end pruning.
- **Push-relabel** (`push_relabel.py`): saturates source arcs, then pushes excess along admissible arcs and relabels stuck vertices. Heights go up to `2n-1` so leftover excess returns to the source, yielding a genuine flow. Uses a FIFO queue and the gap heuristic.
- **Min cut** (`mincut.py`): after max flow, the vertices reachable from `s` in the residual graph form the source side of a minimum cut; `verify_certificate` confirms cut capacity equals the flow value.
- **Generators** (`generators.py`): random graphs, dense layered graphs, bipartite matching and edge-disjoint paths reductions.

## Complexity

| Algorithm | Time | Space |
|-----------|------|-------|
| Edmonds-Karp | O(V E²) | O(V + E) |
| Dinic | O(V² E); O(E√V) on unit networks | O(V + E) |
| Push-relabel (FIFO) | O(V³) | O(V + E) |

## Usage

```python
from flow_suite import FlowNetwork, dinic, min_cut

g = FlowNetwork(4)
g.add_edge(0, 1, 3); g.add_edge(0, 2, 2)
g.add_edge(1, 3, 2); g.add_edge(2, 3, 3)
cut = min_cut(g, 0, 3)
print(cut.value, cut.cut_edges)
```

## Results

Output of `python examples/demo.py` (single run, pure Python):

| Instance | Max flow | Edmonds-Karp | Dinic | Push-relabel |
|----------|---------:|-------------:|------:|-------------:|
| random n=400, m=2400 | 191 | 0.6 ms | 3.5 ms | 4.2 ms |
| random n=1000, m=6000 | 424 | 17.8 ms | 12.0 ms | 20.2 ms |
| layered 6x12 | 318 | 3.8 ms | 0.4 ms | 1.1 ms |
| layered 10x20 | 407 | 40.7 ms | 1.0 ms | 3.4 ms |
| matching 150x150 | 145 | 6.2 ms | 2.2 ms | 3.9 ms |

On dense layered graphs Dinic's blocking flows clearly beat one-path-at-a-time augmentation; on sparse random graphs with short paths Edmonds-Karp is competitive.

Run the tests with `python -m pytest -q` (18 tests, including cross-solver agreement, brute-force min-cut comparison and flow validity checks).

## References

- Edmonds, J. & Karp, R. (1972). Theoretical improvements in algorithmic efficiency for network flow problems. *JACM* 19(2).
- Dinic, E. A. (1970). Algorithm for solution of a problem of maximum flow in a network with power estimation. *Soviet Math. Doklady* 11.
- Goldberg, A. & Tarjan, R. (1988). A new approach to the maximum-flow problem. *JACM* 35(4).
- Cormen, Leiserson, Rivest, Stein. *Introduction to Algorithms*, ch. 26.

---

**Author:** Dr. Kainat Rizwan
