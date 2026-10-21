# Strongly Connected Components, 2-SAT Solver and Implication Graph Toolkit

## Overview
From-scratch, pure Python implementations of three linear-time SCC algorithms (Tarjan, Kosaraju-Sharir, Gabow's path-based), a 2-SAT solver built on the implication graph, and supporting tools: condensation DAGs, bitset transitive closure, transitive reduction, backbone computation, model enumeration, DIMACS I/O, and reductions such as 2-colouring/scheduling and Robbins orientation. All traversals are iterative, so graphs with 50,000-long paths are fine.

## How it works
* **SCC.** Tarjan keeps DFS indices and low-links with a stack; Kosaraju runs DFS on `G` for finishing order, then on the transpose; Gabow replaces low-links with a boundary stack. The condensation is a DAG.
* **2-SAT.** Each clause `(a ∨ b)` yields implications `¬a → b` and `¬b → a`. The formula is unsatisfiable iff some `x` and `¬x` share an SCC (Aspvall–Plass–Tarjan). Otherwise set `x` true when its component comes later in topological order (for Tarjan ids: `comp[x] < comp[¬x]`). For UNSAT instances the solver returns the two implication chains `x ⇝ ¬x` and `¬x ⇝ x` as a certificate.
* **Backbone:** literal `l` is forced iff the formula under assumption `¬l` is UNSAT.
* **Reachability:** bitset DP over the condensation in reverse topological order.

## Complexity
| Operation | Time | Space |
|---|---|---|
| SCC (all three) | O(n + m) | O(n + m) |
| 2-SAT solve | O(n + k) for k clauses | O(n + k) |
| Backbone | O(n (n + k)) | O(n + k) |
| Transitive closure | O(m n / w) | O(n² / w) |

## Usage
```python
from graph_sat import TwoSat, Digraph, tarjan_scc

s = TwoSat(3)
s.add_clause(1, 2); s.add_clause(-1, 3); s.add_clause(-2, -3)
print(s.solve().assignment)      # {1: True, 2: False, 3: True}

comp = tarjan_scc(Digraph(3, [(0, 1), (1, 0), (1, 2)]))
```

Run `python -m pytest -q` (27 tests) and `python examples/demo.py`.

## Results (from `examples/demo.py`)
| Graph | Tarjan | Kosaraju | Gabow |
|---|---|---|---|
| n=20k, m=30k | 21 ms | 25 ms | 15 ms |
| n=20k, m=60k | 34 ms | 34 ms | 25 ms |
| n=50k, m=100k | 64 ms | 73 ms | 56 ms |

Random 2-SAT, n=2000 variables, 20 trials: satisfiable in 20/20 at clause ratio 0.5 and 0.8, 19/20 at 1.0, 1/20 at 1.2, 0/20 at 1.5, showing the sharp threshold at ratio 1. Planted instances: 100,000 variables / 200,000 clauses solved in 363 ms.

## References
* R. E. Tarjan, "Depth-first search and linear graph algorithms", SIAM J. Computing, 1972.
* M. Sharir, "A strong-connectivity algorithm and its applications in data flow analysis", 1981.
* H. N. Gabow, "Path-based depth-first search for strong and biconnected components", IPL, 2000.
* B. Aspvall, M. Plass, R. Tarjan, "A linear-time algorithm for testing the truth of certain quantified Boolean formulas", IPL, 1979.
* K. P. Eswaran, R. E. Tarjan, "Augmentation problems", SIAM J. Computing, 1976.

---

**Author:** Dr. Kainat Rizwan
