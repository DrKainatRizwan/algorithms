# Topological Sort, Critical Path Method and Project Scheduling Engine

## Overview
A from-scratch toolkit for scheduling work with precedence constraints. It covers topological ordering (Kahn, DFS, lexicographic, enumeration and counting), cycle extraction, the Critical Path Method with total and free float, PERT three-point estimation with a Monte Carlo cross-check, and resource-constrained project scheduling (RCPSP) via serial and parallel schedule generation schemes with six priority rules. Python 3.12, standard library + numpy only.

## How it works
- **Ordering** (`graph.py`): Kahn's algorithm repeatedly removes zero in-degree nodes (a heap gives the lexicographically smallest order); an iterative DFS yields reverse post-order and extracts an explicit cycle when one exists. Orders are enumerated by backtracking and counted with a DP over downsets.
- **CPM** (`cpm.py`): a forward pass computes ES/EF, a backward pass computes LS/LF; total float = LS − ES, free float = min(ES of successors) − EF. Zero-float activities are critical and all critical paths can be enumerated.
- **PERT** (`pert.py`): mean = (o + 4m + p)/6, variance = ((p − o)/6)². Project time is approximated as normal along the critical chain; Beta-PERT sampling captures the merge bias that the analytic estimate ignores.
- **RCPSP** (`rcpsp.py`): the serial scheme places eligible activities one at a time at the earliest feasible start; the parallel scheme advances time and starts whatever fits. Priority rules: LFT, LST, minimum slack, shortest processing time, greatest resource demand, most total successors. A validator checks precedence and capacity, and a lower bound combines the critical path with resource work content.

## Complexity
| Part | Time | Space |
|---|---|---|
| Kahn / DFS order | O(V + E) (O(V log V + E) with heap) | O(V) |
| CPM | O(V + E) | O(V) |
| Count orders | O(2^V · V) | O(2^V) |
| Monte Carlo PERT | O(N (V + E)) vectorised | O(N V) |
| Serial / parallel SGS | O(V² · I) with I schedule intervals | O(V) |

## Usage
```python
from project_scheduler import Activity, Resource, critical_path_method, schedule_serial

acts = [Activity("A", 3), Activity("B", 4, ("A",), {"crew": 2}), Activity("C", 2, ("A",), {"crew": 2})]
print(critical_path_method(acts).critical_activities)
print(schedule_serial(acts, [Resource("crew", 2)], rule="lft"))
```
Run `python examples/demo.py` for the full walk-through and `python -m pytest -q` for the 20 tests.

## Results (from `examples/demo.py`)
Seven-task software project: duration 23, one critical path (Design → Backend → Integration → Testing → Release), 7 topological orders.

PERT network (4 tasks): analytic E = 13.17, sd = 1.72, P(≤15) = 0.856; Monte Carlo (20 000 runs) mean = 13.49, P(≤15) = 0.794, showing the optimistic bias at merge points.

RCPSP, 60 random activities, 2 resources (CPM bound 35, resource lower bound 144.7):

| Rule | Serial | Parallel |
|---|---|---|
| LFT | 162 | 156 |
| MSLK | 160 | 165 |
| SPT | 173 | 162 |
| GRD | 170 | 165 |
| MTS | 157 | 156 |

Each scheduling run takes about 5 ms.

## References
- Kahn, A. B. (1962). Topological sorting of large networks. *CACM* 5(11).
- Kelley, J. E. & Walker, M. R. (1959). Critical-path planning and scheduling. *Proc. Eastern Joint Computer Conference*.
- Malcolm, D. G. et al. (1959). Application of a technique for research and development program evaluation. *Operations Research* 7(5).
- Kolisch, R. (1996). Serial and parallel resource-constrained project scheduling methods revisited. *EJOR* 90(2).
- Cormen, Leiserson, Rivest, Stein. *Introduction to Algorithms*, ch. 22.

---

**Author:** Dr. Kainat Rizwan
