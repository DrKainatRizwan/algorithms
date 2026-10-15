# A*, Jump Point Search and D* Lite Pathfinding on Large Grids

## Overview

From-scratch implementations of three grid planners and a benchmark harness:

- **A\*** (plus weighted A\* and Dijkstra) with octile, Euclidean and Manhattan heuristics.
- **Jump Point Search (JPS)**, which prunes symmetric paths on uniform-cost grids.
- **D\* Lite**, an incremental replanner for agents that discover obstacles while moving.

Movement is 8-connected (orthogonal cost 1, diagonal cost sqrt(2)) and diagonal moves may not cut corners. Map generators (random obstacles, cellular-automaton caves, rooms and corridors, mazes) are seeded and deterministic.

## How it works

**A\*.** Best-first search ordered by `f = g + w*h`. With an admissible, consistent heuristic (octile distance is exact on an empty grid) and `w = 1`, each node is expanded at most once and the path is optimal; weighted A\* returns a path of cost at most `w` times optimal with far less effort.

**JPS.** Instead of pushing every neighbour, JPS *jumps* along a direction until it reaches the goal or a cell with a *forced neighbour* (a free cell that is only reachable optimally through the current one because of a nearby obstacle). Diagonal jumps additionally scan their two orthogonal components. Only jump points enter the open list, and the path is interpolated afterwards. It is still optimal, but expands far fewer nodes in open or room-like terrain.

**D\* Lite.** Searches backwards from the goal and keeps `g` and one-step look-ahead `rhs` values. When the agent moves, the key offset `km` is increased by the heuristic distance travelled, so existing queue entries stay valid. When cells change, only vertices within one cell of an edit are re-evaluated, and `ComputeShortestPath` repairs the inconsistent region.

## Complexity

| Algorithm | Time | Space |
|-----------|------|-------|
| Dijkstra / A\* | O(V log V) worst case (V = cells) | O(V) |
| JPS | O(V log V) worst case, but typically far fewer heap operations; scans cost O(path length) per jump | O(V) |
| D\* Lite | first search as A\*; each repair is proportional to the affected region | O(V) |

## Usage

```python
import numpy as np
from grid_pathfinding import astar, jps, room_map, random_query, navigate_dstar, random_obstacles

grid = room_map(160, 160, seed=1)
start, goal = random_query(grid, np.random.default_rng(0))
a, j = astar(grid, start, goal), jps(grid, start, goal)
assert abs(a.cost - j.cost) < 1e-9          # both optimal
print(a.expanded, j.expanded)

world = random_obstacles(60, 60, 0.25, seed=3)   # hidden from the agent
s, t = random_query(world, np.random.default_rng(3), 0.8)
run = navigate_dstar(world, s, t, radius=4)
print(run.success, run.travelled, run.replans)
```

Run the demo with `python examples/demo.py` and the tests with `python -m pytest -q`.

## Results

Measured by `examples/demo.py` (160x160 maps, 8 queries per map, mean per query; Python 3.12, single thread):

| Map | Planner | Time (ms) | Expanded | Cost |
|-----|---------|-----------|----------|------|
| random 20% | Dijkstra | 90.33 | 17653.8 | 147.19 |
| random 20% | A\* | 15.91 | 2451.6 | 147.19 |
| random 20% | weighted A\* (w=2) | 1.26 | 194.1 | 163.70 |
| random 20% | JPS | 10.71 | 1037.1 | 147.19 |
| caves | A\* | 9.28 | 1253.5 | 133.73 |
| caves | JPS | 3.46 | 117.9 | 133.73 |
| rooms | A\* | 7.48 | 1271.0 | 153.98 |
| rooms | JPS | 2.31 | 43.9 | 153.98 |
| maze | A\* | 40.12 | 9541.5 | 444.38 |
| maze | JPS | 20.15 | 2925.0 | 444.38 |

JPS matches A\* in cost everywhere and expands about 30x fewer nodes on rooms and 10x fewer on caves. On unstructured random obstacles the advantage shrinks, because forced neighbours are everywhere.

Navigation in unknown terrain (60x60, 25% hidden obstacles, sensing radius 4, 10 episodes, averages):

| Planner | Expansions | Replans | Distance travelled |
|---------|-----------|---------|--------------------|
| D\* Lite | 1859.7 | 60.3 | 77.65 |
| A\* replanning from scratch (only when the current plan is invalidated) | 1274.3 | 24.1 | 77.86 |

In this setting D\* Lite did not save work in expansion count: the scratch baseline replans only when the remaining path is actually blocked, while D\* Lite repairs after every newly sensed cell, and on small maps the goal-directed A\* search is already cheap. D\* Lite's benefit appears when replans are frequent relative to map size, and it never needs the full-map re-search.

## Tests

19 tests cover corner-cutting rules, A\* vs Dijkstra equivalence, weighted-A\* bounds, JPS optimality against A\* on 160 random maps, D\* Lite agreement with fresh searches after random edits and agent movement, unreachable goals, and full navigation episodes.

## References

- Hart, Nilsson, Raphael. *A Formal Basis for the Heuristic Determination of Minimum Cost Paths.* IEEE Trans. SSC, 1968.
- Harabor, Grastien. *Online Graph Pruning for Pathfinding on Grid Maps.* AAAI 2011.
- Koenig, Likhachev. *D\* Lite.* AAAI 2002.
- Pohl. *Heuristic Search Viewed as Path Finding in a Graph.* Artificial Intelligence, 1970.

---

**Author:** Dr. Kainat Rizwan
