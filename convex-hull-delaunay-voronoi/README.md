# Convex Hull, Delaunay Triangulation and Voronoi Diagram Algorithms

## Overview

A from-scratch planar computational geometry toolkit (Python 3.12, standard library only):

- `geometry_kit/primitives.py` – orientation and in-circle predicates, circumcentres, polygon area.
- `geometry_kit/hull.py` – four convex hull algorithms (Andrew monotone chain, Graham scan, Jarvis march, Quickhull), plus area, perimeter, rotating-calipers diameter and O(log h) point-in-convex-polygon.
- `geometry_kit/delaunay.py` – incremental Bowyer-Watson Delaunay triangulation with a walking point locator, hull extraction and an O(n^4) brute-force reference.
- `geometry_kit/voronoi.py` – bounded Voronoi cells by half-plane clipping against Delaunay neighbours, Voronoi vertices and Lloyd relaxation.
- `geometry_kit/applications.py` – Euclidean MST, closest pair, largest empty circle and greedy nearest-neighbour search on the Delaunay graph.

## How it works

**Hulls.** Monotone chain sorts points lexicographically and builds lower and upper chains with a stack, popping on non-left turns. Graham scan sorts by angle around the lowest point. Jarvis march wraps the set one hull vertex at a time. Quickhull splits on the point farthest from a chord and recurses on the points outside each new edge.

**Delaunay.** Points are inserted in random order into a large enclosing super-triangle. For each point the containing triangle is found by a visibility walk from the previous insertion; triangles whose circumcircle contains the point are found by a flood fill over the adjacency map (a directed-edge dictionary); the resulting star-shaped cavity is re-triangulated by connecting its boundary to the new point. Triangles touching super-triangle vertices are discarded at the end. The in-circle determinant is evaluated relative to the query point to limit rounding error.

**Voronoi.** The Voronoi cell of a site is the intersection of the half-planes closer to it than to each Delaunay neighbour. Starting from a bounding box, each cell is clipped by those bisectors (Sutherland-Hodgman), which is exact because non-neighbours never contribute edges.

**Applications.** The Euclidean MST, the closest pair and the nearest neighbour all live in the Delaunay graph: the MST is a subgraph (Kruskal over O(n) edges), the closest pair is a Delaunay edge, and greedy routing on the graph always reaches the true nearest site.

## Complexity

| Algorithm | Time | Space |
|-----------|------|-------|
| Monotone chain / Graham | O(n log n) | O(n) |
| Jarvis march | O(n h) | O(h) |
| Quickhull | O(n log n) expected, O(n^2) worst | O(n) |
| Rotating calipers diameter | O(h) | O(1) |
| Bowyer-Watson (random order) | O(n log n) expected | O(n) |
| Voronoi cells | O(n) clips of average constant degree | O(n) |
| Euclidean MST / closest pair | O(n log n) | O(n) |

## Usage

```python
from geometry_kit import convex_hull, Delaunay, Voronoi, euclidean_mst

pts = [(0, 0), (4, 0), (4, 3), (0, 3), (2, 1), (1, 2)]
print(convex_hull(pts, "quickhull"))
d = Delaunay(pts)
print(d.triangles, d.is_delaunay())
v = Voronoi(pts)
print(v.cell(4))
print(euclidean_mst(pts))
```

Run `python -m pytest -q` for the tests (51 tests) and `python examples/demo.py` for the demo.

## Results

Measured by `examples/demo.py` (uniform points; times in ms, single run):

| n | distribution | hull size | monotone | graham | jarvis | quickhull |
|---|--------------|-----------|----------|--------|--------|-----------|
| 2000 | square | 18 | 2.5 | 5.7 | 7.3 | 2.5 |
| 2000 | disk | 37 | 2.3 | 5.6 | 15.3 | 2.8 |
| 20000 | square | 21 | 28.1 | 73.4 | n/a | 27.9 |

Delaunay triangulation (triangle counts match the Euler formula 2n - 2 - h):

| n | triangles | time (ms) |
|---|-----------|-----------|
| 500 | 980 | 22.9 |
| 2000 | 3972 | 143.7 |
| 5000 | 9969 | 516.3 |

Lloyd relaxation of 100 random sites in a square reduced the coefficient of variation of Voronoi cell area from 0.630 to 0.327 in five steps, with cell areas always summing to the box area. Greedy nearest-neighbour search on the Delaunay graph was exact on 200/200 queries.

## References

- M. de Berg, O. Cheong, M. van Kreveld, M. Overmars, *Computational Geometry: Algorithms and Applications*, 3rd ed., Springer, 2008.
- A. M. Andrew, "Another efficient algorithm for convex hulls in two dimensions", *Information Processing Letters* 9(5), 1979.
- R. L. Graham, "An efficient algorithm for determining the convex hull of a finite planar set", *Information Processing Letters* 1(4), 1972.
- C. B. Barber, D. P. Dobkin, H. Huhdanpaa, "The Quickhull algorithm for convex hulls", *ACM TOMS* 22(4), 1996.
- A. Bowyer, "Computing Dirichlet tessellations", *The Computer Journal* 24(2), 1981; D. F. Watson, same issue.
- S. Lloyd, "Least squares quantization in PCM", *IEEE Trans. Information Theory* 28(2), 1982.
- L. J. Guibas, D. E. Knuth, M. Sharir, "Randomized incremental construction of Delaunay and Voronoi diagrams", *Algorithmica* 7, 1992.

---

**Author:** Dr. Kainat Rizwan
