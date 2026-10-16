# Interval Trees and Segment Trees for Computational Geometry Queries

## Overview

A from-scratch Python library of interval and segment tree structures and the
plane-sweep algorithms built on them:

- `CenteredIntervalTree` — static centred interval tree for stabbing and overlap queries.
- `DynamicIntervalTree` — augmented treap (subtree max of right endpoints) with insert, delete and overlap queries.
- `SortedEndpointIndex` — O(log n) *counting* of stabbing/overlapping intervals.
- `CoverageTree` — segment tree maintaining covered length under add/remove of ranges.
- `MaxAddTree` — lazy range-add / range-max-with-argmax segment tree.
- Sweeps: union area of rectangles (Klee's measure), union length, deepest overlap point, rectangle intersection reporting, window queries and the skyline problem.
- `naive` — brute-force oracles used by the test-suite and benchmarks.

## How it works

**Centred interval tree.** Choose the median endpoint as the node centre. Intervals
containing it stay at the node, stored twice (sorted by left and by right endpoint);
intervals entirely left or right recurse. A stab at `x < center` scans the left-sorted
list until `lo > x`, then descends left; symmetric on the right. Every scanned
interval is a hit, hence O(log n + k).

**Augmented treap.** Nodes are ordered by `(lo, id)`, balanced by random priorities
and carry `max_hi` for their subtree. Overlap search prunes any subtree whose
`max_hi < query_lo` and any right subtree when `node.lo > query_hi`.
`any_overlap` follows a single path (CLRS 14.3).

**Coverage tree.** Each node holds a cover count and the covered length of its subtree.
Counts are never pushed down, so removing a range exactly undoes its insertion. The
root value is the union length, available in O(1).

**Union area / intersections.** Sweep over x. For the area, a `CoverageTree` over the
compressed y-coordinates gives the covered height between events. For intersection
reporting, active rectangles' y-intervals live in a `DynamicIntervalTree`; each
insertion queries it, so total cost is output-sensitive.

**Max overlap, skyline.** Compress coordinates, then use lazy range-add/max
(`MaxAddTree`) or max-tags resolved top-down (skyline).

## Complexity

| Structure / algorithm | Build | Query / update | Space |
|---|---|---|---|
| Centred interval tree | O(n log n) | O(log n + k) | O(n) |
| Dynamic interval treap | — | O(log n) expected insert/delete, O(min(n, k log n)) overlap | O(n) |
| Sorted-endpoint counting | O(n log n) | O(log n) | O(n) |
| Coverage tree | O(n) | O(log n) update, O(1) union length | O(n) |
| Lazy max-add tree | O(n) | O(log n) | O(n) |
| Union area / max overlap / skyline | — | O(n log n) | O(n) |
| Rectangle intersections | — | O((n + k) log n) | O(n) |

## Usage

```python
from geometry_trees import CenteredIntervalTree, union_area, rect_intersections

tree = CenteredIntervalTree([(1, 5), (4, 9), (10, 12)])
tree.stab(4.5)          # [0, 1]
tree.overlap(8, 11)     # [1, 2]

union_area([(0, 0, 2, 2), (1, 1, 3, 3)])            # 7
rect_intersections([(0, 0, 1, 1), (1, 1, 2, 2)])    # [(0, 1)]
```

Run the tests with `python -m pytest -q` and the demo with `python examples/demo.py`.

## Results

Output of `examples/demo.py` (seeded; single run, CPython 3.12):

| Task | Result |
|---|---|
| 2000 stabbing queries over 20 000 intervals, centred tree | 70 ms |
| same, dynamic treap (depth 34) | 161 ms |
| same, brute force (extrapolated) | ~2180 ms |
| 3000 random rectangles: sweep intersection reporting (3920 pairs) | 41 ms |
| same, brute force O(n²) | 285 ms |
| 300 window queries over 3000 rectangles | 15 ms |

All structures agree with the brute-force oracles in the test-suite (20 tests,
randomised with fixed seeds, including treap invariant checks after random insert/delete mixes).

## References

- M. de Berg, O. Cheong, M. van Kreveld, M. Overmars, *Computational Geometry: Algorithms and Applications*, 3rd ed., Springer, 2008 (ch. 10).
- T. H. Cormen, C. E. Leiserson, R. L. Rivest, C. Stein, *Introduction to Algorithms*, 3rd ed., MIT Press, 2009 (§14.3, interval trees).
- H. Edelsbrunner, "Dynamic data structures for orthogonal intersection queries", Technical Report F59, TU Graz, 1980.
- J. L. Bentley, "Algorithms for Klee's rectangle problems", Carnegie Mellon University, 1977.
- C. R. Aragon, R. Seidel, "Randomized search trees", FOCS 1989.

---

**Author:** Dr. Kainat Rizwan
