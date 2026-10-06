# Persistent Segment Tree Library with Time-Travel Range Queries

## Overview
A from-scratch library of persistent (versioned) segment trees. Every update returns a new version while all earlier versions stay queryable, at a cost of O(log n) new nodes per update. Included:

- `PersistentSegmentTree` – point update / range query over any monoid (sum, min, max, gcd, or your own), with branching history (you can update any old version).
- `PersistentLazyTree` – range add / range sum with permanent (non-propagating) tags, so no node is ever mutated.
- `RangeKth` – k-th smallest and rank queries in any subarray using prefix versions of a count tree.
- `TimeTravelArray` – timestamped writes, queries "as of" any time, rollback and per-cell history.
- `SnapshotArray` – a naive full-copy baseline for validation and benchmarking.

## How it works
Path copying: an update rebuilds only the nodes on the root-to-leaf path and reuses every other subtree by reference. A version is just a root pointer. For range add, tags are stored on nodes and never pushed down; a query adds `tag * overlap` for each ancestor visited. For k-th queries, version `i` indexes the first `i` elements by compressed value, so the counts of `a[l:r]` are the difference between versions `r` and `l`, and a single descent finds the k-th value.

## Complexity
| Operation | Time | Extra space |
|---|---|---|
| Build | O(n) | O(n) |
| Point update | O(log n) | O(log n) nodes |
| Range query | O(log n) | O(1) |
| Range add (lazy) | O(log n) | O(log n) nodes |
| k-th in `a[l:r]` | O(log n) | O(n log n) build |
| Time lookup | O(log v) bisect | O(1) |

## Usage
```python
from persistent_segtree import PersistentSegmentTree, TimeTravelArray, RangeKth, MAX

t = PersistentSegmentTree([3, 1, 4, 1, 5], MAX)
v1 = t.update(0, 1, 9)
print(t.query(0, 0, 5), t.query(v1, 0, 5))   # 5 9

tt = TimeTravelArray([0] * 4)
tt.set(10, 2, 7)
print(tt.query(5, 0, 4), tt.query(10, 0, 4))  # 0 7

print(RangeKth([5, 2, 8, 1, 9]).kth(0, 5, 3))  # 5
```

## Results
From `python examples/demo.py` (n = 20,000, 2,000 updates, range-max):

| Metric | Persistent tree | Full snapshots |
|---|---|---|
| Update time | 0.089 s | 0.263 s |
| Memory for history | 30,745 extra nodes (~15.4 per update) | 40,000,000 cells |

All 300 randomly chosen historical range-max queries matched the snapshot baseline, and the range median of 14,000 elements matched a sorted-list check.

Run the tests with `python -m pytest -q` (18 tests; randomised comparisons against the naive baseline use fixed seeds).

## References
- Driscoll, Sarnak, Sleator, Tarjan, "Making Data Structures Persistent", JCSS 38(1), 1989.
- Okasaki, *Purely Functional Data Structures*, Cambridge University Press, 1998.
- de Berg et al., *Computational Geometry: Algorithms and Applications*, 3rd ed., Springer, 2008.

---

**Author:** Dr. Kainat Rizwan
