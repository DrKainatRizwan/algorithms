# Concurrent Skip List with Probabilistic Balancing Analysis

## Overview
A from-scratch skip list toolkit in pure Python (standard library only):

- `SkipList` – sequential ordered map with span-augmented pointers, giving `rank`, `kth`, `floor`, `ceiling` and range scans.
- `LockFreeSkipList` – a lock-free-style concurrent set/map (Herlihy–Shavit / Fraser) using markable references, logical-then-physical deletion and a wait-free `contains`.
- Level policies – seeded geometric levels, and hash-derived levels that make the structure history independent.
- Analysis helpers – expected height, Pugh's search-cost bound, tail bounds, chi-square test of tower heights and an optimal-`p` search.

CPython has no hardware CAS on references, so `AtomicMarkableRef` emulates it with a per-reference lock; the algorithm never holds more than one such lock at a time.

## How it works
Each key owns a tower of height `h` with `P(h >= k) = p^(k-1)`. Searching starts at the top level and moves right while the next key is smaller, then drops down. Expected path length is `log_{1/p}(n)/p + 1/(1-p)`, which is minimised near `p = 1/e`.

Each forward pointer also stores a *span* (how many bottom-level nodes it skips); summing spans along a search path yields order statistics in the same expected time.

The concurrent list deletes in two phases: all next-pointers of the victim are marked (logical removal; the level-0 mark is the linearisation point and only one remover wins), then any later traversal CAS-unlinks marked nodes. Insertion links level 0 first (linearisation point) and then builds the tower upward.

## Complexity
| Operation | Expected time | Space |
|-----------|---------------|-------|
| search / insert / delete | O(log n) | O(n) total, 1/(1-p) pointers per node |
| rank / kth | O(log n) | – |
| range scan of m items | O(log n + m) | – |
| concurrent add/remove | O(log n) plus retries | – |

Height exceeds `k` with probability at most `n·p^k`.

## Usage
```python
from skiplist_lab import SkipList, LockFreeSkipList, GeometricLevels

sl = SkipList(GeometricLevels(p=0.5, seed=1))
for k in range(0, 100, 3):
    sl.insert(k, str(k))
print(sl.rank(51), sl.kth(10), list(sl.range(30, 50)))

lst = LockFreeSkipList()
lst.add(5, "x"); lst.remove(5); print(lst.contains(5))
```

Run `python examples/demo.py` for the full demo and `python -m pytest -q` for the tests (23 tests).

## Results
From `examples/demo.py` (seeded; mean key comparisons per successful lookup, p = 0.5):

| n | height (theory) | comparisons | Pugh bound |
|---|-----------------|-------------|-----------|
| 1,000 | 14 (12.0) | 19.42 | 21.93 |
| 10,000 | 15 (15.3) | 27.21 | 28.58 |
| 100,000 | 18 (18.6) | 31.62 | 35.22 |

Effect of `p` at n = 20,000: 0.125 → 31.26, 0.25 → 25.81, 0.37 → 26.32, 0.5 → 27.32, 0.75 → 42.14 comparisons; the analytic optimum is about 0.36.

Four threads running 3,000 mixed operations each on 100 overlapping keys reached ~64k ops/s with 6 CAS failures, and the final contents matched the per-key net add/remove log exactly. Pure Python is slower than `bisect` on a plain list for 20k inserts (0.21 s vs 0.03 s); the point here is the algorithm, not raw speed, and the GIL prevents parallel speed-up.

## References
- W. Pugh, "Skip Lists: A Probabilistic Alternative to Balanced Trees", CACM 33(6), 1990.
- W. Pugh, "A Skip List Cookbook", UMIACS-TR-90-72, 1990.
- M. Herlihy, N. Shavit, *The Art of Multiprocessor Programming*, ch. 14.
- K. Fraser, "Practical Lock-Freedom", PhD thesis, Cambridge, 2004.
- T. Harris, "A Pragmatic Implementation of Non-Blocking Linked Lists", DISC 2001.

---

**Author:** Dr. Kainat Rizwan
