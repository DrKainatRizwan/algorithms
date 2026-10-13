# Fenwick Trees, Sparse Tables and Sqrt Decomposition: Range Query Benchmark Suite

## Overview
A from-scratch library of classic range-query structures in pure Python, with a brute-force reference and a small benchmark harness. All ranges are half-open `[lo, hi)`.

- `Fenwick`: point update / prefix sum, O(n) build, `lower_bound` by binary lifting.
- `RangeFenwick`: range add and range sum using two Fenwick trees.
- `Fenwick2D`: point update and rectangle sum.
- `SparseTable` / `ArgMinSparseTable`: O(1) idempotent queries (min, max, gcd, argmin).
- `DisjointSparseTable`: O(1) queries for any associative (even non-commutative) operation.
- `SqrtDecomposition` / `SqrtRangeAdd`: block decomposition with point updates, lazy range add, sum/min/max.
- `mo_distinct_counts`: offline distinct-count queries with Mo's algorithm.

## How it works
- **Fenwick tree**: node `i` stores the sum of the `lowbit(i)` elements ending at `i`. Prefix queries walk down by clearing the lowest set bit; updates walk up by adding it. Range add/sum keeps a difference array `d` and uses `prefix(x) = x*sum(d[i]) - sum(i*d[i])`, one tree per term.
- **Sparse table**: `T[k][i]` = op over `a[i : i+2^k]`. A query overlaps two blocks of size `2^floor(log2(len))`, correct only for idempotent ops.
- **Disjoint sparse table**: at level `k`, each block of size `2^(k+1)` stores aggregates running outward from its midpoint. A query is answered at the level of the highest differing bit of `lo` and `hi-1`, combining one value from each side.
- **Sqrt decomposition**: blocks of size ~sqrt(n) store sum/min/max; a query touches two partial blocks and whole blocks in between. Range add keeps a lazy offset per block.
- **Mo's algorithm**: queries are sorted by block of `lo` (alternating the order of `hi`) and a window is slid, maintaining frequencies.

## Complexity
| Structure | Build | Update | Query | Space |
|---|---|---|---|---|
| Fenwick | O(n) | O(log n) | O(log n) | O(n) |
| RangeFenwick | O(n) | O(log n) | O(log n) | O(n) |
| Fenwick2D | O(RC) | O(log R log C) | O(log R log C) | O(RC) |
| Sparse table | O(n log n) | static | O(1) | O(n log n) |
| Disjoint sparse table | O(n log n) | static | O(1) | O(n log n) |
| Sqrt decomposition | O(n) | O(sqrt n) | O(sqrt n) | O(n) |
| Mo's algorithm | - | offline | O((n+q) sqrt n) total | O(n) |

## Usage
```python
from range_queries import Fenwick, SparseTable, DisjointSparseTable, mo_distinct_counts

f = Fenwick([3, 1, 4, 1, 5])
f.add(2, 10)
print(f.sum(1, 4))                  # 1 + 14 + 1 = 16

st = SparseTable([5, 2, 7, 2, 9])
print(st.query(1, 5))               # 2

ds = DisjointSparseTable(list("abcde"), lambda x, y: x + y)
print(ds.query(1, 4))               # 'bcd'

print(mo_distinct_counts([1, 2, 1, 3], [(0, 3), (1, 4)]))   # [2, 3]
```

Run `python examples/demo.py` for the demo and `python -m pytest -q` for the tests.

## Results
Output of the demo (seconds, single run, CPython):

| Workload | Structure | Time |
|---|---|---|
| 4000 mixed updates/sums, n=20000 | naive | 0.1827 |
| | sqrt decomposition | 0.0256 |
| | Fenwick | 0.0134 |
| 5000 range-min queries, n=100000 | naive | 2.3283 |
| | sqrt decomposition | 0.0434 |
| | sparse table (query) | 0.0026 |
| | sparse table (build) | 0.2808 |

The sparse table pays an O(n log n) build, which amortises after a few thousand queries on static data.

## References
- P. Fenwick, "A new data structure for cumulative frequency tables", Software: Practice and Experience, 1994.
- M. Bender, M. Farach-Colton, "The LCA Problem Revisited", LATIN 2000.
- Codeforces blog (mnbvmar), "Disjoint Sparse Table".
- Cormen, Leiserson, Rivest, Stein, *Introduction to Algorithms*.
- Mo's algorithm: folklore, see competitive-programming references (e.g. Competitive Programmer's Handbook, A. Laaksonen).

---

**Author:** Dr. Kainat Rizwan
