# B+ Tree Storage Engine with Range Scans and Bulk Loading

## Overview

A from-scratch B+ tree in pure Python, built the way a database index is: all
data lives in chained leaf pages, internal pages hold only routing keys, and
every operation is expressed in terms of page visits. On top of the tree sit a
bottom-up bulk loader, a fixed-size page serialiser, an LRU buffer-pool
simulator driven by real access traces, and a small key-value store with
atomic batches, a write-ahead log and prefix scans.

Modules (`bplus_engine/`):

| Module | Purpose |
|---|---|
| `nodes.py` | Leaf and internal node types, leaf chaining |
| `tree.py` | Insert, delete (borrow/merge), point lookup, range scans, floor/ceiling, invariant checker |
| `bulk_load.py` | O(n) bottom-up construction with a fill factor |
| `serializer.py` | 4 KiB-style page images: `dumps` / `loads` / `save` / `load` |
| `buffer_pool.py` | LRU page cache replayed over recorded node-access traces |
| `engine.py` | `KVStore`: WAL, atomic batches, snapshots, order-preserving string keys |

## How it works

* **Capacity.** With page capacity `m` (`max_keys`) each node holds at most `m`
  keys; every non-root node holds at least `m // 2`.
* **Routing.** In an internal node `children[i]` covers keys `< keys[i]` and
  `children[i+1]` covers keys `>= keys[i]`; descent uses `bisect_right`.
* **Insert.** Descend to the leaf, insert, and on overflow split in half. A leaf
  split *copies* the first key of the right half up; an internal split *moves*
  the middle key up. Splits propagate to the root, which is the only way the
  height grows.
* **Delete.** Remove from the leaf; if a node drops below `m // 2` keys, borrow
  from a sibling that has spare keys (rotating through the parent separator) or
  merge with a sibling. A root with one child is dropped, shrinking the height.
* **Range scan.** One root-to-leaf descent to the lower bound, then a walk along
  the `next` pointers: `O(log n + k)`. The `prev` pointers give descending scans.
* **Bulk loading.** Sorted input is cut into near-equal leaf chunks of
  `fill * m` entries (respecting the minimum occupancy), then each level of
  parents is built from the level below using the minimum key of each subtree as
  separator.
* **Serialisation.** Every node becomes one page; pointers become page numbers,
  with a header page recording root, size and page count.
  `max_keys_for_page` picks the capacity that fits a page.
* **Buffer pool.** `tree.trace` records node ids touched by an operation; the pool
  replays them to give hit rate and simulated disk reads.

## Complexity

| Operation | Time (page visits) | Space |
|---|---|---|
| search / insert / delete | `O(log_m n)` | `O(n)` total |
| range scan of `k` keys | `O(log_m n + k/m)` pages | `O(1)` extra |
| bulk load | `O(n)` | `O(n)` |
| serialise / load | `O(n)` | `O(n)` |

Height is at most `log_{m/2+1}(n) + 1`.

## Usage

```python
from bplus_engine import BPlusTree, bulk_load, KVStore, dumps, loads

t = BPlusTree(max_keys=32)
for k in range(1000):
    t[k] = k * k
print(t[10], list(t.range(5, 8)), t.floor(2000))

fast = bulk_load(((k, b'v') for k in range(10_000)), max_keys=64, fill=0.9)
blob = dumps(fast, page_size=4096)   # paging needs int keys and bytes values
assert list(loads(blob).items()) == list(fast.items())

store = KVStore()
store.put(1, b"one")
store.batch([("put", 2, b"two"), ("del", 1, None)])
```

Run `python examples/demo.py` for the full tour and `python -m pytest -q` for the tests.

## Results

Output of `examples/demo.py` (seeded; 100,000 random keys, `max_keys=64`;
timings from a single CI run):

| Build method | Time | Height | Leaves | Leaf fill |
|---|---|---|---|---|
| 100k random inserts | 0.145 s | 3 | 2241 | 0.70 |
| Bulk load (fill 0.9) | 0.065 s | 3 | 1725 | 0.91 |

A 500-key range scan returns in about 0.1 ms. LRU hit rate for 5,000 random point
lookups on the 1,756-page tree:

| Frames | Share of pages | Hit rate | Disk reads |
|---|---|---|---|
| 4 | 0.2% | 0.345 | 9830 |
| 64 | 3.6% | 0.589 | 6167 |
| 256 | 14.6% | 0.708 | 4375 |
| 1024 | 58.3% | 0.834 | 2484 |

Random inserts reach the textbook ~70% fill, bulk loading gives ~91% and so
fewer pages. After deleting 60% of the keys the leaf fill falls to 0.61 while all
invariants hold. 20,000 entries with 8-byte values fit in 112 pages of 4 KiB
(capacity 203 keys per page).

The 57 tests cover randomised comparison against a dict for several orders,
invariants after every phase, range/floor/ceiling edge cases, bulk loads of many
sizes and fill factors, serialisation round trips and overflow errors, buffer
pool behaviour, WAL recovery and batch atomicity.

## References

* R. Bayer, E. McCreight, "Organization and Maintenance of Large Ordered Indexes", *Acta Informatica* 1, 1972.
* D. Comer, "The Ubiquitous B-Tree", *ACM Computing Surveys* 11(2), 1979.
* G. Graefe, "Modern B-Tree Techniques", *Foundations and Trends in Databases* 3(4), 2011.
* A. Silberschatz, H. Korth, S. Sudarshan, *Database System Concepts*, ch. on indexing.
* T. Cormen et al., *Introduction to Algorithms*, ch. 18 (B-Trees).

---

**Author:** Dr. Kainat Rizwan
