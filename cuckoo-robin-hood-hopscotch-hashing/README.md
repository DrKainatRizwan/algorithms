# Open-Addressing Hash Tables: Cuckoo, Robin Hood and Hopscotch Hashing

## Overview

A from-scratch study of four open-addressing hash table designs, all behind one `MutableMapping` interface with automatic growth and exact probe accounting:

| Scheme | Idea | Lookup bound |
|--------|------|--------------|
| `LinearProbingTable` | Walk forward from the home slot; tombstone deletion | none (clusters) |
| `RobinHoodTable` | Poorer keys evict richer ones; backward-shift deletion | early exit on misses, low variance |
| `CuckooTable` | Two hash functions, buckets of 1-8 slots, random-walk eviction | at most `2 * bucket_size` slots |
| `HopscotchTable` | Keys stay within `H` slots of home, tracked by a bitmap | at most `H` flagged slots |

Four hash families (identity, multiply-shift, simple tabulation, SplitMix) are provided so that hash quality can be studied separately from the table layout. Only the standard library is used.

## How it works

* **Linear probing** probes `h(k), h(k)+1, ...`. Deletions mark tombstones; these count against the load limit and trigger an in-place rebuild when they accumulate.
* **Robin Hood** stores each key's displacement `d` from its home slot. On insertion the carried key swaps with any resident whose displacement is smaller, so displacements stay balanced. Searches stop as soon as they meet a slot whose displacement is below the current probe distance, which shortens unsuccessful lookups. Deletion shifts the following cluster back by one slot, so no tombstones are needed.
* **Cuckoo** places each key in one of two buckets. If both are full, a random resident is evicted to its alternative bucket, repeating up to `max_kicks` times. On failure the table is rehashed with fresh hash functions and grown if that keeps failing. One-slot buckets only sustain a load of about 50%; four-slot buckets reach roughly 90%+.
* **Hopscotch** finds the nearest free slot by linear probing; if it lies beyond the neighbourhood `H`, the algorithm repeatedly moves a resident that may legally occupy the free slot, pulling the hole towards the home bucket. Each home slot stores an `H`-bit bitmap of the slots in its neighbourhood that hold its keys, so lookups inspect only flagged slots. If no legal move exists the table grows.

## Complexity

| Operation | Linear | Robin Hood | Cuckoo | Hopscotch |
|-----------|--------|------------|--------|-----------|
| Lookup | O(1) expected, O(n) worst | O(1) expected, small variance | O(1) worst case | O(H) worst case |
| Insert | O(1) expected | O(1) expected | O(1) expected amortised | O(1) expected amortised |
| Delete | O(1) | O(1) expected | O(1) | O(1) |
| Space | O(n) | O(n) plus displacement array | O(n) | O(n) plus bitmap per slot |

All schemes use O(n) space and amortised O(1) growth by doubling.

## Usage

```python
from open_addressing import RobinHoodTable, CuckooTable, HopscotchTable, TabulationHash

t = RobinHoodTable(hash_factory=TabulationHash, seed=1)
t["alpha"] = 1
t[42] = "answer"
del t["alpha"]
print(len(t), t.load_factor, t.probe_count(42))

c = CuckooTable(bucket_size=4)          # worst-case 8 probes per lookup
h = HopscotchTable(neighborhood=32)
```

Run the tests and demo with:

```
python -m pytest -q
python examples/demo.py
```

## Results

Output of `examples/demo.py` (capacity 4096, random 48-bit keys, probes = slots inspected, `--` means the table had to grow or rehash to reach that load):

| Scheme | 30% | 50% | 70% | 85% | 95% |
|--------|-----|-----|-----|-----|-----|
| linear (hit mean / max, miss mean) | 1.21/7, 1.55 | 1.58/25, 2.77 | 2.23/53, 6.50 | 3.91/192, 21.99 | 10.19/1028, 223.74 |
| robin-hood | 1.21/4, 1.38 | 1.58/9, 1.82 | 2.23/16, 2.59 | 3.91/28, 4.39 | 10.19/41, 10.71 |
| cuckoo b=1 | 1.18/2, 2.00 | 1.33/2, 2.00 | -- | -- | -- |
| cuckoo b=4 | 1.61/7, 8.00 | 2.11/8, 8.00 | 2.68/8, 8.00 | 3.36/8, 8.00 | 4.26/8, 8.00 |
| hopscotch (H=32) | 1.16/4, 1.05 | 1.27/5, 1.12 | 1.36/6, 1.21 | 1.43/6, 1.30 | -- |

Observations: Robin Hood has the same mean successful-lookup cost as linear probing (it only reorders keys) but a dramatically shorter tail (max 41 vs 1028 probes at 95%) and about 20x cheaper misses. Cuckoo with one-slot buckets cannot go beyond about half load, while four-slot buckets stay at 8 probes for any miss up to 95%. Hopscotch's bitmaps make misses nearly free.

Hash quality on keys that are multiples of 1024 (Robin Hood table, n = 1500, capacity 4096):

| Hash | Mean hit | Max | Mean miss |
|------|----------|-----|-----------|
| identity | 188.00 | 375 | 376.00 |
| multiply-shift | 1.00 | 1 | 1.02 |
| tabulation | 1.22 | 4 | 1.47 |

Under steady-state churn (2000 live keys, 20000 delete and insert rounds, max load 0.8), linear probing with tombstones measured 1.22/1.64 mean hit/miss probes versus 1.43/1.70 for Robin Hood. At this moderate load the tombstone clean-up keeps linear probing competitive, so Robin Hood's advantage shows up mainly at high load.

Pure-Python wall-clock for 20000 inserts, hits and misses is about 0.06 s for linear, Robin Hood and hopscotch, and 0.17 s for four-slot cuckoo (the built-in `dict` is implemented in C and is not comparable).

## Project layout

```
open_addressing/
    hashing.py    hash families and stable key reduction
    base.py       mapping API, growth policy, probe statistics
    probing.py    linear probing and Robin Hood
    cuckoo.py     bucketised cuckoo hashing
    hopscotch.py  hopscotch hashing
    benchmark.py  workloads, probe profiles, load sweeps, churn
tests/            46 deterministic pytest cases
examples/demo.py  benchmark walkthrough (about 1 s)
```

## References

* R. Pagh and F. Rodler, "Cuckoo Hashing", *Journal of Algorithms* 51(2), 2004.
* P. Celis, P.-A. Larson, J. I. Munro, "Robin Hood Hashing", *FOCS* 1985.
* M. Herlihy, N. Shavit, M. Tzafrir, "Hopscotch Hashing", *DISC* 2008.
* M. Dietzfelbinger et al., "A Reliable Randomized Algorithm for the Closest-Pair Problem", *J. Algorithms* 25(1), 1997 (multiply-shift).
* M. Patrascu and M. Thorup, "The Power of Simple Tabulation Hashing", *J. ACM* 59(3), 2012.
* D. Fotakis, R. Pagh, P. Sanders, P. Spirakis, "Space Efficient Hash Tables with Worst Case Constant Access Time", *STACS* 2003.
* D. E. Knuth, *The Art of Computer Programming, Vol. 3*, Section 6.4.

---

**Author:** Dr. Kainat Rizwan
