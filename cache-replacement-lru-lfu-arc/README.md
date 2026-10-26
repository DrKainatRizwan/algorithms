# LRU, LFU and ARC Cache Algorithms with Trace-Driven Hit-Rate Benchmarks

## Overview

A from-scratch library of cache replacement policies with a trace-driven simulator. It implements FIFO, Random, CLOCK, LRU, an O(1) LFU (with optional count aging), simplified 2Q and the Adaptive Replacement Cache (ARC), plus Belady's clairvoyant OPT as an offline upper bound. Seeded workload generators (Zipf, loops, scan pollution, shifting working sets, mixed) make it easy to see where each policy wins and where it collapses. Only the standard library and NumPy are used.

## How it works

* **LRU** – hash map plus recency list (`OrderedDict`); every hit moves the key to the MRU end, eviction takes the LRU end.
* **LFU** – Shah/Mitra/Matani O(1) design: keys sit in per-frequency buckets and a `min_freq` pointer finds the victim directly. Ties are broken by recency. Optional aging halves all counts periodically so stale favourites can be displaced.
* **CLOCK** – ring buffer with a reference bit; the hand gives referenced pages a second chance.
* **2Q** – new keys enter a FIFO probation queue; only keys re-referenced after falling into the ghost queue are promoted to the main LRU, which filters one-shot scans.
* **ARC** – four lists: `T1` (seen once), `T2` (seen at least twice) and ghost lists `B1`/`B2` of recent evictions. A hit in `B1` means recency was under-served and grows the target size `p` of `T1`; a hit in `B2` shrinks it. The invariants `|T1|+|T2| <= c`, `|T1|+|B1| <= c` and total `<= 2c` are checked in the tests.
* **OPT** – evicts the resident key whose next use is farthest in the future (next-use indices precomputed in one backward pass, victims found with a lazy max-heap).

## Complexity

| Policy | Time per request | Space |
|--------|------------------|-------|
| FIFO, LRU, CLOCK (amortised), Random | O(1) | O(c) |
| LFU | O(1) (aging pass is O(c) every `period` requests) | O(c) |
| 2Q | O(1) | O(c) |
| ARC | O(1) | O(2c) |
| OPT (offline) | O(log c) | O(n + c) |

## Usage

```python
from cache_policies import ARCCache, compare, format_table, zipf_trace, cached

trace = zipf_trace(30000, universe=2000, alpha=1.0, seed=1)
cache = ARCCache(64)
print(cache.run(trace).hit_rate)

caps = [16, 32, 64]
print(format_table(caps, compare(trace, caps, ["LRU", "LFU", "ARC"])))

@cached("ARC", capacity=256)       # memoiser governed by a chosen policy
def slow(x): ...
```

Run `python examples/demo.py` for the full comparison and `python -m pytest -q` for the tests.

## Results

Hit rates (%) from `examples/demo.py`, 30,000 requests per workload (capacities 16 / 32 / 64 / 128):

| Workload | LRU | LFU | 2Q | ARC | OPT |
|----------|-----|-----|----|-----|-----|
| Zipf (alpha=1, 2000 keys) | 24.3 / 34.0 / 43.8 / 54.2 | 39.9 / 47.2 / 53.7 / 62.4 | 35.5 / 44.0 / 52.6 / 60.5 | 36.6 / 45.3 / 53.7 / 61.6 | 45.9 / 54.9 / 63.8 / 72.2 |
| Loop (length 100) | 0 / 0 / 0 / 99.7 | 0 / 0 / 0 / 99.7 | 0 / 0 / 0 / 99.7 | 0 / 0 / 0 / 99.7 | 15.1 / 31.2 / 63.4 / 99.7 |
| Scan pollution | 15.8 / 30.8 / 44.3 / 44.3 | 16.0 / 32.5 / 50.5 / 50.5 | 16.1 / 32.5 / 44.3 / 44.3 | 16.0 / 32.5 / 50.5 / 50.5 | 32.1 / 43.8 / 50.5 / 50.5 |
| Shifting working set | 48.9 / 73.2 / 99.0 / 99.0 | 15.2 / 19.1 / 27.3 / 49.5 | 52.5 / 72.9 / 99.0 / 99.0 | 51.2 / 73.0 / 99.0 / 98.9 | 72.2 / 88.3 / 99.0 / 99.0 |
| Mixed Zipf+loop+scan | 10.2 / 15.8 / 22.5 / 29.6 | 22.1 / 28.2 / 33.3 / 42.0 | 20.4 / 26.6 / 33.0 / 39.8 | 21.9 / 28.0 / 34.5 / 41.7 | 29.0 / 35.8 / 44.4 / 59.4 |

Observations: LFU wins on stationary skew but fails badly when the hot set moves (15-27% vs 99% for the recency-based policies); LRU is the opposite and thrashes on loops larger than the cache. ARC is never far from the best of the two on any workload without tuning, which is its design goal.

## References

* N. Megiddo and D. Modha, "ARC: A Self-Tuning, Low Overhead Replacement Cache", USENIX FAST, 2003.
* L. Belady, "A study of replacement algorithms for a virtual-storage computer", IBM Systems Journal, 1966.
* D. Shah, K. Mitra and D. Matani, "An O(1) algorithm for implementing the LFU cache eviction scheme", 2010.
* T. Johnson and D. Shasha, "2Q: A Low Overhead High Performance Buffer Management Replacement Algorithm", VLDB, 1994.
* F. Corbato, "A paging experiment with the Multics system", 1968 (CLOCK).

---

**Author:** Dr. Kainat Rizwan
