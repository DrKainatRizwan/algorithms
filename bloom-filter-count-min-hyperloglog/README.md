# Bloom Filter, Count-Min Sketch and HyperLogLog Streaming Analytics Library

## Overview
A from-scratch library of probabilistic data structures for stream processing, using only the standard library and NumPy. Three questions are answered in sublinear memory: *is x in the set?* (Bloom filters), *how often did x occur?* (Count-Min sketch, heavy hitters) and *how many distinct items were seen?* (HyperLogLog). A seeded 64-bit hash (FNV-1a plus the splitmix64 finaliser) makes every result reproducible regardless of `PYTHONHASHSEED`.

Modules in `stream_sketches/`: `hashing.py`, `bloom.py` (standard, counting, scalable), `countmin.py` (plain, conservative update, heavy hitters, inner product), `hyperloglog.py`, `streams.py` (Zipf generators), `analytics.py` (error evaluation).

## How it works
- **Bloom filter**: m bits and k hash functions, with m = -n ln p / (ln 2)^2 and k = (m/n) ln 2. Indices come from double hashing g_i = h1 + i·h2 (Kirsch–Mitzenmacher). No false negatives; false-positive rate (1 - e^{-kn/m})^k. Cardinality is recovered from the fill ratio. The counting variant uses saturating 8-bit counters to support deletion; the scalable variant chains slices with geometrically tighter rates so the total stays below the target.
- **Count-Min sketch**: a d×w counter table with w = ⌈e/ε⌉ and d = ⌈ln 1/δ⌉. The estimate is the row-wise minimum, so it never underestimates and exceeds the truth by more than εN with probability at most δ. Conservative update only raises counters up to min+count and reduces over-counting. Sketches merge by addition. A bounded candidate set on top gives top-k heavy hitters.
- **HyperLogLog**: 2^p registers store the maximum leading-zero rank of the hash suffix. The harmonic mean with bias constant α_m gives the estimate, with linear counting for small cardinalities. Merge is the register-wise maximum, and intersections use inclusion–exclusion.

## Complexity
| Structure | Update | Query | Space |
|-----------|--------|-------|-------|
| Bloom filter | O(k) | O(k) | O(n log(1/p)) bits |
| Count-Min | O(d) | O(d) | O(d·w) = O((1/ε) log(1/δ)) |
| HyperLogLog | O(1) | O(m) | O(2^p) bytes |

## Usage
```python
from stream_sketches import BloomFilter, CountMinSketch, HyperLogLog

bf = BloomFilter(capacity=10_000, fp_rate=0.01)
bf.add("alice"); print("alice" in bf)

cms = CountMinSketch(epsilon=0.002, delta=0.01, conservative=True)
for word in ["a", "b", "a"]: cms.add(word)
print(cms.estimate("a"))

hll = HyperLogLog(p=14)
hll.update(range(100_000)); print(hll.estimate())
```
Run `python examples/demo.py` for the full demo and `python -m pytest -q` for the 19 tests.

## Results (demo, 100k-event Zipf(1.1) stream, 10,422 distinct keys)
| Bloom target FP | Empirical | Memory |
|---|---|---|
| 0.1 | 0.0995 | 6.1 KiB |
| 0.01 | 0.0100 | 12.2 KiB |
| 0.001 | 0.0011 | 18.3 KiB |

Compared with roughly 611 KiB for an exact set, that is a 50× saving at 1% error.

| Count-Min (ε=0.002, δ=0.01, 53 KiB) | Mean error | Max error | Bound εN |
|---|---|---|---|
| Plain | 10.02 | 62 | 200 |
| Conservative | 4.84 | 35 | 200 |

The top-10 heavy hitters were recovered exactly (recall 1.00).

| HLL precision p | Bytes | Theoretical error | Empirical error (n=50k) |
|---|---|---|---|
| 6 | 64 | 0.1300 | 0.1234 |
| 10 | 1024 | 0.0325 | 0.0227 |
| 14 | 16384 | 0.0081 | 0.0067 |

With p=14, the distinct count of the stream was estimated at 10,490 (true 10,422). A union of two 60k ranges overlapping by 20k was estimated at 99,519 (true 100,000).

## References
- B. H. Bloom, "Space/time trade-offs in hash coding with allowable errors", CACM 1970.
- A. Kirsch, M. Mitzenmacher, "Less hashing, same performance: building a better Bloom filter", ESA 2006.
- P. Almeida et al., "Scalable Bloom Filters", Information Processing Letters 2007.
- G. Cormode, S. Muthukrishnan, "An improved data stream summary: the count-min sketch and its applications", J. Algorithms 2005.
- P. Flajolet, É. Fusy, O. Gandouet, F. Meunier, "HyperLogLog: the analysis of a near-optimal cardinality estimation algorithm", AofA 2007.

---

**Author:** Dr. Kainat Rizwan
