# Locality-Sensitive Hashing Toolkit: MinHash, SimHash and Random-Projection Nearest Neighbour Search

## Overview
Locality-sensitive hashing (LSH) maps similar items to the same bucket with high probability, turning nearest-neighbour search into a few hash lookups plus a small re-ranking step. This project implements, from scratch with NumPy only:

- **MinHash** signatures estimating Jaccard similarity of shingled documents.
- **Banded MinHash index** with automatic `(bands, rows)` selection for a target similarity threshold.
- **SimHash** 64-bit fingerprints with an *exact* Hamming-ball index (block tables via pigeonhole).
- **Random-hyperplane LSH** (cosine) and **Gaussian p-stable LSH** (Euclidean), combined in a multi-table `VectorLSHIndex` with exact re-ranking.

## How it works
- **MinHash**: for a random hash `h(x) = (ax+b) mod p`, `P[min h(A) = min h(B)] = J(A,B)`. A signature of `m` such minima gives an unbiased estimate with standard error `sqrt(J(1-J)/m)`.
- **Banding**: the signature is cut into `b` bands of `r` rows; two sets become candidates if any band matches entirely, which happens with probability `1-(1-s^r)^b` (an S-curve with midpoint near `(1/b)^(1/r)`). `optimal_bands` minimises weighted false-positive and false-negative areas of this curve.
- **SimHash**: sum signed token-hash bits weighted by token frequency, keep signs. Hamming distance approximates angular distance. To find all fingerprints within distance `k`, split into `k+1` blocks; any match must agree exactly on one block, so lookup is complete.
- **Hyperplane LSH**: `P[bit agrees] = 1 - θ/π`. Concatenating `K` bits per table and using `L` tables amplifies the gap between near and far points.
- **p-stable LSH**: `h(x) = floor((a·x + b)/w)` with Gaussian `a`; collision probability is a decreasing function of L2 distance (Datar et al.).

## Complexity
| Operation | Time | Space |
|---|---|---|
| MinHash signature of set size `n` | O(m·n) | O(m) |
| Banded insert / query | O(b) lookups + candidates | O(N·m) |
| SimHash fingerprint | O(tokens·bits) | O(1) |
| SimHash index query | O(k+1) lookups + candidates | O(N·(k+1)) |
| Vector index build | O(N·L·K·d) | O(N·L) |
| Vector query | O(L·K·d + C·d), C = candidates | — |

## Usage
```python
from lsh_toolkit import MinHasher, MinHashLSHIndex, char_shingles

index = MinHashLSHIndex.for_threshold(MinHasher(128, seed=0), threshold=0.5)
index.insert("a", char_shingles("the quick brown fox jumps over the lazy dog"))
index.insert("b", char_shingles("an entirely different sentence about hashing"))
print(index.query(char_shingles("the quick brown fox jumped over the lazy dog"), 0.3))
```

```
python -m pytest -q
python examples/demo.py
```

## Results
From `examples/demo.py` (seeded):

- MinHash estimate of J = 0.333: error 0.146 (16 perms), 0.083 (64), 0.025 (256), 0.035 (1024).
- Banding for threshold 0.7 with 128 permutations chose 14 bands × 9 rows: candidate probability 0.000 / 0.027 / 0.438 / 0.999 at J = 0.3 / 0.5 / 0.7 / 0.9.
- 600 documents (100 sources × 5 copies, 5% word edits): recall 0.999 with on average 6.0 candidates out of 600.
- SimHash index (k=3) returned exactly the brute-force result set.
- 4000 points, dim 32, recall@10 against brute force:

| Metric | Tables | Recall@10 | Avg. candidates (of 4000) |
|---|---|---|---|
| cosine | 4 | 0.968 | 296.2 |
| cosine | 12 | 1.000 | 417.2 |
| L2 | 4 | 0.554 | 63.6 |
| L2 | 12 | 0.858 | 120.6 |

## References
- A. Broder, "On the resemblance and containment of documents", SEQUENCES 1997.
- M. Charikar, "Similarity estimation techniques from rounding algorithms", STOC 2002.
- G. Manku, A. Jain, A. Das Sarma, "Detecting near-duplicates for web crawling", WWW 2007.
- M. Datar, N. Immorlica, P. Indyk, V. Mirrokni, "Locality-sensitive hashing scheme based on p-stable distributions", SoCG 2004.
- J. Leskovec, A. Rajaraman, J. Ullman, *Mining of Massive Datasets*, ch. 3.

---

**Author:** Dr. Kainat Rizwan
