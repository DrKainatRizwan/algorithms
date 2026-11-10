# FM-Index and Wavelet Tree for Compressed Full-Text Search

## Overview
A from-scratch compressed text index: suffix array construction (prefix doubling), Kasai LCP, the Burrows-Wheeler transform and its inverse, a rank/select bit vector, a wavelet tree, and an FM-index supporting `count`, `locate` and `extract` without keeping the original text.

## How it works
1. **BWT.** Sort all suffixes of `T$`; the last column of the sorted rotations is the BWT. It is reversible through the LF-mapping.
2. **Wavelet tree.** The BWT is stored in a balanced tree of bit vectors, giving `rank(c, i)`, `select(c, k)`, `access(i)` and range quantiles in O(log σ).
3. **Backward search.** For pattern `P`, the suffix-array interval is refined from the last character: `lo = C[c] + rank(c, lo)`, `hi = C[c] + rank(c, hi)`. The interval size is the occurrence count.
4. **Locate.** Only every `s`-th text position keeps its suffix-array entry; other rows walk LF until a sample is hit (≤ s steps).
5. **Extract.** Starting from a sampled inverse-SA row after the requested range, LF steps emit text backwards.

## Complexity
| Operation | Time | 
|---|---|
| Build SA (doubling) | O(n log² n) |
| count(P) | O(\|P\| log σ) |
| locate(P) | O((\|P\| + occ·s) log σ) |
| extract(i, L) | O((L + s) log σ) |

Space: n·⌈log σ⌉ bits for the wavelet tree plus (n/s) sampled entries.

## Usage
```python
from compressed_index import FMIndex
fm = FMIndex(b"mississippi", sa_sample=4)
fm.count(b"ssi")      # 2
fm.locate(b"ssi")     # [2, 5]
fm.extract(4, 4)      # b"issi"
```

## Results
`python examples/demo.py` on a random 20,000-base DNA string, 200 patterns of length 8:

| SA sample | locate (200 patterns) | naive scan | sampled SA size |
|---|---|---|---|
| 1 | 3.1 ms | 11.2 ms | 160,008 B |
| 8 | 4.0 ms | 11.0 ms | 20,008 B |
| 32 | 6.8 ms | 10.6 ms | 5,008 B |

Sampling trades a modest slowdown for a 32× reduction in suffix-array storage. Results are validated against brute-force search in the test suite (`python -m pytest -q`).

## References
- Burrows, M. & Wheeler, D. (1994). *A Block-sorting Lossless Data Compression Algorithm.*
- Ferragina, P. & Manzini, G. (2000). *Opportunistic Data Structures with Applications.* FOCS.
- Grossi, R., Gupta, A. & Vitter, J. (2003). *High-order entropy-compressed text indexes.* SODA.
- Manber, U. & Myers, G. (1993). *Suffix arrays.* SIAM J. Comput.
- Kasai et al. (2001). *Linear-time longest-common-prefix computation in suffix arrays.*

---

**Author:** Dr. Kainat Rizwan
