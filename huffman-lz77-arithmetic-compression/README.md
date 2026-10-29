# Huffman, LZ77 and Arithmetic Coding Compression Suite

## Overview
From-scratch lossless compressors for byte streams, built on a shared bit-level I/O layer:

- **Huffman** – canonical prefix codes with length limiting and a table-driven decoder.
- **LZ77** – sliding-window parsing with hash chains and lazy matching, followed by Huffman coding of the literal/length alphabet (a simplified DEFLATE layout).
- **Arithmetic coding** – 32-bit integer range coder (Witten–Neal–Cleary) with an adaptive Fenwick-tree frequency model, order-0 or order-1 contexts.
- **Container** – `pack`/`unpack` tries each codec and keeps the smallest, falling back to stored mode on incompressible input.

Modules: `bitio`, `huffman`, `lz77`, `arithmetic`, `entropy` (entropy measures and synthetic data), `suite` (container and benchmark).

## How it works
**Huffman.** Code lengths come from repeatedly merging the two lightest subtrees in a heap. Codes are then assigned canonically (sorted by length, then symbol), so only the lengths need to be stored. Decoding uses per-length first-code tables. The average length L satisfies H ≤ L < H + 1.

**LZ77.** For each position, a hash chain over 3-byte keys yields candidate earlier occurrences within the window; the longest match is taken. With lazy matching, a match is deferred if the next position yields a longer one. Literals and match lengths share one Huffman alphabet (256 literals + 256 length symbols); distances are sent as a 5-bit width followed by the low bits.

**Arithmetic coding.** The interval [low, high] over 32-bit integers is narrowed to the sub-interval of each symbol. Leading bits are emitted when both ends agree, and the straddle case (E3) is handled with pending bits. The model starts uniform and adds 32 to a symbol's count on each occurrence, halving all counts when the total reaches 2^16. An explicit EOF symbol terminates the stream. Because symbols may cost fractional bits, the rate approaches the entropy even when it is below 1 bit/symbol, which Huffman cannot reach.

## Complexity
| Codec | Time | Space |
|-------|------|-------|
| Huffman | O(n + σ log σ) | O(σ) |
| LZ77 | O(n · chain) | O(n) for chains |
| Arithmetic | O(n log σ) via Fenwick tree | O(σ) per context |

σ is the alphabet size, n the input length, `chain` the maximum chain length examined.

## Usage
```python
from compression_suite import pack, unpack, huffman, lz77, arithmetic

data = b"abracadabra" * 100
blob = pack(data)               # best of all codecs
assert unpack(blob) == data

small = lz77.compress(data, window=4096, max_chain=32)
ac = arithmetic.compress(data, order=1)
assert arithmetic.decompress(ac) == data
```

Run the tests with `python -m pytest -q` and the benchmark with `python examples/demo.py`.

## Results
20,000-byte synthetic inputs (sizes in bytes; bits per byte in parentheses), from `examples/demo.py`:

| Dataset | H0 | Huffman | LZ77 | Arith-0 | Arith-1 |
|---------|----|---------|------|---------|---------|
| text (Zipf words) | 3.63 | 9302 (3.72) | 6973 (2.79) | 9124 (3.65) | **5213 (2.09)** |
| repetitive | 7.62 | 19277 (7.71) | **1548 (0.62)** | 19382 (7.75) | 5299 (2.12) |
| skewed (p0=0.9) | 0.86 | 3644 (1.46) | 3944 (1.58) | **2195 (0.88)** | 2323 (0.93) |
| random | 7.99 | 20164 (8.07) | 20332 (8.13) | 20312 (8.13) | 24026 (9.61) |

Huffman is stuck above 1 bit/symbol on the skewed source while arithmetic coding sits within 0.02 bits of the entropy. LZ77 wins by a large margin on repetitive data, order-1 modelling wins on text, and the container stores random data with a single byte of overhead.

## References
- D. A. Huffman, "A Method for the Construction of Minimum-Redundancy Codes", Proc. IRE, 1952.
- J. Ziv and A. Lempel, "A Universal Algorithm for Sequential Data Compression", IEEE Trans. Inf. Theory, 1977.
- I. H. Witten, R. M. Neal and J. G. Cleary, "Arithmetic Coding for Data Compression", CACM, 1987.
- P. Deutsch, "DEFLATE Compressed Data Format Specification", RFC 1951, 1996.
- P. Fenwick, "A New Data Structure for Cumulative Frequency Tables", Software: Practice and Experience, 1994.
- T. Cover and J. Thomas, *Elements of Information Theory*, Wiley.

---

**Author:** Dr. Kainat Rizwan
