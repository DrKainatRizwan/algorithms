# Suffix Automaton and Suffix Array Toolkit

## Overview
A from-scratch toolkit for large-scale string analysis in pure Python (standard library only). It provides a suffix array (prefix doubling with radix sort), Kasai's LCP array, an online suffix automaton, a binary-search pattern index, a Burrows-Wheeler transform and several classic analyses built on them.

## How it works
- **Suffix array**: suffixes are ranked by their first 2^k characters; each round uses two stable counting sorts, doubling k until all ranks are distinct.
- **Kasai LCP**: walks suffixes in text order and reuses the previous LCP minus one, giving linear total work.
- **Suffix automaton**: built online by appending characters; clone states keep the automaton minimal (at most 2n-1 states). Occurrence counts are propagated up suffix links in decreasing length order.
- **Analyses**: distinct substrings (`n(n+1)/2 - sum(LCP)` or `sum(len(v)-len(link(v)))`), longest repeated substring, longest common substring (automaton scan, or SA of `a#b`), k-th lexicographic substring, pattern counting/location, BWT and its inverse.

## Complexity
| Component | Time | Space |
|---|---|---|
| Suffix array (doubling) | O(n log n) | O(n) |
| Kasai LCP | O(n) | O(n) |
| Suffix automaton build | O(n) amortised (dict transitions) | O(n·σ) |
| SA pattern search | O(m log n) | O(n) |
| SAM count / contains | O(m) after O(n log n) one-off sort for counts | O(n) |

## Usage
```python
from suffix_toolkit import SuffixAutomaton, SuffixArrayIndex, build_suffix_array

build_suffix_array("banana")            # [5, 3, 1, 0, 4, 2]
sam = SuffixAutomaton("abracadabra")
sam.count("abra")                       # 2
SuffixArrayIndex("abracadabra").find_all("abra")  # [0, 7]
```

Run `python examples/demo.py` for the demo and `python -m pytest -q` for the tests.

## Results
Random DNA text, single run (`examples/demo.py`):

| n | SA doubling | SA naive | SAM build | SAM states | distinct substrings |
|---|---|---|---|---|---|
| 1,000 | 1.2 ms | 0.3 ms | 0.7 ms | 1,621 | 496,307 |
| 5,000 | 5.7 ms | 4.9 ms | 3.6 ms | 8,094 | 12,475,891 |
| 20,000 | 24.0 ms | – | 14.7 ms | 32,411 | 199,883,247 |

The naive sort is competitive on random text (suffixes differ early) but degrades quadratically on repetitive input; the tests cover such cases. Both distinct-substring methods agree.

## References
- Manber & Myers, "Suffix Arrays: A New Method for On-Line String Searches", 1993.
- Kasai et al., "Linear-Time Longest-Common-Prefix Computation in Suffix Arrays", 2001.
- Blumer et al., "The Smallest Automaton Recognizing the Subwords of a Text", 1985.
- Crochemore & Rytter, *Text Algorithms*, 1994.
- Burrows & Wheeler, "A Block-Sorting Lossless Data Compression Algorithm", 1994.

---

**Author:** Dr. Kainat Rizwan
