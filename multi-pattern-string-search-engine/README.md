# Rabin-Karp, KMP, Z-Algorithm and Aho-Corasick: Multi-Pattern Search Engine

## Overview
From-scratch implementations of the classic exact string matchers, plus a small search engine on top of an Aho-Corasick automaton. Pure Python 3.12, standard library only.

Modules in `pattern_search/`:
- `single_pattern.py` – naive oracle, KMP (prefix function), Z-array and Z-based search.
- `rolling_hash.py` – polynomial rolling hash mod 2^61-1, Rabin-Karp and a length-bucketed multi-pattern variant.
- `aho_corasick.py` – trie with failure links and dictionary (output) links.
- `engine.py` – case folding, whole-word filtering, overlap policies (`all`, `leftmost_longest`, `leftmost_first`, `max_cover`), highlight / redact / replace.
- `analysis.py` – borders, shortest period, distinct-pattern counts.

## How it works
- **KMP** precomputes for each prefix its longest proper border, so on a mismatch the pattern shifts without re-reading text.
- **Z-algorithm** computes, for every position, the longest common prefix with the string; matches are positions in `pattern + sentinel + text` with `z >= m`.
- **Rabin-Karp** compares a rolling hash of each window to the pattern hash and verifies hits, giving expected linear time. The multi-pattern version buckets patterns by length and does one pass per distinct length.
- **Aho-Corasick** merges all patterns into a trie, adds BFS failure links, and a dictionary link that jumps to the next terminal suffix state, so every occurrence is reported in O(1) amortised after the scan reaches it.
- `max_cover` solves weighted interval scheduling (binary search + DP) to maximise the number of characters covered by non-overlapping matches.

## Complexity
| Algorithm | Preprocessing | Search | Space |
|---|---|---|---|
| KMP | O(m) | O(n) | O(m) |
| Z search | – | O(n + m) | O(n + m) |
| Rabin-Karp | O(m) | O(n) expected, O(nm) worst | O(1) |
| Rabin-Karp multi | O(Σ\|p\|) | O(n·L + z), L = distinct lengths | O(n + k) |
| Aho-Corasick | O(Σ\|p\|) | O(n + z) | O(Σ\|p\|) |

## Usage
```python
from pattern_search import AhoCorasick, SearchEngine

ac = AhoCorasick(["he", "she", "his", "hers"])
print([(m.start, m.pattern) for m in ac.find_all("ushers")])
# [(1, 'she'), (2, 'he'), (2, 'hers')]

eng = SearchEngine(["password", "token"], case_sensitive=False, whole_word=True)
print(eng.redact("Send the Password, not tokens."))
```

## Results
`python examples/demo.py` (random DNA text of 200,000 characters, seed 42):

| Task | Time |
|---|---|
| Single 6-mer, naive / KMP / Z / Rabin-Karp | 21 / 18 / 57 / 81 ms (55 hits) |
| 200 patterns: Aho-Corasick build | 1.0 ms (995 states) |
| 200 patterns: Aho-Corasick scan | 54 ms (2664 hits) |
| 200 patterns: Rabin-Karp multi | 553 ms |
| 200 patterns: KMP run once per pattern | 3342 ms |

Aho-Corasick is about 60x faster than repeated KMP here. In pure Python the naive loop is competitive on single short patterns because slicing runs in C. All matchers are checked against the naive oracle in the test suite (`python -m pytest -q`, 26 tests).

## References
- D. E. Knuth, J. H. Morris, V. R. Pratt, "Fast Pattern Matching in Strings", SIAM J. Computing 6(2), 1977.
- R. M. Karp, M. O. Rabin, "Efficient randomized pattern-matching algorithms", IBM J. Res. Dev. 31(2), 1987.
- A. V. Aho, M. J. Corasick, "Efficient string matching: an aid to bibliographic search", CACM 18(6), 1975.
- D. Gusfield, *Algorithms on Strings, Trees, and Sequences*, Cambridge University Press, 1997.

---

**Author:** Dr. Kainat Rizwan
