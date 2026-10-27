# Dynamic Programming Masterclass: Knapsack, Edit Distance, Matrix Chain and Bitmask DP

## Overview

`dp_toolkit` is a from-scratch collection of classic dynamic programs, written with
type hints and validated against brute-force references. Every solver returns not only
the optimal value but also a reconstruction (items chosen, edit script, parenthesisation,
tour), so results can be checked independently.

| Module | Contents |
|--------|----------|
| `knapsack.py` | 0/1, unbounded and bounded (binary splitting) knapsack; big-int bitset subset-sum |
| `sequences.py` | Weighted edit distance, edit scripts, banded edit distance, LCS, Hirschberg LCS, LIS |
| `interval.py` | Matrix-chain order, optimal BST (Knuth speed-up), burst balloons, polygon triangulation |
| `bitmask.py` | Held-Karp TSP, assignment DP, sum-over-subsets, Hamiltonian path counting, set cover |
| `memo.py` | Memoisation wrapper counting subproblems, timing helper |

## How it works

Each problem is expressed as a recurrence over overlapping subproblems:

* **Knapsack:** `f[i][c] = max(f[i-1][c], f[i-1][c-w_i] + v_i)`. The row update is vectorised
  with NumPy and a take-matrix is kept for reconstruction. Bounded multiplicities are split
  into 1, 2, 4, ... copies so that `k` copies cost `O(log k)` items.
* **Edit distance:** `d[i][j] = min(d[i-1][j-1] + [a_i != b_j], d[i-1][j] + 1, d[i][j-1] + 1)`;
  rolling rows give linear space, traceback of the full table gives an edit script.
  The banded variant only fills `|i-j| <= k`.
* **Hirschberg:** splits `a` in half, computes forward and backward LCS rows and picks the
  optimal split of `b`, recursing on both halves for `O(n+m)` space.
* **LIS:** patience sorting with binary search over pile tops.
* **Interval DP:** `c[i][j] = min_k c[i][k] + c[k+1][j] + p_i p_{k+1} p_{j+1}`; the optimal-BST
  roots are monotone (`root[i][j-1] <= root[i][j] <= root[i+1][j]`), giving Knuth's `O(n^2)`.
* **Bitmask DP:** states are subsets of visited cities, used jobs or covered elements.

## Complexity

| Algorithm | Time | Space |
|-----------|------|-------|
| 0/1 knapsack | O(nW) | O(nW) bits (O(W) values) |
| Unbounded knapsack | O(nW) | O(W) |
| Bounded knapsack | O(nW log k) | O(nW log k) |
| Bitset subset-sum | O(nT/w) | O(nT/w) for reconstruction |
| Edit distance | O(nm) | O(min(n,m)) |
| Banded edit distance | O(kn) | O(k) |
| LCS / Hirschberg | O(nm) | O(nm) / O(n+m) |
| LIS | O(n log n) | O(n) |
| Matrix chain, balloons, triangulation | O(n^3) | O(n^2) |
| Optimal BST (Knuth) | O(n^2) | O(n^2) |
| Held-Karp | O(2^n n^2) | O(2^n n) |
| Assignment DP | O(2^n n) | O(2^n) |
| Sum over subsets | O(2^b b) | O(2^b) |

## Usage

```python
from dp_toolkit import knapsack_01, edit_script, apply_script, matrix_chain, parenthesize, held_karp

res = knapsack_01([12, 7, 11, 8, 9], [24, 13, 23, 15, 16], 26)
print(res.value, res.counts)

script = edit_script("kitten", "sitting")
assert apply_script("kitten", script) == "sitting"

cost, split = matrix_chain([10, 30, 5, 60])
print(cost, parenthesize(split))   # 4500 ((A0 x A1) x A2)
```

Run the tests with `python -m pytest -q` and the demo with `python examples/demo.py`.

## Results

Output of `examples/demo.py` (seeded; timings from one run on a CI machine):

```
0/1 knapsack n=100 W=2000        value 9054        1.4 ms
unbounded n=10 W=2000            value 45600       4.7 ms
bitset subset-sum n=60           found             8.5 ms
edit distance, 300x300 DNA       159 (banded k=159 agrees, script verified)
LCS 300x300 DNA                  191 (full table 0.04 s, Hirschberg 0.03 s)
LIS of 100000 random ints        length 606        32 ms
matrix chain n=40                141140 mults
Held-Karp, 12 cities             tour 3.4133       0.04 s
Hamiltonian paths in K9          362880 (= 9!)
```

The 23 tests compare every solver with brute force or an independent formulation
(permutations for TSP and assignment, expanded 0/1 items for bounded knapsack,
round-tripped edit scripts, the CLRS optimal-BST example with cost 2.75).

## References

* R. Bellman, *Dynamic Programming*, Princeton University Press, 1957.
* T. Cormen, C. Leiserson, R. Rivest, C. Stein, *Introduction to Algorithms*, 4th ed., MIT Press, 2022.
* D. S. Hirschberg, "A linear space algorithm for computing maximal common subsequences", *CACM* 18(6), 1975.
* M. Held and R. Karp, "A dynamic programming approach to sequencing problems", *J. SIAM* 10(1), 1962.
* D. E. Knuth, "Optimum binary search trees", *Acta Informatica* 1, 1971.
* V. Levenshtein, "Binary codes capable of correcting deletions, insertions and reversals", 1966.

---

**Author:** Dr. Kainat Rizwan
