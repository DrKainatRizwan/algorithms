# LLL Lattice Reduction, BKZ and Knapsack / LCG Attacks

From-scratch lattice toolkit in pure Python (standard library only): exact integral LLL, BKZ with an enumeration SVP oracle, exact SVP/CVP and Babai's nearest-plane algorithm, and three classic cryptanalytic applications — the low-density subset-sum attack, integer-relation / minimal-polynomial recovery, and state recovery for a truncated linear congruential generator.

## How it works

* **LLL** (`lll.py`): Cohen's integral variant (Alg. 2.6.7). Gram-Schmidt data is stored as integers `d_i` and `lambda_ij = d_j mu_ij`, so no floating point is involved. The basis is size-reduced (`|mu_ij| <= 1/2`) and Lovász-swapped (`|b_k*|^2 >= (delta - mu^2)|b_{k-1}*|^2`). It returns a vector with `|b_1| <= (4/(4 delta-1))^((n-1)/4) det^(1/n)`.
* **Enumeration** (`enumeration.py`): Schnorr-Euchner depth-first search over Gram-Schmidt coordinates with zig-zag ordering and a shrinking radius. It powers exact `svp` and `cvp`; `babai_nearest_plane` gives the polynomial-time approximation.
* **BKZ** (`bkz.py`): slides a window of size beta over the basis, finds the shortest vector of each projected block by enumeration and inserts it with a Euclid-style unimodular transform, then re-runs LLL, until a tour changes nothing.
* **Attacks** (`attacks.py`):
  * CJLOSS knapsack lattice with rows `(2e_i | N a_i)` and `(1..1 | N s)`; the solution appears as a `(+-1,...,+-1 | 0)` vector.
  * Integer relations: scale reals by `10^d`, reduce `(e_i | round(C v_i))`, read off a short row with tiny residual; used for minimal polynomials.
  * Truncated LCG: hidden low bits `e_i` satisfy `e_{i+1} - a e_i = t_i (mod m)`; the centred error is a bounded-distance CVP target in the lattice `{v : v_{i+1} = a v_i mod m}`.

## Complexity

| Routine | Time | Space |
|---|---|---|
| LLL (dimension n, entries <= B bits) | O(n^4 log B) arithmetic ops on O(n log B)-bit integers | O(n^2) integers |
| Babai nearest plane | O(n^2) after reduction | O(n^2) |
| Enumeration (SVP/CVP) | 2^O(n log n) worst case, much better after LLL/BKZ | O(n) beyond the basis |
| BKZ-beta | tours x n x enumeration(beta) + LLL | O(n^2) |

## Usage

```python
import random
from lattice_reduction import lll_reduce, svp, solve_knapsack, random_knapsack

basis = [[1, 0, 0, 12345], [0, 1, 0, 67890], [0, 0, 1, 13579]]
print(lll_reduce(basis))

w, secret, s = random_knapsack(26, 0.9, random.Random(1))
x = solve_knapsack(w, s, block_size=8)   # LLL first, BKZ fallback
assert sum(a * b for a, b in zip(w, x)) == s
```

Run `python examples/demo.py` for the full demo and `python -m pytest -q` for the tests (19 tests).

## Results (output of the demo)

```
== Reduction quality on random q-ary lattices (dim 24, q=2^24) ==
LLL  : |b1|=      5589.4  root-Hermite=1.0130  swaps=6134  0.30s
BKZ-6 : |b1|=      5589.4  root-Hermite=1.0130  tours=3  2.19s
BKZ-10: |b1|=      5589.4  root-Hermite=1.0130  tours=3  2.25s

== Low-density subset-sum attack (n=26, 8 instances each) ==
density  LLL-solved  BKZ-8-solved
  0.60       8/8          8/8
  0.90       8/8          8/8
  1.00       7/8          8/8

== Minimal polynomials via integer relations ==
sqrt2+sqrt3   -> coeffs (const..lead) = [1, 0, -10, 0, 1]
2^(1/3)+1     -> coeffs (const..lead) = [-3, 3, -3, 1]
golden ratio  -> coeffs (const..lead) = [-1, -1, 1]

== Truncated LCG state recovery (m=2^31, 8 outputs) ==
hidden bits  success/10
    10         10
    16         10
    20         10
    24         10
    27          4
predicted next states: [1503445755, 1950986008, 2068385137] | actual: [1503445755, 1950986008, 2068385137] | match: True

== Exact SVP on a dim-12 q-ary lattice ==
shortest vector norm = 32.696 (LLL gave 32.833) in 0.01s
```

Notes: the knapsack attack is perfect well below density 1 and degrades near it, where BKZ rescues instances LLL misses. Truncated-LCG recovery succeeds reliably while the unknown bits are well below `m^((n-1)/n)` (about 2^27 for 8 outputs), and starts failing at 27 hidden bits of 31. On the random q-ary lattice LLL already reaches a root-Hermite factor near 1.013, so BKZ gives no improvement on that instance at the cost of several seconds.

## References

* A. K. Lenstra, H. W. Lenstra, L. Lovász, *Factoring polynomials with rational coefficients*, Math. Ann. 261 (1982).
* H. Cohen, *A Course in Computational Algebraic Number Theory*, Springer (1993), §2.6.
* C. P. Schnorr, M. Euchner, *Lattice basis reduction: improved practical algorithms and solving subset sum problems*, Math. Programming 66 (1994).
* M. J. Coster et al., *Improved low-density subset sum algorithms*, Computational Complexity 2 (1992).
* J. C. Lagarias, A. M. Odlyzko, *Solving low-density subset sum problems*, JACM 32 (1985).
* L. Babai, *On Lovász' lattice reduction and the nearest lattice point problem*, Combinatorica 6 (1986).
* H. Ritter, *Breaking knapsack cryptosystems by max-norm enumeration*; J. Stern, *Secret linear congruential generators are not cryptographically secure*, FOCS 1987.
* N. Gama, P. Nguyen, *Predicting lattice reduction*, EUROCRYPT 2008.

---

**Author:** Dr. Kainat Rizwan
