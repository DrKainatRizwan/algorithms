# Stable Matching Toolkit: Gale-Shapley, Hospitals/Residents, Irving's Roommates and the Rotation Lattice

## Overview
From-scratch implementations of the main stable-matching algorithms, using only the standard library and NumPy (for instance generation):

- `gale_shapley` – deferred acceptance for stable marriage (incomplete lists supported), blocking-pair checker.
- `hospitals_residents` – many-to-one matching with capacities, resident- and hospital-oriented.
- `irving_roommates` – Irving's two-phase algorithm for the stable roommates problem (may report no solution).
- `lattice` – rotations, enumeration of all stable matchings, and egalitarian / minimum-regret / sex-equal selection.
- `analysis` – brute-force oracles and Monte-Carlo experiments.

## How it works
**Deferred acceptance.** Free proposers propose down their lists; a receiver keeps the best offer so far. The result is stable and optimal for the proposing side (and worst for the other side).

**Rotations.** Starting from the proposer-optimal matching, lists are reduced (a woman is kept in a man's list only if she prefers him to her current partner). An exposed rotation is a cycle of men m_i -> the man matched to m_i's second choice. Eliminating it shifts each man to his second choice and prunes the lists. Every stable matching is reachable by eliminating rotations in a poset order, which the enumerator explores with deduplication.

**Irving's algorithm.** Phase 1 is a proposal sequence in which every acceptance deletes all worse candidates from the acceptor's list (symmetrically). Phase 2 repeatedly finds a rotation (p_i, q_i = second(p_i), p_{i+1} = last(q_i)), deletes the pairs (q_i, p_{i+1}) and re-trims; an empty list proves no stable matching exists.

**Hospitals/residents.** Deferred acceptance with capacities; the rural hospitals theorem (same hospitals filled to the same count in every stable assignment) is checked in the tests.

## Complexity
| Algorithm | Time | Space |
|---|---|---|
| Gale-Shapley | O(n^2) worst, ~n ln n proposals on random input | O(n^2) lists |
| Hospitals/residents | O(total list length) | O(total list length) |
| Irving (this implementation) | O(n^3) worst (list scans are not linked-list optimised; the textbook version is O(n^2)) | O(n^2) |
| Rotation enumeration | O(n^2) per matching generated; output may be exponential | O(n^2) per state |

## Usage
```python
from stable_matching import *
men, women = random_marriage_instance(8, seed=20)
m = gale_shapley(men, women).match
assert is_stable(men, women, m)
all_matchings = enumerate_stable_matchings(men, women)
best = egalitarian_matching(men, women)

prefs = random_roommates_instance(10, seed=1)
pairing = irving_roommates(prefs)      # None if no stable matching exists
```

## Results (`python examples/demo.py`)
Random instance (n=8, seed 20): 6 stable matchings, 4 rotations, enumeration verified against brute force over all 8! permutations.

Mean proposals over 20 random instances:

| n | men propose | women propose | n ln n |
|---|---|---|---|
| 50 | 213.9 | 191.4 | 195.6 |
| 100 | 489.8 | 521.2 | 460.5 |
| 200 | 1140.1 | 1221.5 | 1059.7 |
| 400 | 2457.1 | 2786.5 | 2396.6 |

Probability that a random roommates instance has a stable matching (200 trials): n=6 0.93, n=10 0.90, n=20 0.81, n=40 0.75.

Mean number of stable matchings in random marriage instances (20 trials): n=6 1.80, n=8 1.90, n=10 3.20, n=12 3.60.

Run the tests with `python -m pytest -q` (147 tests, including exhaustive cross-checks against brute force).

## References
- D. Gale, L. S. Shapley, "College Admissions and the Stability of Marriage", Amer. Math. Monthly, 1962.
- R. W. Irving, "An efficient algorithm for the stable roommates problem", J. Algorithms, 1985.
- R. W. Irving, P. Leather, "The complexity of counting stable marriages", SIAM J. Comput., 1986.
- D. Gusfield, R. W. Irving, *The Stable Marriage Problem: Structure and Algorithms*, MIT Press, 1989.
- A. E. Roth, "The evolution of the labor market for medical interns and residents", J. Political Economy, 1984.

---

**Author:** Dr. Kainat Rizwan
