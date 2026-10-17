# Hungarian Algorithm and Auction Algorithm for Optimal Assignment

## Overview

From-scratch NumPy implementations of two classic solvers for the linear assignment problem: given an `n x m` cost matrix, match every row of the smaller side to a distinct column so that the total cost is minimal (or maximal).

- **Hungarian / Kuhn-Munkres** – shortest augmenting paths with dual potentials, rectangular matrices, forbidden (`inf`) entries and a verifiable optimality certificate.
- **Bertsekas auction** – price-based bidding with epsilon scaling; exact for integer costs, within `n * eps` otherwise.
- **Extras** – bottleneck (min-max) assignment, gated data association for tracking, LP-dual certificate checking and "forced pair" regret queries.

## How it works

**Hungarian.** Rows are inserted one at a time. For each new row a Dijkstra-style search over reduced costs `c_ij - u_i - v_j >= 0` finds the cheapest alternating path to a free column; potentials are shifted by the minimum slack so that reduced costs stay non-negative and matched edges stay tight. Weak duality (`sum u + sum v <= optimum`) makes the final potentials a proof of optimality.

**Auction.** Each unassigned person bids on the object with the best value `b_ij - p_j`, raising its price by `(best - second best) + eps`. The outbid owner re-enters the pool. At termination the matching satisfies eps-complementary slackness so it is within `n * eps` of optimal; choosing `eps < 1/n` with integer data gives the exact optimum. Epsilon scaling starts with a large `eps` and divides it by a constant each phase while keeping prices, which sharply cuts the number of bids. Rectangular problems are padded with zero-benefit dummy persons.

**Bottleneck.** Binary search over sorted distinct costs, testing each threshold with a maximum bipartite matching; ties are then broken by min-sum assignment under the threshold.

**Gated association.** Costs above a gate are replaced by a prohibitive penalty, so the solution maximises the number of valid matches first and total cost second.

## Complexity

| Method | Time | Space |
|--------|------|-------|
| Hungarian (`n <= m`) | `O(n^2 m)` | `O(nm)` |
| Auction (eps scaling) | `O(n m log(nC))` bids-work, typically near `O(n^2 log(nC))` | `O(nm)` |
| Bottleneck | `O(log(nm) * n * nm)` matching + one Hungarian | `O(nm)` |

`C` is the largest cost magnitude.

## Usage

```python
import numpy as np
from assignment_solvers import hungarian, auction_assignment, verify_duals

cost = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]], dtype=float)
res = hungarian(cost)
print(res.as_dict(), res.cost)        # {0: 1, 1: 0, 2: 2} 5.0
print(verify_duals(cost, res))        # True
print(auction_assignment(cost).cost)  # 5.0
```

Run the tests and demo with:

```
python -m pytest -q
python examples/demo.py
```

## Results

Integer costs in `[0, 1000)`, single run on a CI machine (pure Python loops over NumPy rows):

| n | Hungarian (ms) | Auction (ms) | Bids | Phases | Same optimum |
|---|---------------:|-------------:|-----:|-------:|:------------:|
| 25 | 3.1 | 2.7 | 338 | 7 | yes |
| 50 | 6.3 | 5.3 | 696 | 8 | yes |
| 100 | 18.7 | 8.5 | 1852 | 8 | yes |
| 200 | 37.5 | 21.2 | 4384 | 9 | yes |

Effect of epsilon scaling (n = 60, costs up to 10000): scale 5 needs 946 bids against 7144 with a single phase. On a 40x40 instance the bottleneck solution lowers the largest matched edge from 9 to 6 at the price of a total cost of 128 instead of 117.

The test suite (36 tests) checks every solver against exhaustive search, validates dual certificates, rectangular and maximisation cases, forbidden entries, and the auction error bound.

## References

- H. W. Kuhn, "The Hungarian method for the assignment problem", *Naval Research Logistics Quarterly*, 1955.
- J. Munkres, "Algorithms for the assignment and transportation problems", *J. SIAM*, 1957.
- J. Edmonds and R. M. Karp, "Theoretical improvements in algorithmic efficiency for network flow problems", *JACM*, 1972.
- D. P. Bertsekas, "The auction algorithm: a distributed relaxation method for the assignment problem", *Annals of Operations Research*, 1988.
- R. Burkard, M. Dell'Amico, S. Martello, *Assignment Problems*, SIAM, 2009.

---

**Author:** Dr. Kainat Rizwan
