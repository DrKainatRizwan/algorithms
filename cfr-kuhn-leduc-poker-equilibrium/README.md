# Counterfactual Regret Minimisation for Kuhn and Leduc Poker

## Overview

From-scratch implementations of counterfactual regret minimisation (CFR) for two-player zero-sum imperfect-information games. The package defines Kuhn poker and Leduc hold'em, compiles each into flat arrays, and solves them with full-traversal CFR (vanilla, CFR+, linear CFR, discounted CFR) and Monte-Carlo CFR (external and outcome sampling). Solutions are checked with an exact best-response calculator and against the analytic Nash family of Kuhn poker.

Only Python 3.12 and NumPy are needed.

## How it works

* **Game tree compilation** (`tree.py`): a game is enumerated once into pre-order node arrays, edge arrays grouped into depth layers, and information sets with strategy slots. Perfect recall is verified during the build.
* **Vectorised CFR** (`cfr.py`): each pass propagates reach probabilities top-down and node values bottom-up with NumPy layer operations. The counterfactual regret of action `a` at infoset `I` is `sum_h pi_{-i}(h) (v(ha) - v(h))`, accumulated with `np.bincount`. Strategies come from regret matching; the *average* strategy converges to a Nash equilibrium.
* **Variants**: *vanilla* (simultaneous, uniform averaging), *CFR+* (alternating updates, regret floor at 0, linear averaging), *linear CFR* and *DCFR* (discounting with exponents alpha, beta, gamma).
* **Monte-Carlo CFR** (`mccfr.py`): external sampling samples chance and opponent actions and enumerates the traverser's actions; outcome sampling samples one trajectory with an epsilon-greedy behaviour policy and uses importance-weighted regret estimates.
* **Evaluation** (`evaluation.py`): exact expected values, an exact best response (infosets processed in order of the player's own decision depth) and exploitability `(br_0 + br_1) / 2`.
* **Analysis** (`analysis.py`): the closed-form one-parameter family of Kuhn equilibria (game value -1/18 for the first player) and the regret bound used in the tests.

## Complexity

Let `N` be the number of tree nodes, `L` the depth, `S` the number of strategy slots and `T` the iterations.

| Method | Time | Space |
|---|---|---|
| Vanilla / CFR+ / DCFR | `O(T * N)` (vectorised over `L` layers) | `O(N + S)` |
| Best response | `O(N * (levels + 1))` | `O(N + S)` |
| External sampling | `O(T * N_sampled)`, where `N_sampled` is far below `N` | `O(S)` |
| Outcome sampling | `O(T * L)` | `O(S)` |

Average overall regret is at most `Delta |I| sqrt(A) / sqrt(T)`, so exploitability of the average strategy shrinks as `O(1/sqrt(T))` for vanilla CFR. CFR+ is empirically much faster.

## Usage

```python
from counterfactual_regret import (CFRSolver, LeducHoldem, build_tree,
                                   exploitability, expected_value)

tree = build_tree(LeducHoldem())
solver = CFRSolver(tree, "plus").run(500)
avg = solver.average_strategy
print(exploitability(tree, avg), expected_value(tree, avg))
print(tree.strategy_dict(avg)["1|-|"])     # queen, pre-flop, first to act
```

Run the demo with `python examples/demo.py` and the tests with `python -m pytest -q`.

## Results

Numbers from `examples/demo.py` (exploitability of the average strategy, in chips with an ante of 1):

Kuhn poker (value -0.05556 = -1/18):

| Variant | 100 it. | 1000 it. | 5000 it. |
|---|---|---|---|
| vanilla | 2.57e-02 | 7.27e-03 | 3.30e-03 |
| CFR+ | 1.19e-03 | 8.74e-05 | 2.77e-05 |
| linear | 2.45e-02 | 8.42e-03 | 3.72e-03 |
| DCFR | 2.05e-02 | 5.03e-03 | 1.24e-03 |

Leduc hold'em (9,457 nodes, 288 information sets):

| Variant | 50 it. | 200 it. | 500 it. | seconds |
|---|---|---|---|---|
| vanilla | 2.81e-01 | 9.80e-02 | 5.58e-02 | 0.52 |
| CFR+ | 3.41e-02 | 5.07e-03 | 9.26e-04 | 0.92 |
| linear | 3.23e-01 | 1.11e-01 | 6.82e-02 | 0.53 |
| DCFR | 1.29e-01 | 3.72e-02 | 1.38e-02 | 0.46 |

CFR+ reaches a Leduc game value of -0.0856, matching the value reported in the literature. Monte-Carlo CFR on Leduc (seed 0): external sampling 0.257 after 20k iterations and 0.214 after 100k; outcome sampling 0.477 after 100k. Sampling is cheap per iteration but converges much more slowly than full traversal on a game this small.

## Layout

```
counterfactual_regret/
    games.py       Kuhn poker and Leduc hold'em
    tree.py        game-tree compiler
    cfr.py         vectorised CFR variants
    mccfr.py       external and outcome sampling
    evaluation.py  expected value, best response, exploitability
    analysis.py    analytic Kuhn equilibria and helpers
tests/             28 deterministic pytest tests
examples/demo.py   end-to-end demo
```

## References

* M. Zinkevich, M. Johanson, M. Bowling, C. Piccione. *Regret Minimization in Games with Incomplete Information.* NIPS 2007.
* O. Tammelin, N. Burch, M. Johanson, M. Bowling. *Solving Heads-Up Limit Texas Hold'em.* IJCAI 2015 (CFR+).
* M. Lanctot, K. Waugh, M. Zinkevich, M. Bowling. *Monte Carlo Sampling for Regret Minimization in Extensive Games.* NIPS 2009.
* N. Brown, T. Sandholm. *Solving Imperfect-Information Games via Discounted Regret Minimization.* AAAI 2019.
* H. W. Kuhn. *A Simplified Two-Person Poker.* Contributions to the Theory of Games, 1950.
* F. Southey et al. *Bayes' Bluff: Opponent Modelling in Poker.* UAI 2005 (Leduc hold'em).
* T. Neller, M. Lanctot. *An Introduction to Counterfactual Regret Minimization.* 2013.

---

**Author:** Dr. Kainat Rizwan
