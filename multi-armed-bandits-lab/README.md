# Multi-Armed Bandits Lab: Epsilon-Greedy, UCB1, Thompson Sampling and EXP3

## Overview
A small laboratory for the exploration/exploitation trade-off, written from scratch with NumPy. It contains stochastic, non-stationary and adversarial environments, eight policies, a regret-accounting simulator and helpers for the classical theoretical bounds.

| Module | Contents |
|---|---|
| `bandit_lab/environments.py` | Bernoulli, clipped Gaussian, switching (piecewise-stationary) and oblivious-adversarial bandits |
| `bandit_lab/policies.py` | Epsilon-greedy (fixed/decaying), softmax, UCB1, UCB-Tuned, Beta and Gaussian Thompson sampling, EXP3, sliding-window UCB |
| `bandit_lab/simulation.py` | Run loops, pseudo-regret and best-fixed-arm regret, averaging over seeds |
| `bandit_lab/analysis.py` | Bernoulli KL, Lai-Robbins constant, UCB1 and EXP3 bounds, log-regret fit |

## How it works
* **Epsilon-greedy** explores uniformly with probability epsilon, otherwise plays the empirical best arm. A fixed epsilon gives linear regret; decaying it as `K/t` gives logarithmic regret.
* **UCB1** plays `argmax mean_i + sqrt(c ln t / n_i)`. **UCB-Tuned** replaces the width with a variance-aware term.
* **Thompson sampling** keeps a Beta(α, β) posterior per arm, samples one value from each and plays the maximum.
* **EXP3** keeps exponential weights, mixes with uniform exploration `γ` and uses importance-weighted reward estimates `r / p_i`, giving `O(sqrt(T K ln K))` regret against any oblivious adversary.
* **Sliding-window UCB** computes UCB indices from the last `W` rounds only, so it can track changing means.

## Complexity
Per round: epsilon-greedy, UCB and Thompson are `O(K)` time; sliding-window UCB is `O(K)` amortised with `O(W)` memory; all others use `O(K)` memory. A full run of horizon `T` costs `O(TK)`.

## Usage
```python
from bandit_lab import BernoulliBandit, ThompsonBernoulli, run

env = BernoulliBandit([0.2, 0.4, 0.8], seed=1)
result = run(ThompsonBernoulli(3, seed=0), env, 2000)
print(result.final_regret, result.arms[-10:])
```
Run `python examples/demo.py` for the full comparison and `python -m pytest -q` for the tests.

## Results
Demo output (arms `[0.10, 0.25, 0.40, 0.45, 0.55]`, T = 5000, mean of 10 runs):

| Policy | Cumulative regret | Slope vs ln t |
|---|---|---|
| eps-greedy (0.1) | 163.8 | 50.6 |
| eps-greedy (decaying) | 110.5 | 32.7 |
| UCB1 (c = 2) | 190.3 | 55.4 |
| UCB-Tuned | 31.8 | 6.3 |
| Thompson (Beta) | 40.3 | 8.7 |

The Lai-Robbins lower-bound constant for this instance is 10.9 (≈ 92.9 at T = 5000, asymptotically) and the UCB1 theoretical upper bound is 1518. UCB1 with `c = 2` is known to be conservative, which is why variance-aware and Bayesian methods do much better.

Switching environment (means `[0.8, 0.5]` → `[0.2, 0.5]` at t = 3000, T = 6000): UCB1 regret 615.4 versus sliding-window UCB (W = 400) 134.4.

Adversarial alternating table (T = 3000): EXP3 regret against the best fixed arm 83.9 (worst-case bound 169); UCB1 reached 73.8 on this particular sequence.

## References
* Lai, T. L. & Robbins, H. (1985). Asymptotically efficient adaptive allocation rules. *Advances in Applied Mathematics*.
* Auer, P., Cesa-Bianchi, N. & Fischer, P. (2002). Finite-time analysis of the multiarmed bandit problem. *Machine Learning*.
* Auer, P., Cesa-Bianchi, N., Freund, Y. & Schapire, R. (2002). The nonstochastic multiarmed bandit problem. *SIAM J. Computing*.
* Thompson, W. R. (1933). On the likelihood that one unknown probability exceeds another. *Biometrika*.
* Garivier, A. & Moulines, E. (2011). On upper-confidence bound policies for switching bandit problems. *ALT*.
* Lattimore, T. & Szepesvári, C. (2020). *Bandit Algorithms*. Cambridge University Press.

---

**Author:** Dr. Kainat Rizwan
