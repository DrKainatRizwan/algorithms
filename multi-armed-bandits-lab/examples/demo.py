"""Compare bandit policies on stochastic, switching and adversarial environments."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from bandit_lab import (EXP3, AdversarialBandit, BernoulliBandit, EpsilonGreedy,
                        SlidingWindowUCB, SwitchingBandit, ThompsonBernoulli, UCB1,
                        UCBTuned, alternating_adversary, average_regret, exp3_regret_bound,
                        lai_robbins_constant, log_fit_slope, run_adversarial,
                        ucb1_regret_bound)

PROBS = [0.10, 0.25, 0.40, 0.45, 0.55]
T, RUNS = 5000, 10

print(f"== Stochastic Bernoulli bandit {PROBS}, T={T}, {RUNS} runs ==")
factories = {
    "eps-greedy(0.1)": lambda s: EpsilonGreedy(5, 0.1, seed=s),
    "eps-greedy(decay)": lambda s: EpsilonGreedy(5, 0.5, decay=True, seed=s),
    "UCB1": lambda s: UCB1(5, seed=s),
    "UCB-Tuned": lambda s: UCBTuned(5, seed=s),
    "Thompson": lambda s: ThompsonBernoulli(5, seed=s),
}
for name, f in factories.items():
    curve = average_regret(f, lambda s: BernoulliBandit(PROBS, seed=s), T, RUNS)
    print(f"{name:18s} regret={curve[-1]:8.1f}  log-slope={log_fit_slope(curve):6.1f}")
print(f"Lai-Robbins constant: {lai_robbins_constant(PROBS):.1f} (x ln T = {lai_robbins_constant(PROBS)*np.log(T):.1f})")
print(f"UCB1 upper bound: {ucb1_regret_bound(PROBS, T):.0f}")

print("\n== Switching environment (means flip at t=3000), T=6000 ==")
sched = [[0.8, 0.5], [0.2, 0.5]]
mk = lambda s: SwitchingBandit(sched, period=3000, seed=s)
for name, f in {"UCB1": lambda s: UCB1(2, seed=s),
                "SW-UCB(400)": lambda s: SlidingWindowUCB(2, window=400, c=0.5, seed=s)}.items():
    print(f"{name:18s} regret={average_regret(f, mk, 6000, 5)[-1]:8.1f}")

print("\n== Adversarial reward table, T=3000 ==")
table = alternating_adversary(3000)
for name, p in {"EXP3": EXP3(2, horizon=3000, seed=0), "UCB1": UCB1(2, seed=0)}.items():
    res = run_adversarial(p, AdversarialBandit(table), 3000)
    print(f"{name:18s} regret vs best fixed arm={res.final_regret:8.1f}")
print(f"EXP3 worst-case bound: {exp3_regret_bound(2, 3000):.0f}")
