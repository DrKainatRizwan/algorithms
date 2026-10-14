"""Compare REINFORCE variants and A2C on CartPole."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from policy_gradient import (A2C, CartPole, Corridor, Reinforce, VectorEnv,
                             evaluate_policy, moving_average)


def main() -> None:
    t0 = time.time()
    print("== Corridor sanity check (optimal return %.2f) ==" % Corridor(7).optimal_return(0.99))
    env = Corridor(7)
    ag = Reinforce(7, 2, hidden=16, lr=0.02, baseline="mean", seed=0)
    ag.train(env, 150)
    print("REINFORCE eval return: %.3f" % evaluate_policy(ag, env, 3))

    print("\n== CartPole-200: REINFORCE baselines (250 episodes, 2 seeds) ==")
    for base in ("none", "mean", "value"):
        finals, evals = [], []
        for seed in range(2):
            env = CartPole(seed=seed)
            agent = Reinforce(4, 2, hidden=32, lr=0.01, baseline=base, seed=seed)
            hist = agent.train(env, 250)
            finals.append(np.mean(hist[-30:]))
            evals.append(evaluate_policy(agent, CartPole(), 5))
        print(f"baseline={base:5s} last-30 train return {np.mean(finals):6.1f}  greedy eval {np.mean(evals):6.1f}")

    print("\n== CartPole-200: A2C (8 envs, 16-step rollouts, 3 seeds) ==")
    finals, evals = [], []
    for seed in range(3):
        venv = VectorEnv(lambda s: CartPole(seed=s), 8, seed=seed)
        agent = A2C(4, 2, seed=seed)
        rets = agent.train(venv, 300, n_steps=16)
        ma = moving_average(rets, 20)
        finals.append(ma[-1])
        evals.append(evaluate_policy(agent, CartPole(), 5))
        print(f"seed {seed}: {len(rets)} episodes, final MA20 {ma[-1]:6.1f}, greedy eval {evals[-1]:6.1f}")
    print(f"A2C mean: train {np.mean(finals):.1f}, eval {np.mean(evals):.1f}")
    print(f"\nTotal time {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
