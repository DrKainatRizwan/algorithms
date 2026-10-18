"""Simulation loop and regret accounting."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .environments import AdversarialBandit, BanditEnvironment
from .policies import Policy


@dataclass
class RunResult:
    arms: np.ndarray
    rewards: np.ndarray
    regret: np.ndarray  # cumulative pseudo-regret per round

    @property
    def final_regret(self) -> float:
        return float(self.regret[-1])


def run(policy: Policy, env: BanditEnvironment, horizon: int) -> RunResult:
    """Play ``horizon`` rounds; regret is measured against the per-round best mean."""
    arms = np.empty(horizon, dtype=np.int64)
    rewards = np.empty(horizon)
    inst = np.empty(horizon)
    for i in range(horizon):
        mu = env.means()
        a = policy.select()
        r = env.pull(a)
        policy.update(a, r)
        arms[i], rewards[i] = a, r
        inst[i] = mu.max() - mu[a]
    return RunResult(arms, rewards, np.cumsum(inst))


def run_adversarial(policy: Policy, env: AdversarialBandit, horizon: int | None = None) -> RunResult:
    """Regret against the best fixed arm in hindsight (can be negative)."""
    horizon = horizon or env.horizon
    arms = np.empty(horizon, dtype=np.int64)
    rewards = np.empty(horizon)
    for i in range(horizon):
        a = policy.select()
        r = env.pull(a)
        policy.update(a, r)
        arms[i], rewards[i] = a, r
    best_arm = int(np.argmax(env.table[:horizon].sum(axis=0)))
    regret = np.cumsum(env.table[:horizon, best_arm]) - np.cumsum(rewards)
    return RunResult(arms, rewards, regret)


def average_regret(
    make_policy: Callable[[int], Policy],
    make_env: Callable[[int], BanditEnvironment],
    horizon: int,
    n_runs: int,
    seed: int = 0,
) -> np.ndarray:
    """Mean cumulative regret curve over independent runs."""
    acc = np.zeros(horizon)
    for k in range(n_runs):
        acc += run(make_policy(seed + k), make_env(10_000 + seed + k), horizon).regret
    return acc / n_runs


def best_arm_rate(result: RunResult, best_arm: int, last: int | None = None) -> float:
    """Fraction of plays (optionally only the last ``last``) on ``best_arm``."""
    a = result.arms if last is None else result.arms[-last:]
    return float(np.mean(a == best_arm))
