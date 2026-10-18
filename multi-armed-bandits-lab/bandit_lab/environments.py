"""Bandit environments: stochastic, non-stationary and adversarial."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BanditEnvironment(ABC):
    """Base class. Rewards are expected to lie in [0, 1] for all environments."""

    def __init__(self, n_arms: int, seed: int = 0) -> None:
        if n_arms < 2:
            raise ValueError("need at least two arms")
        self.n_arms = n_arms
        self.rng = np.random.default_rng(seed)
        self.t = 0

    @abstractmethod
    def means(self) -> np.ndarray:
        """Expected reward of every arm at the current round."""

    @abstractmethod
    def _draw(self, arm: int) -> float:
        """Sample a reward for ``arm`` at the current round."""

    def pull(self, arm: int) -> float:
        if not 0 <= arm < self.n_arms:
            raise IndexError(f"arm {arm} out of range")
        r = self._draw(arm)
        self.t += 1
        return r

    def best_mean(self) -> float:
        return float(np.max(self.means()))


class BernoulliBandit(BanditEnvironment):
    """Arms pay 1 with probability ``probs[i]`` and 0 otherwise."""

    def __init__(self, probs, seed: int = 0) -> None:
        p = np.asarray(probs, dtype=float)
        if np.any(p < 0) or np.any(p > 1):
            raise ValueError("probabilities must lie in [0, 1]")
        super().__init__(len(p), seed)
        self.probs = p

    def means(self) -> np.ndarray:
        return self.probs

    def _draw(self, arm: int) -> float:
        return float(self.rng.random() < self.probs[arm])


class GaussianBandit(BanditEnvironment):
    """Gaussian rewards clipped to [0, 1] (the clipping bias is ignored for small sigma)."""

    def __init__(self, mus, sigma: float = 0.1, seed: int = 0) -> None:
        m = np.asarray(mus, dtype=float)
        super().__init__(len(m), seed)
        self.mus = m
        self.sigma = float(sigma)

    def means(self) -> np.ndarray:
        return self.mus

    def _draw(self, arm: int) -> float:
        return float(np.clip(self.rng.normal(self.mus[arm], self.sigma), 0.0, 1.0))


class SwitchingBandit(BanditEnvironment):
    """Bernoulli bandit whose probability vector changes every ``period`` rounds."""

    def __init__(self, prob_schedule, period: int, seed: int = 0) -> None:
        sched = np.asarray(prob_schedule, dtype=float)
        if sched.ndim != 2:
            raise ValueError("schedule must be 2-D (phases x arms)")
        super().__init__(sched.shape[1], seed)
        self.schedule = sched
        self.period = int(period)

    def means(self) -> np.ndarray:
        return self.schedule[(self.t // self.period) % len(self.schedule)]

    def _draw(self, arm: int) -> float:
        return float(self.rng.random() < self.means()[arm])


class AdversarialBandit(BanditEnvironment):
    """Oblivious adversary: a fixed reward table ``rewards[t, arm]`` in [0, 1].

    ``means`` returns the current row, so per-round "regret" is relative to the
    best arm of that round; use :meth:`best_fixed_total` for the usual
    best-fixed-arm-in-hindsight benchmark.
    """

    def __init__(self, rewards, seed: int = 0) -> None:
        r = np.asarray(rewards, dtype=float)
        if r.ndim != 2 or r.min() < 0 or r.max() > 1:
            raise ValueError("rewards must be a (T, K) array in [0, 1]")
        super().__init__(r.shape[1], seed)
        self.table = r

    @property
    def horizon(self) -> int:
        return self.table.shape[0]

    def means(self) -> np.ndarray:
        return self.table[min(self.t, self.horizon - 1)]

    def _draw(self, arm: int) -> float:
        if self.t >= self.horizon:
            raise RuntimeError("adversarial sequence exhausted")
        return float(self.table[self.t, arm])

    def best_fixed_total(self, upto: int | None = None) -> float:
        tab = self.table if upto is None else self.table[:upto]
        return float(tab.sum(axis=0).max())


def alternating_adversary(horizon: int, n_arms: int = 2, gap: float = 0.3) -> np.ndarray:
    """Deterministic reward table that punishes greedy follow-the-leader play.

    Arm 0 pays slightly more on average but its payoff is front-loaded and then
    alternates with the other arms, so empirical-mean policies are misled.
    """
    tab = np.full((horizon, n_arms), 0.5 - gap / 2)
    for t in range(horizon):
        tab[t, t % n_arms] = 0.5 + gap / 2
    tab[: horizon // 10, 0] = 0.9
    tab[: horizon // 10, 1:] = 0.1
    return tab
