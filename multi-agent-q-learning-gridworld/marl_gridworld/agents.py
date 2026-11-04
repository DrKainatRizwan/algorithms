"""Tabular learners: independent (optionally hysteretic) and joint-action."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np


@dataclass
class EpsilonSchedule:
    """Linear decay of exploration rate from ``start`` to ``end``."""

    start: float = 1.0
    end: float = 0.05
    steps: int = 3000

    def __call__(self, episode: int) -> float:
        frac = min(1.0, episode / max(1, self.steps))
        return self.start + frac * (self.end - self.start)


class IndependentQLearner:
    """Q-learning agent that treats teammates as part of the environment.

    With ``beta < alpha`` the update is hysteretic: negative TD errors are
    applied with the smaller rate ``beta``, making the agent optimistic about
    teammates' exploratory mistakes (Matignon et al., 2007).
    """

    def __init__(self, n_actions: int, alpha: float = 0.2, gamma: float = 0.97,
                 beta: float | None = None, seed: int = 0) -> None:
        self.n_actions = n_actions
        self.alpha = alpha
        self.beta = alpha if beta is None else beta
        self.gamma = gamma
        self.q: dict[tuple, np.ndarray] = {}
        self.rng = np.random.default_rng(seed)

    def values(self, obs: tuple) -> np.ndarray:
        if obs not in self.q:
            self.q[obs] = np.zeros(self.n_actions)
        return self.q[obs]

    def greedy(self, obs: tuple) -> int:
        v = self.values(obs)
        best = np.flatnonzero(v == v.max())
        return int(best[0]) if len(best) == 1 else int(self.rng.choice(best))

    def act(self, obs: tuple, eps: float) -> int:
        if self.rng.random() < eps:
            return int(self.rng.integers(self.n_actions))
        return self.greedy(obs)

    def update(self, obs: tuple, a: int, r: float, nobs: tuple, done: bool) -> float:
        target = r if done else r + self.gamma * self.values(nobs).max()
        td = target - self.values(obs)[a]
        self.values(obs)[a] += (self.alpha if td >= 0 else self.beta) * td
        return float(td)


class JointQLearner:
    """Centralised learner over the product of all agents' actions.

    Used as an upper-bound baseline; its table grows as ``n_actions ** n``.
    """

    def __init__(self, n_agents: int, n_actions: int, alpha: float = 0.2,
                 gamma: float = 0.97, seed: int = 0) -> None:
        self.n_agents, self.n_actions = n_agents, n_actions
        self.joint = list(product(range(n_actions), repeat=n_agents))
        self.alpha, self.gamma = alpha, gamma
        self.q: dict[tuple, np.ndarray] = {}
        self.rng = np.random.default_rng(seed)

    def values(self, obs: tuple) -> np.ndarray:
        if obs not in self.q:
            self.q[obs] = np.zeros(len(self.joint))
        return self.q[obs]

    def act(self, obs: tuple, eps: float) -> list[int]:
        if self.rng.random() < eps:
            k = int(self.rng.integers(len(self.joint)))
        else:
            v = self.values(obs)
            best = np.flatnonzero(v == v.max())
            k = int(self.rng.choice(best))
        return list(self.joint[k])

    def update(self, obs: tuple, actions: list[int], r: float, nobs: tuple,
               done: bool) -> float:
        k = self.joint.index(tuple(actions))
        target = r if done else r + self.gamma * self.values(nobs).max()
        td = target - self.values(obs)[k]
        self.values(obs)[k] += self.alpha * td
        return float(td)
