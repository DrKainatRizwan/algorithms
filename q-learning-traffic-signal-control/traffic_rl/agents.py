"""Tabular temporal-difference agents: Q-learning, Double Q-learning, Expected SARSA."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ALGORITHMS = ("q", "double_q", "expected_sarsa")


@dataclass(frozen=True)
class StateEncoder:
    """Discretises the local observation into a single integer state index."""

    edges: tuple[int, ...] = (1, 4, 8, 14, 22)

    @property
    def n_bins(self) -> int:
        return len(self.edges) + 1

    @property
    def n_states(self) -> int:
        return 2 * 2 * self.n_bins * self.n_bins

    def bin(self, queue: int) -> int:
        return int(np.searchsorted(self.edges, queue, side="right"))

    def encode(self, obs: tuple[int, int, int, int]) -> int:
        phase, ok, ns, ew = obs
        s = phase
        s = s * 2 + ok
        s = s * self.n_bins + self.bin(ns)
        return s * self.n_bins + self.bin(ew)

    def decode(self, s: int) -> tuple[int, int, int, int]:
        """Inverse of :meth:`encode` up to the queue bin (returns bin indices)."""
        s, ew = divmod(s, self.n_bins)
        s, ns = divmod(s, self.n_bins)
        phase, ok = divmod(s, 2)
        return phase, ok, ns, ew


class EpsilonSchedule:
    """Exponentially decaying exploration rate with a floor."""

    def __init__(self, start: float = 1.0, end: float = 0.05, decay: float = 0.97) -> None:
        self.start, self.end, self.decay = start, end, decay

    def value(self, episode: int) -> float:
        return max(self.end, self.start * self.decay ** episode)


class TabularAgent:
    """Tabular value-based agent over a discrete state space and two actions."""

    def __init__(self, n_states: int, n_actions: int = 2, algo: str = "q", alpha: float = 0.1,
                 gamma: float = 0.9, epsilon: float = 0.1, seed: int = 0,
                 optimistic_init: float = 0.0) -> None:
        if algo not in ALGORITHMS:
            raise ValueError(f"unknown algorithm {algo!r}")
        self.algo, self.alpha, self.gamma, self.epsilon = algo, alpha, gamma, epsilon
        self.n_states, self.n_actions = n_states, n_actions
        self.rng = np.random.default_rng(seed)
        self.qa = np.full((n_states, n_actions), optimistic_init, dtype=float)
        self.qb = self.qa.copy() if algo == "double_q" else None
        self.visits = np.zeros((n_states, n_actions), dtype=np.int64)

    @property
    def q(self) -> np.ndarray:
        """Value estimate used for acting (average of both tables for Double Q)."""
        return self.qa if self.qb is None else 0.5 * (self.qa + self.qb)

    def greedy(self, s: int) -> int:
        row = self.q[s]
        best = np.flatnonzero(row == row.max())
        return int(best[0]) if len(best) == 1 else int(self.rng.choice(best))

    def act(self, s: int, explore: bool = True) -> int:
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return self.greedy(s)

    def _expected_next(self, s2: int) -> float:
        row = self.qa[s2]
        eps = self.epsilon
        probs = np.full(self.n_actions, eps / self.n_actions)
        probs[int(np.argmax(row))] += 1.0 - eps
        return float(probs @ row)

    def update(self, s: int, a: int, r: float, s2: int, done: bool = False) -> float:
        """One TD update; returns the TD error that was applied."""
        self.visits[s, a] += 1
        if self.algo == "double_q":
            first = self.rng.random() < 0.5
            a_tab, b_tab = (self.qa, self.qb) if first else (self.qb, self.qa)
            target = r if done else r + self.gamma * b_tab[s2, int(np.argmax(a_tab[s2]))]
            err = target - a_tab[s, a]
            a_tab[s, a] += self.alpha * err
            return float(err)
        if done:
            nxt = 0.0
        elif self.algo == "q":
            nxt = float(self.qa[s2].max())
        else:
            nxt = self._expected_next(s2)
        err = r + self.gamma * nxt - self.qa[s, a]
        self.qa[s, a] += self.alpha * err
        return float(err)

    def policy_table(self) -> np.ndarray:
        """Greedy action per state (ties resolved to action 0, i.e. keep)."""
        return np.argmax(self.q, axis=1)
