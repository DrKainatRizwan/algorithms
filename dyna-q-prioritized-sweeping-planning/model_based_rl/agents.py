"""Tabular agents: Q-learning, Dyna-Q, Dyna-Q+ and prioritized sweeping."""

from __future__ import annotations

import heapq
import itertools

import numpy as np

from .model import TabularModel


class QLearning:
    """One-step tabular Q-learning with epsilon-greedy exploration."""

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.1,
        seed: int = 0,
    ) -> None:
        self.nS, self.nA = n_states, n_actions
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.Q = np.zeros((n_states, n_actions))
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.planning_updates = 0

    def act(self, s: int) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.nA))
        return self.greedy(s)

    def greedy(self, s: int) -> int:
        """Greedy action with random tie-breaking."""
        q = self.Q[s]
        best = np.flatnonzero(q == q.max())
        return int(best[self.rng.integers(len(best))])

    def policy(self) -> np.ndarray:
        return self.Q.argmax(axis=1)

    def _backup(self, s: int, a: int, r: float, s2: int, done: bool) -> float:
        target = r if done else r + self.gamma * self.Q[s2].max()
        delta = target - self.Q[s, a]
        self.Q[s, a] += self.alpha * delta
        return delta

    def observe(self, s: int, a: int, r: float, s2: int, done: bool) -> None:
        """Process a real transition."""
        self.t += 1
        self._backup(s, a, r, s2, done)


class DynaQ(QLearning):
    """Dyna-Q: direct RL plus ``n_planning`` simulated updates per step."""

    def __init__(self, *args, n_planning: int = 10, model_mode: str = "last", **kw) -> None:
        super().__init__(*args, **kw)
        self.n_planning = n_planning
        self.model = TabularModel(model_mode)

    def _learn_model(self, s: int, a: int, r: float, s2: int, done: bool) -> None:
        self.model.update(s, a, r, s2, done)

    def observe(self, s: int, a: int, r: float, s2: int, done: bool) -> None:
        self.t += 1
        self._backup(s, a, r, s2, done)
        self._learn_model(s, a, r, s2, done)
        self._plan()

    def _plan(self) -> None:
        keys = self.model.keys
        for _ in range(self.n_planning):
            s, a = keys[int(self.rng.integers(len(keys)))]
            r, s2, done = self.model.sample(s, a, self.rng)
            self._backup(s, a, self._planning_reward(s, a, r), s2, done)
            self.planning_updates += 1

    def _planning_reward(self, s: int, a: int, r: float) -> float:
        return r


class DynaQPlus(DynaQ):
    """Dyna-Q+: adds an exploration bonus ``kappa * sqrt(tau)`` in planning.

    ``tau`` is the number of real steps since ``(s, a)`` was last tried.
    Untried actions of visited states enter the model as self-loops with
    zero reward so that planning can discover that they have changed.
    """

    def __init__(self, *args, kappa: float = 1e-3, **kw) -> None:
        super().__init__(*args, **kw)
        self.kappa = kappa
        self.last_tried = np.zeros((self.nS, self.nA), dtype=np.int64)
        self._visited: set[int] = set()

    def _learn_model(self, s: int, a: int, r: float, s2: int, done: bool) -> None:
        if s not in self._visited:
            self._visited.add(s)
            for b in range(self.nA):
                if (s, b) not in self.model:
                    self.model.update(s, b, 0.0, s, False)
        self.model.update(s, a, r, s2, done)
        self.last_tried[s, a] = self.t

    def _planning_reward(self, s: int, a: int, r: float) -> float:
        return r + self.kappa * np.sqrt(self.t - self.last_tried[s, a])


class PrioritizedSweeping(QLearning):
    """Prioritized sweeping (Moore & Atkeson, 1993).

    Keeps a max-priority queue of state-action pairs ordered by the size of
    their pending Bellman error; after each update the predecessors of the
    changed state are re-prioritised. Planning uses expected backups over
    the learned model.
    """

    def __init__(
        self, *args, n_planning: int = 10, theta: float = 1e-4, model_mode: str = "last", **kw
    ) -> None:
        super().__init__(*args, **kw)
        self.n_planning = n_planning
        self.theta = theta
        self.model = TabularModel(model_mode)
        self._pq: list[tuple[float, int, int, int]] = []
        self._tie = itertools.count()

    def _expected_error(self, s: int, a: int) -> float:
        total = 0.0
        for p, r, s2, done in self.model.distribution(s, a):
            target = r if done else r + self.gamma * self.Q[s2].max()
            total += p * target
        return total - self.Q[s, a]

    def _push(self, s: int, a: int) -> None:
        p = abs(self._expected_error(s, a))
        if p > self.theta:
            heapq.heappush(self._pq, (-p, next(self._tie), s, a))

    def observe(self, s: int, a: int, r: float, s2: int, done: bool) -> None:
        self.t += 1
        self.model.update(s, a, r, s2, done)
        self._push(s, a)
        for _ in range(self.n_planning):
            if not self._pq:
                break
            _, _, ps, pa = heapq.heappop(self._pq)
            # Full expected backup of the popped pair.
            err = self._expected_error(ps, pa)
            self.Q[ps, pa] += err
            self.planning_updates += 1
            for qs, qa in list(self.model.predecessors[ps]):
                self._push(qs, qa)
