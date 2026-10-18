"""Bandit policies: epsilon-greedy, softmax, UCB1, UCB-Tuned, Thompson sampling, EXP3."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Policy(ABC):
    """Interface: ``select()`` returns an arm, ``update(arm, reward)`` feeds back."""

    name = "policy"

    def __init__(self, n_arms: int, seed: int = 0) -> None:
        self.n_arms = n_arms
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> None:
        self.t = 0
        self.counts = np.zeros(self.n_arms, dtype=np.int64)
        self.sums = np.zeros(self.n_arms)
        self.sq_sums = np.zeros(self.n_arms)

    @property
    def estimates(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.counts > 0, self.sums / np.maximum(self.counts, 1), 0.0)

    @abstractmethod
    def select(self) -> int: ...

    def update(self, arm: int, reward: float) -> None:
        self.t += 1
        self.counts[arm] += 1
        self.sums[arm] += reward
        self.sq_sums[arm] += reward * reward

    def _untried(self) -> int | None:
        idx = np.flatnonzero(self.counts == 0)
        return int(idx[0]) if idx.size else None

    def _argmax_random_tie(self, values: np.ndarray) -> int:
        best = np.flatnonzero(values >= values.max() - 1e-12)
        return int(self.rng.choice(best))


class EpsilonGreedy(Policy):
    """Explore uniformly with probability epsilon (optionally decayed as c/t)."""

    def __init__(self, n_arms: int, epsilon: float = 0.1, decay: bool = False, seed: int = 0) -> None:
        self.epsilon = epsilon
        self.decay = decay
        self.name = f"eps-greedy({'decay' if decay else epsilon})"
        super().__init__(n_arms, seed)

    def select(self) -> int:
        u = self._untried()
        if u is not None:
            return u
        eps = min(1.0, self.epsilon * self.n_arms / (self.t + 1)) if self.decay else self.epsilon
        if self.rng.random() < eps:
            return int(self.rng.integers(self.n_arms))
        return self._argmax_random_tie(self.estimates)


class Softmax(Policy):
    """Boltzmann exploration with fixed temperature."""

    def __init__(self, n_arms: int, temperature: float = 0.1, seed: int = 0) -> None:
        self.temperature = temperature
        self.name = f"softmax({temperature})"
        super().__init__(n_arms, seed)

    def probabilities(self) -> np.ndarray:
        z = self.estimates / self.temperature
        z -= z.max()
        p = np.exp(z)
        return p / p.sum()

    def select(self) -> int:
        return int(self.rng.choice(self.n_arms, p=self.probabilities()))


class UCB1(Policy):
    """Auer et al. (2002): mean + sqrt(c ln t / n)."""

    def __init__(self, n_arms: int, c: float = 2.0, seed: int = 0) -> None:
        self.c = c
        self.name = "UCB1"
        super().__init__(n_arms, seed)

    def index(self) -> np.ndarray:
        return self.estimates + np.sqrt(self.c * np.log(max(self.t, 1)) / np.maximum(self.counts, 1))

    def select(self) -> int:
        u = self._untried()
        return u if u is not None else self._argmax_random_tie(self.index())


class UCBTuned(UCB1):
    """UCB1 with a variance-aware confidence width (Auer et al., 2002)."""

    def __init__(self, n_arms: int, seed: int = 0) -> None:
        super().__init__(n_arms, seed=seed)
        self.name = "UCB-Tuned"

    def index(self) -> np.ndarray:
        n = np.maximum(self.counts, 1)
        lt = np.log(max(self.t, 1))
        var = self.sq_sums / n - self.estimates**2 + np.sqrt(2 * lt / n)
        return self.estimates + np.sqrt(lt / n * np.minimum(0.25, var))


class ThompsonBernoulli(Policy):
    """Beta-Bernoulli Thompson sampling; rewards in [0,1] are binarised stochastically."""

    def __init__(self, n_arms: int, alpha0: float = 1.0, beta0: float = 1.0, seed: int = 0) -> None:
        self.alpha0, self.beta0 = alpha0, beta0
        self.name = "Thompson(Beta)"
        super().__init__(n_arms, seed)

    def reset(self) -> None:
        super().reset()
        self.alpha = np.full(self.n_arms, self.alpha0)
        self.beta = np.full(self.n_arms, self.beta0)

    def select(self) -> int:
        return int(np.argmax(self.rng.beta(self.alpha, self.beta)))

    def update(self, arm: int, reward: float) -> None:
        super().update(arm, reward)
        r = 1.0 if self.rng.random() < reward else 0.0
        self.alpha[arm] += r
        self.beta[arm] += 1.0 - r


class ThompsonGaussian(Policy):
    """Gaussian Thompson sampling with known noise scale and a N(0.5, 1) prior."""

    def __init__(self, n_arms: int, sigma: float = 0.25, seed: int = 0) -> None:
        self.sigma = sigma
        self.name = "Thompson(Gauss)"
        super().__init__(n_arms, seed)

    def select(self) -> int:
        prec = 1.0 + self.counts / self.sigma**2
        mean = (0.5 + self.sums / self.sigma**2) / prec
        return int(np.argmax(self.rng.normal(mean, 1.0 / np.sqrt(prec))))


class EXP3(Policy):
    """Exponential weights for adversarial bandits (Auer et al., 2002b)."""

    def __init__(self, n_arms: int, gamma: float | None = None, horizon: int | None = None, seed: int = 0) -> None:
        if gamma is None:
            if horizon is None:
                gamma = 0.1
            else:
                k = n_arms
                gamma = min(1.0, float(np.sqrt(k * np.log(k) / ((np.e - 1) * horizon))))
        self.gamma = gamma
        self.name = "EXP3"
        super().__init__(n_arms, seed)

    def reset(self) -> None:
        super().reset()
        self.log_w = np.zeros(self.n_arms)
        self._last_p = np.full(self.n_arms, 1.0 / self.n_arms)

    def probabilities(self) -> np.ndarray:
        w = np.exp(self.log_w - self.log_w.max())
        return (1 - self.gamma) * w / w.sum() + self.gamma / self.n_arms

    def select(self) -> int:
        self._last_p = self.probabilities()
        return int(self.rng.choice(self.n_arms, p=self._last_p))

    def update(self, arm: int, reward: float) -> None:
        super().update(arm, reward)
        xhat = reward / self._last_p[arm]
        self.log_w[arm] += self.gamma * xhat / self.n_arms


class SlidingWindowUCB(Policy):
    """UCB over the last ``window`` rounds, for non-stationary environments."""

    def __init__(self, n_arms: int, window: int = 500, c: float = 2.0, seed: int = 0) -> None:
        self.window, self.c = window, c
        self.name = f"SW-UCB({window})"
        super().__init__(n_arms, seed)

    def reset(self) -> None:
        super().reset()
        self.history: list[tuple[int, float]] = []
        self.w_counts = np.zeros(self.n_arms)
        self.w_sums = np.zeros(self.n_arms)

    def select(self) -> int:
        u = np.flatnonzero(self.w_counts == 0)
        if u.size:
            return int(u[0])
        horizon = min(self.t, self.window)
        idx = self.w_sums / self.w_counts + np.sqrt(self.c * np.log(max(horizon, 1)) / self.w_counts)
        return self._argmax_random_tie(idx)

    def update(self, arm: int, reward: float) -> None:
        super().update(arm, reward)
        self.history.append((arm, reward))
        self.w_counts[arm] += 1
        self.w_sums[arm] += reward
        if len(self.history) > self.window:
            old_arm, old_r = self.history.pop(0)
            self.w_counts[old_arm] -= 1
            self.w_sums[old_arm] -= old_r
