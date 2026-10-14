"""Small control environments with a gym-like interface, no external dependencies."""

from __future__ import annotations

import math
from typing import Callable

import numpy as np


class CartPole:
    """Classic cart-pole balancing task (Barto, Sutton & Anderson, 1983).

    Observation: ``[x, x_dot, theta, theta_dot]``. Actions: 0 = push left,
    1 = push right. Reward is +1 per step; the episode terminates when the pole
    falls past 12 degrees or the cart leaves +-2.4, and is truncated at
    ``max_steps``.
    """

    obs_dim = 4
    n_actions = 2

    def __init__(self, max_steps: int = 200, seed: int = 0) -> None:
        self.gravity = 9.8
        self.mass_cart = 1.0
        self.mass_pole = 0.1
        self.length = 0.5  # half pole length
        self.force_mag = 10.0
        self.tau = 0.02
        self.theta_limit = 12 * 2 * math.pi / 360
        self.x_limit = 2.4
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.state = np.zeros(4)
        self.t = 0

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = self.rng.uniform(-0.05, 0.05, size=4)
        self.t = 0
        return self.state.copy()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool]:
        """Advance one step; returns ``(obs, reward, terminated, truncated)``."""
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        total_mass = self.mass_cart + self.mass_pole
        pm_len = self.mass_pole * self.length
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        temp = (force + pm_len * theta_dot ** 2 * sin_t) / total_mass
        theta_acc = (self.gravity * sin_t - cos_t * temp) / (
            self.length * (4.0 / 3.0 - self.mass_pole * cos_t ** 2 / total_mass))
        x_acc = temp - pm_len * theta_acc * cos_t / total_mass
        x += self.tau * x_dot
        x_dot += self.tau * x_acc
        theta += self.tau * theta_dot
        theta_dot += self.tau * theta_acc
        self.state = np.array([x, x_dot, theta, theta_dot])
        self.t += 1
        terminated = bool(abs(x) > self.x_limit or abs(theta) > self.theta_limit)
        truncated = (not terminated) and self.t >= self.max_steps
        return self.state.copy(), 1.0, terminated, truncated


class Corridor:
    """Deterministic 1-D corridor: start in the middle, reward +1 at the right end.

    Observation is a one-hot vector of the position. Actions: 0 = left, 1 = right.
    Each step costs -0.01; falling off the left end gives -1. Useful as a tiny
    sanity-check task whose optimal return is known analytically.
    """

    n_actions = 2

    def __init__(self, length: int = 7, max_steps: int = 50, seed: int = 0) -> None:
        self.length = length
        self.obs_dim = length
        self.max_steps = max_steps
        self.pos = length // 2
        self.t = 0

    def _obs(self) -> np.ndarray:
        o = np.zeros(self.length)
        o[self.pos] = 1.0
        return o

    def reset(self, seed: int | None = None) -> np.ndarray:
        self.pos = self.length // 2
        self.t = 0
        return self._obs()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool]:
        self.pos += 1 if action == 1 else -1
        self.t += 1
        if self.pos >= self.length - 1:
            self.pos = self.length - 1
            return self._obs(), 1.0, True, False
        if self.pos < 0:
            self.pos = 0
            return self._obs(), -1.0, True, False
        return self._obs(), -0.01, False, self.t >= self.max_steps

    def optimal_return(self, gamma: float = 1.0) -> float:
        steps = self.length - 1 - self.length // 2
        return sum(-0.01 * gamma ** k for k in range(steps - 1)) + gamma ** (steps - 1)


class VectorEnv:
    """Run several independent copies of an environment with auto-reset.

    ``step`` returns ``(obs, rewards, terminated, truncated, final_obs)`` where
    ``obs`` already holds the reset observation for finished sub-environments and
    ``final_obs[i]`` holds the last real observation (needed to bootstrap on
    time-limit truncation).
    """

    def __init__(self, make_env: Callable[[int], object], n_envs: int, seed: int = 0) -> None:
        self.envs = [make_env(seed + i) for i in range(n_envs)]
        self.n_envs = n_envs
        self.obs_dim = self.envs[0].obs_dim
        self.n_actions = self.envs[0].n_actions
        self.episode_returns = np.zeros(n_envs)
        self.finished_returns: list[float] = []

    def reset(self) -> np.ndarray:
        self.episode_returns[:] = 0
        return np.stack([e.reset() for e in self.envs])

    def step(self, actions: np.ndarray):
        obs, final = [], []
        rew = np.zeros(self.n_envs)
        term = np.zeros(self.n_envs, dtype=bool)
        trunc = np.zeros(self.n_envs, dtype=bool)
        for i, (env, a) in enumerate(zip(self.envs, actions)):
            o, r, te, tr = env.step(int(a))
            rew[i], term[i], trunc[i] = r, te, tr
            final.append(o)
            self.episode_returns[i] += r
            if te or tr:
                self.finished_returns.append(float(self.episode_returns[i]))
                self.episode_returns[i] = 0.0
                o = env.reset()
            obs.append(o)
        return np.stack(obs), rew, term, trunc, np.stack(final)
