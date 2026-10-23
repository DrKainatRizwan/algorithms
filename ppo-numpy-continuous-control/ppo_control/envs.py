"""Small continuous-control environments with a gym-like reset/step API."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


class Env:
    """Base class: ``reset() -> obs`` and ``step(a) -> (obs, reward, done, info)``."""

    obs_dim: int
    act_dim: int
    act_limit: float = 1.0
    max_steps: int = 100

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)
        self.t = 0

    def reset(self) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def step(self, action: np.ndarray):  # pragma: no cover - abstract
        raise NotImplementedError

    def _clip(self, action: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(action, dtype=float).reshape(self.act_dim),
                       -self.act_limit, self.act_limit)


class PointMassEnv(Env):
    """1-D double integrator: drive position to the origin and stop.

    State ``(x, v)``; dynamics ``v += dt*a``, ``x += dt*v``.
    Reward ``-(x^2 + 0.1 v^2 + 0.01 a^2)`` per step.
    """

    obs_dim, act_dim = 2, 1
    max_steps = 100

    def __init__(self, seed: int = 0, dt: float = 0.1) -> None:
        super().__init__(seed)
        self.dt = dt
        self.state = np.zeros(2)

    def reset(self) -> np.ndarray:
        self.state = np.array([self.rng.uniform(-1.0, 1.0), self.rng.uniform(-0.5, 0.5)])
        self.t = 0
        return self.state.copy()

    def step(self, action):
        a = float(self._clip(action)[0])
        x, v = self.state
        v = v + self.dt * a * 2.0
        x = x + self.dt * v
        self.state = np.array([x, v])
        self.t += 1
        reward = -(x * x + 0.1 * v * v + 0.01 * a * a)
        done = self.t >= self.max_steps
        return self.state.copy(), reward, done, {}


class ReachEnv2D(Env):
    """2-D point mass with velocity damping that must reach a random target.

    Observation ``(px, py, vx, vy, tx-px, ty-py)``; reward is negative distance
    to target plus a small control cost, with a bonus when inside the goal radius.
    """

    obs_dim, act_dim = 6, 2
    max_steps = 80

    def __init__(self, seed: int = 0, dt: float = 0.1, goal_radius: float = 0.1) -> None:
        super().__init__(seed)
        self.dt, self.goal_radius = dt, goal_radius
        self.pos = np.zeros(2)
        self.vel = np.zeros(2)
        self.target = np.zeros(2)

    def _obs(self) -> np.ndarray:
        return np.concatenate([self.pos, self.vel, self.target - self.pos])

    def reset(self) -> np.ndarray:
        self.pos = self.rng.uniform(-1, 1, 2)
        self.vel = np.zeros(2)
        self.target = self.rng.uniform(-1, 1, 2)
        self.t = 0
        return self._obs()

    def step(self, action):
        a = self._clip(action)
        self.vel = 0.9 * self.vel + self.dt * 3.0 * a
        self.pos = self.pos + self.dt * self.vel
        self.t += 1
        dist = float(np.linalg.norm(self.target - self.pos))
        reward = -dist - 0.01 * float(a @ a) + (0.5 if dist < self.goal_radius else 0.0)
        return self._obs(), reward, self.t >= self.max_steps, {"dist": dist}


class PendulumEnv(Env):
    """Torque-limited inverted pendulum (swing-up variant of the classic task).

    Observation ``(cos th, sin th, thdot)``; ``th = 0`` is upright. Use a small
    ``start_range`` for the easier balancing task.
    """

    obs_dim, act_dim = 3, 1
    max_steps = 200
    max_speed = 8.0
    max_torque = 2.0

    def __init__(self, seed: int = 0, dt: float = 0.05, g: float = 10.0,
                 start_range: float = np.pi) -> None:
        super().__init__(seed)
        self.dt, self.g, self.start_range = dt, g, start_range
        self.th = 0.0
        self.thdot = 0.0

    def _obs(self) -> np.ndarray:
        return np.array([np.cos(self.th), np.sin(self.th), self.thdot])

    def reset(self) -> np.ndarray:
        self.th = float(self.rng.uniform(-self.start_range, self.start_range))
        self.thdot = float(self.rng.uniform(-1, 1))
        self.t = 0
        return self._obs()

    def step(self, action):
        u = float(self._clip(action)[0]) * self.max_torque
        th_norm = ((self.th + np.pi) % (2 * np.pi)) - np.pi
        cost = th_norm ** 2 + 0.1 * self.thdot ** 2 + 0.001 * u ** 2
        self.thdot = float(np.clip(self.thdot + (3 * self.g / 2 * np.sin(self.th) + 3.0 * u) * self.dt,
                                   -self.max_speed, self.max_speed))
        self.th = self.th + self.thdot * self.dt
        self.t += 1
        return self._obs(), -cost, self.t >= self.max_steps, {}


class VecEnv:
    """Run several independent env copies in lock-step with auto-reset."""

    def __init__(self, factory: Callable[[int], Env], n_envs: int, seed: int = 0) -> None:
        self.envs: Sequence[Env] = [factory(seed + i) for i in range(n_envs)]
        self.n = n_envs
        self.obs_dim = self.envs[0].obs_dim
        self.act_dim = self.envs[0].act_dim
        self.act_limit = self.envs[0].act_limit
        self.ep_returns = np.zeros(n_envs)
        self.finished: list[float] = []

    def reset(self) -> np.ndarray:
        self.ep_returns[:] = 0
        return np.stack([e.reset() for e in self.envs])

    def step(self, actions: np.ndarray):
        obs, rews, dones = [], np.zeros(self.n), np.zeros(self.n, dtype=bool)
        for i, (e, a) in enumerate(zip(self.envs, actions)):
            o, r, d, _ = e.step(a)
            self.ep_returns[i] += r
            rews[i], dones[i] = r, d
            if d:
                self.finished.append(float(self.ep_returns[i]))
                self.ep_returns[i] = 0.0
                o = e.reset()
            obs.append(o)
        return np.stack(obs), rews, dones
