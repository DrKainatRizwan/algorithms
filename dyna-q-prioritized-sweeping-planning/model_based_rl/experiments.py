"""Experiment drivers: episodic learning curves and non-stationary runs."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .envs import GridWorld


def run_episodes(
    env: GridWorld, agent, n_episodes: int, seed: int = 0, max_steps: int = 2000
) -> np.ndarray:
    """Train for ``n_episodes``; returns the number of steps in each."""
    rng = np.random.default_rng(seed)
    steps = np.zeros(n_episodes, dtype=int)
    for ep in range(n_episodes):
        s = env.reset()
        for k in range(1, max_steps + 1):
            a = agent.act(s)
            s2, r, done = env.step(a, rng)
            agent.observe(s, a, r, s2, done)
            s = s2
            if done:
                break
        steps[ep] = k
    return steps


def run_steps(
    env: GridWorld,
    agent,
    total_steps: int,
    seed: int = 0,
    switch_at: int | None = None,
    on_switch: Callable[[GridWorld], None] | None = None,
) -> np.ndarray:
    """Run for a fixed number of real steps; returns cumulative reward.

    At ``switch_at`` the environment is altered by ``on_switch`` (the agent
    is not told), which is how the blocking and shortcut mazes are run.
    """
    rng = np.random.default_rng(seed)
    cum = np.zeros(total_steps)
    s = env.reset()
    total = 0.0
    for t in range(total_steps):
        if switch_at is not None and t == switch_at and on_switch is not None:
            on_switch(env)
        a = agent.act(s)
        s2, r, done = env.step(a, rng)
        agent.observe(s, a, r, s2, done)
        total += r
        cum[t] = total
        s = env.reset() if done else s2
    return cum


def greedy_rollout_length(env: GridWorld, policy: np.ndarray, limit: int = 500) -> int | None:
    """Steps for a deterministic greedy policy to reach the goal, or ``None``."""
    rng = np.random.default_rng(0)
    env.reset()
    for k in range(1, limit + 1):
        _, _, done = env.step(int(policy[env.state]), rng)
        if done:
            return k
    return None
