"""Evaluation helpers."""

from __future__ import annotations

import numpy as np


def evaluate_policy(agent, env, episodes: int = 10, greedy: bool = True, seed: int = 1000) -> float:
    """Mean undiscounted return of ``agent`` over fresh episodes."""
    total = 0.0
    for ep in range(episodes):
        obs = env.reset(seed=seed + ep)
        while True:
            obs, r, term, trunc = env.step(agent.act(obs, greedy=greedy))
            total += r
            if term or trunc:
                break
    return total / episodes


def moving_average(x: list[float] | np.ndarray, window: int) -> np.ndarray:
    """Trailing moving average (shorter windows at the start)."""
    x = np.asarray(x, dtype=float)
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = np.empty(len(x))
    for i in range(len(x)):
        lo = max(0, i + 1 - window)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out
