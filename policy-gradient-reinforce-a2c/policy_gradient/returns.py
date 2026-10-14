"""Return and advantage estimators."""

from __future__ import annotations

import numpy as np


def discounted_returns(rewards: np.ndarray, gamma: float, bootstrap: float = 0.0) -> np.ndarray:
    """Reward-to-go ``G_t = r_t + gamma * G_{t+1}`` for a single trajectory.

    ``bootstrap`` is the value assigned beyond the last step (0 for terminal).
    """
    out = np.zeros(len(rewards))
    running = bootstrap
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        out[t] = running
    return out


def gae(rewards: np.ndarray, values: np.ndarray, next_values: np.ndarray,
        dones: np.ndarray, gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Generalised advantage estimation (Schulman et al., 2016).

    Arrays have shape ``(T, N)`` for ``T`` steps in ``N`` environments.
    ``values[t]`` is V(s_t), ``next_values[t]`` is V(s_{t+1}) (already
    corrected for time-limit truncation) and ``dones[t]`` marks true
    termination, which cuts both the bootstrap and the trace.
    Returns ``(advantages, value_targets)``.
    """
    T = rewards.shape[0]
    adv = np.zeros_like(rewards, dtype=float)
    last = np.zeros(rewards.shape[1:])
    for t in reversed(range(T)):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_values[t] * nonterminal - values[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    return adv, adv + values


def normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Standardise to zero mean and unit variance."""
    return (x - x.mean()) / (x.std() + eps)
