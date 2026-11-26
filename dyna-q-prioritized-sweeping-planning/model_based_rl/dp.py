"""Exact dynamic programming on a known tabular MDP."""

from __future__ import annotations

import numpy as np


def value_iteration(
    P: np.ndarray,
    R: np.ndarray,
    terminal: np.ndarray,
    gamma: float = 0.95,
    tol: float = 1e-10,
    max_iter: int = 100_000,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return ``(V, Q, sweeps)`` for the optimal value function.

    ``P`` is ``[S, A, S]``, ``R`` is ``[S, A]``; terminal states have value 0.
    """
    S, A, _ = P.shape
    V = np.zeros(S)
    for k in range(1, max_iter + 1):
        Q = R + gamma * (P @ V)
        V_new = Q.max(axis=1)
        V_new[terminal] = 0.0
        diff = np.abs(V_new - V).max()
        V = V_new
        if diff < tol:
            break
    Q = R + gamma * (P @ V)
    Q[terminal] = 0.0
    return V, Q, k


def policy_evaluation(
    P: np.ndarray, R: np.ndarray, terminal: np.ndarray, policy: np.ndarray, gamma: float
) -> np.ndarray:
    """Solve ``V = R_pi + gamma P_pi V`` exactly with a linear solve."""
    S = P.shape[0]
    idx = np.arange(S)
    P_pi = P[idx, policy]
    R_pi = R[idx, policy].copy()
    P_pi = P_pi.copy()
    P_pi[terminal] = 0.0
    R_pi[terminal] = 0.0
    return np.linalg.solve(np.eye(S) - gamma * P_pi, R_pi)


def policy_iteration(
    P: np.ndarray, R: np.ndarray, terminal: np.ndarray, gamma: float = 0.95
) -> tuple[np.ndarray, np.ndarray, int]:
    """Howard's policy iteration; returns ``(V, policy, iterations)``."""
    S = P.shape[0]
    policy = np.zeros(S, dtype=int)
    it = 0
    while True:
        it += 1
        V = policy_evaluation(P, R, terminal, policy, gamma)
        Q = R + gamma * (P @ V)
        # Keep the current action unless another is strictly better.
        cur = Q[np.arange(S), policy]
        better = Q.max(axis=1) > cur + 1e-12
        new_policy = np.where(better, Q.argmax(axis=1), policy)
        if np.array_equal(new_policy, policy):
            return V, policy, it
        policy = new_policy


def optimal_steps(
    P: np.ndarray, R: np.ndarray, terminal: np.ndarray, start: int, gamma: float = 0.95
) -> int:
    """Length of the greedy optimal path from ``start`` (deterministic MDPs)."""
    V, Q, _ = value_iteration(P, R, terminal, gamma)
    s, n = start, 0
    while not terminal[s] and n < P.shape[0] * 2:
        a = int(Q[s].argmax())
        s = int(P[s, a].argmax())
        n += 1
    return n
