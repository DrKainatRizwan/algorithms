"""Scaled forward-backward recursions, smoothing and Viterbi decoding."""
from __future__ import annotations

import numpy as np

from .models import HMM


def forward(hmm: HMM, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scaled forward pass.

    Returns ``(alpha_hat, c)`` where ``alpha_hat[t]`` is the filtered
    distribution ``P(z_t | x_{1:t})`` and ``c[t] = P(x_t | x_{1:t-1})``.
    Then ``log P(x) = sum(log c)``.
    """
    x = hmm.check_obs(obs)
    T, N = x.size, hmm.n_states
    alpha = np.empty((T, N))
    c = np.empty(T)
    a = hmm.pi * hmm.B[:, x[0]]
    c[0] = a.sum()
    alpha[0] = a / c[0] if c[0] > 0 else a
    for t in range(1, T):
        a = (alpha[t - 1] @ hmm.A) * hmm.B[:, x[t]]
        c[t] = a.sum()
        alpha[t] = a / c[t] if c[t] > 0 else a
    return alpha, c


def backward(hmm: HMM, obs: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Scaled backward pass using the scale factors from :func:`forward`."""
    x = hmm.check_obs(obs)
    T, N = x.size, hmm.n_states
    beta = np.ones((T, N))
    for t in range(T - 2, -1, -1):
        b = hmm.A @ (hmm.B[:, x[t + 1]] * beta[t + 1])
        beta[t] = b / c[t + 1] if c[t + 1] > 0 else b
    return beta


def likelihood(hmm: HMM, obs: np.ndarray) -> float:
    """Log-likelihood ``log P(obs | hmm)`` (``-inf`` for impossible data)."""
    _, c = forward(hmm, obs)
    if np.any(c <= 0):
        return float("-inf")
    return float(np.log(c).sum())


def posteriors(hmm: HMM, obs: np.ndarray) -> np.ndarray:
    """Smoothed marginals ``gamma[t, i] = P(z_t = i | x_{1:T})``."""
    alpha, c = forward(hmm, obs)
    beta = backward(hmm, obs, c)
    g = alpha * beta
    return g / g.sum(axis=1, keepdims=True)


def _xi_from(hmm: HMM, x: np.ndarray, alpha: np.ndarray, beta: np.ndarray,
             c: np.ndarray) -> np.ndarray:
    """Vectorised sum over t of the pairwise posteriors xi_t(i, j)."""
    if x.size < 2:
        return np.zeros((hmm.n_states, hmm.n_states))
    right = hmm.B[:, x[1:]].T * beta[1:] / c[1:, None]
    return hmm.A * (alpha[:-1].T @ right)


def expected_transitions(hmm: HMM, obs: np.ndarray) -> np.ndarray:
    """Expected transition counts ``sum_t P(z_t=i, z_{t+1}=j | x)``, (N, N)."""
    x = hmm.check_obs(obs)
    alpha, c = forward(hmm, x)
    beta = backward(hmm, x, c)
    return _xi_from(hmm, x, alpha, beta, c)


def viterbi(hmm: HMM, obs: np.ndarray) -> tuple[np.ndarray, float]:
    """Most probable state path and its log joint probability."""
    x = hmm.check_obs(obs)
    T, N = x.size, hmm.n_states
    with np.errstate(divide="ignore"):
        lp, lA, lB = np.log(hmm.pi), np.log(hmm.A), np.log(hmm.B)
    delta = np.empty((T, N))
    psi = np.zeros((T, N), dtype=int)
    delta[0] = lp + lB[:, x[0]]
    for t in range(1, T):
        cand = delta[t - 1][:, None] + lA
        psi[t] = np.argmax(cand, axis=0)
        delta[t] = cand[psi[t], np.arange(N)] + lB[:, x[t]]
    path = np.empty(T, dtype=int)
    path[-1] = int(np.argmax(delta[-1]))
    for t in range(T - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]
    return path, float(delta[-1].max())


def posterior_decode(hmm: HMM, obs: np.ndarray) -> np.ndarray:
    """Per-position MAP states (maximises expected number of correct states)."""
    return np.argmax(posteriors(hmm, obs), axis=1)
