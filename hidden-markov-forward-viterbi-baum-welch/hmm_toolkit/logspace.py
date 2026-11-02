"""Log-domain forward and Viterbi, an independent numerical cross-check."""
from __future__ import annotations

import numpy as np

from .models import HMM


def logsumexp(a: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Numerically stable ``log(sum(exp(a)))`` that tolerates ``-inf``."""
    a = np.asarray(a, dtype=float)
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore"):
        out = np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True)) + m
    return out.squeeze(axis=axis) if axis is not None else out.reshape(())


def log_forward(hmm: HMM, obs: np.ndarray) -> tuple[np.ndarray, float]:
    """Return ``(log_alpha, log P(x))`` computed fully in log space."""
    x = hmm.check_obs(obs)
    with np.errstate(divide="ignore"):
        lp, lA, lB = np.log(hmm.pi), np.log(hmm.A), np.log(hmm.B)
    la = np.empty((x.size, hmm.n_states))
    la[0] = lp + lB[:, x[0]]
    for t in range(1, x.size):
        la[t] = logsumexp(la[t - 1][:, None] + lA, axis=0) + lB[:, x[t]]
    return la, float(logsumexp(la[-1]))


def log_viterbi(hmm: HMM, obs: np.ndarray) -> tuple[np.ndarray, float]:
    """Log-domain Viterbi with explicit back-pointers (ties -> lowest index)."""
    x = hmm.check_obs(obs)
    with np.errstate(divide="ignore"):
        lp, lA, lB = np.log(hmm.pi), np.log(hmm.A), np.log(hmm.B)
    T, N = x.size, hmm.n_states
    d = lp + lB[:, x[0]]
    back = np.zeros((T, N), dtype=int)
    for t in range(1, T):
        cand = d[:, None] + lA
        back[t] = cand.argmax(axis=0)
        d = cand.max(axis=0) + lB[:, x[t]]
    path = [int(d.argmax())]
    for t in range(T - 1, 0, -1):
        path.append(int(back[t, path[-1]]))
    return np.array(path[::-1]), float(d.max())
