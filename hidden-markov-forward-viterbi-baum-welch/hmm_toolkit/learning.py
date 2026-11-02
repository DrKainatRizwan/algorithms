"""Parameter estimation: Baum-Welch EM and supervised maximum likelihood."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .inference import forward, backward, _xi_from
from .models import HMM, random_hmm


@dataclass
class FitResult:
    """Outcome of :func:`baum_welch`."""

    hmm: HMM
    log_likelihoods: list[float] = field(default_factory=list)
    n_iter: int = 0
    converged: bool = False

    @property
    def final_ll(self) -> float:
        return self.log_likelihoods[-1]


def _normalise_rows(m: np.ndarray, floor: float) -> np.ndarray:
    m = m + floor
    s = m.sum(axis=-1, keepdims=True)
    return m / s


def _em_step(hmm: HMM, seqs: list[np.ndarray], pseudo: float
             ) -> tuple[HMM, float]:
    N, M = hmm.n_states, hmm.n_symbols
    pi_acc = np.zeros(N)
    A_acc = np.zeros((N, N))
    B_acc = np.zeros((N, M))
    total = 0.0
    for x in seqs:
        alpha, c = forward(hmm, x)
        beta = backward(hmm, x, c)
        total += float(np.log(c).sum())
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)
        pi_acc += gamma[0]
        A_acc += _xi_from(hmm, x, alpha, beta, c)
        B_acc += gamma.T @ np.eye(M)[x]
    new = HMM(_normalise_rows(pi_acc[None, :], pseudo)[0],
              _normalise_rows(A_acc, pseudo), _normalise_rows(B_acc, pseudo))
    return new, total


def baum_welch(obs: Sequence[np.ndarray] | np.ndarray, n_states: int,
               rng: np.random.Generator, n_symbols: int | None = None,
               init: HMM | None = None, max_iter: int = 200,
               tol: float = 1e-6, pseudocount: float = 1e-8,
               n_restarts: int = 1) -> FitResult:
    """Fit an HMM by EM on one or several observation sequences.

    The log-likelihood is non-decreasing at every iteration (up to the tiny
    pseudocount smoothing).  With ``n_restarts > 1`` the best run by final
    log-likelihood is returned.
    """
    seqs = [np.asarray(obs, dtype=int)] if np.ndim(obs[0]) == 0 else \
        [np.asarray(s, dtype=int) for s in obs]
    M = n_symbols if n_symbols is not None else int(max(s.max() for s in seqs)) + 1
    best: FitResult | None = None
    for r in range(max(1, n_restarts)):
        hmm = init.copy() if (init is not None and r == 0) else \
            random_hmm(n_states, M, rng, concentration=2.0)
        lls: list[float] = []
        converged = False
        for it in range(max_iter):
            hmm, ll = _em_step(hmm, seqs, pseudocount)
            lls.append(ll)
            if len(lls) > 1 and abs(lls[-1] - lls[-2]) < tol * (1 + abs(lls[-2])):
                converged = True
                break
        # evaluate the final parameters
        lls.append(sum(float(np.log(forward(hmm, s)[1]).sum()) for s in seqs))
        res = FitResult(hmm, lls, len(lls) - 1, converged)
        if best is None or res.final_ll > best.final_ll:
            best = res
    assert best is not None
    return best


def supervised_fit(states: Sequence[np.ndarray], obs: Sequence[np.ndarray],
                   n_states: int, n_symbols: int, smoothing: float = 1.0) -> HMM:
    """Maximum-likelihood (Laplace-smoothed) estimate from labelled data."""
    pi = np.full(n_states, smoothing)
    A = np.full((n_states, n_states), smoothing)
    B = np.full((n_states, n_symbols), smoothing)
    for z, x in zip(states, obs):
        z, x = np.asarray(z), np.asarray(x)
        pi[z[0]] += 1
        np.add.at(A, (z[:-1], z[1:]), 1)
        np.add.at(B, (z, x), 1)
    return HMM(pi / pi.sum(), _normalise_rows(A, 0), _normalise_rows(B, 0))
