"""Brute-force oracles, state matching and model selection utilities."""
from __future__ import annotations

from itertools import permutations, product
from typing import Sequence

import numpy as np

from .learning import baum_welch
from .models import HMM


def _path_prob(hmm: HMM, z: Sequence[int], x: np.ndarray) -> float:
    p = hmm.pi[z[0]] * hmm.B[z[0], x[0]]
    for t in range(1, len(z)):
        p *= hmm.A[z[t - 1], z[t]] * hmm.B[z[t], x[t]]
    return float(p)


def brute_force_likelihood(hmm: HMM, obs: np.ndarray) -> float:
    """Sum over all N^T state paths -- exponential, for tiny test cases."""
    x = hmm.check_obs(obs)
    total = sum(_path_prob(hmm, z, x)
                for z in product(range(hmm.n_states), repeat=x.size))
    return float(np.log(total)) if total > 0 else float("-inf")


def brute_force_viterbi(hmm: HMM, obs: np.ndarray) -> tuple[np.ndarray, float]:
    """Exhaustive argmax over state paths (log joint probability)."""
    x = hmm.check_obs(obs)
    best, best_p = None, -1.0
    for z in product(range(hmm.n_states), repeat=x.size):
        p = _path_prob(hmm, z, x)
        if p > best_p:
            best, best_p = z, p
    return np.array(best), float(np.log(best_p))


def match_states(true_states: np.ndarray, pred_states: np.ndarray,
                 n_states: int) -> np.ndarray:
    """Permutation ``perm`` so that ``perm[pred]`` best agrees with truth."""
    conf = np.zeros((n_states, n_states), dtype=int)
    np.add.at(conf, (pred_states, true_states), 1)
    best, best_score = None, -1
    for p in permutations(range(n_states)):
        s = sum(conf[i, p[i]] for i in range(n_states))
        if s > best_score:
            best, best_score = p, s
    return np.array(best)


def state_accuracy(true_states: np.ndarray, pred_states: np.ndarray,
                   n_states: int) -> float:
    """Accuracy after optimally relabelling the predicted states."""
    perm = match_states(true_states, pred_states, n_states)
    return float(np.mean(perm[pred_states] == true_states))


def bic(ll: float, n_states: int, n_symbols: int, n_obs: int) -> float:
    """Bayesian information criterion (lower is better)."""
    k = (n_states - 1) + n_states * (n_states - 1) + n_states * (n_symbols - 1)
    return -2.0 * ll + k * float(np.log(n_obs))


def select_n_states(seqs: Sequence[np.ndarray], candidates: Sequence[int],
                    n_symbols: int, rng: np.random.Generator,
                    n_restarts: int = 3) -> dict[int, float]:
    """BIC score per candidate number of hidden states."""
    n_obs = sum(len(s) for s in seqs)
    out = {}
    for n in candidates:
        res = baum_welch(list(seqs), n, rng, n_symbols=n_symbols,
                         n_restarts=n_restarts, max_iter=100)
        out[n] = bic(res.final_ll, n, n_symbols, n_obs)
    return out
