"""Theory helpers: Lai-Robbins bound, UCB1 bound, log-fit of regret curves."""
from __future__ import annotations

import numpy as np


def bernoulli_kl(p: float, q: float, eps: float = 1e-12) -> float:
    """KL(Ber(p) || Ber(q))."""
    p = min(max(p, eps), 1 - eps)
    q = min(max(q, eps), 1 - eps)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def lai_robbins_constant(probs) -> float:
    """Asymptotic coefficient c with regret >= c * ln(T) for consistent policies."""
    p = np.asarray(probs, dtype=float)
    best = p.max()
    return float(sum((best - m) / bernoulli_kl(m, best) for m in p if m < best))


def ucb1_regret_bound(probs, horizon: int) -> float:
    """Auer et al. upper bound: sum_i [8 ln T / delta_i + (1 + pi^2/3) delta_i]."""
    p = np.asarray(probs, dtype=float)
    d = p.max() - p
    d = d[d > 0]
    return float(np.sum(8 * np.log(horizon) / d + (1 + np.pi**2 / 3) * d))


def log_fit_slope(regret: np.ndarray, start: int = 100) -> float:
    """Least-squares slope of regret against ln(t); ~Lai-Robbins constant if logarithmic."""
    t = np.arange(1, len(regret) + 1)[start:]
    x = np.log(t)
    y = regret[start:]
    xm, ym = x.mean(), y.mean()
    return float(((x - xm) * (y - ym)).sum() / ((x - xm) ** 2).sum())


def exp3_regret_bound(n_arms: int, horizon: int) -> float:
    """EXP3 worst-case bound 2 sqrt(e - 1) sqrt(T K ln K) (Auer et al., 2002b)."""
    return float(2 * np.sqrt(np.e - 1) * np.sqrt(horizon * n_arms * np.log(n_arms)))
