"""Synthetic loss sequences: stochastic, adversarial and regression streams."""
from __future__ import annotations

import numpy as np

from .losses import LinearLoss, LogisticLoss, Loss, SquaredLoss


def stochastic_linear(T: int, d: int, seed: int = 0, noise: float = 0.5) -> list[Loss]:
    """Gaussian gradients with a fixed random mean (unit-norm), clipped to norm <= 2."""
    rng = np.random.default_rng(seed)
    mu = rng.normal(size=d)
    mu /= np.linalg.norm(mu)
    out = []
    for _ in range(T):
        g = mu + noise * rng.normal(size=d)
        n = np.linalg.norm(g)
        out.append(LinearLoss(g if n <= 2.0 else g * 2.0 / n))
    return out


def ftl_adversary(T: int) -> list[Loss]:
    """1-D losses on [-1, 1] that force Follow-the-Leader to linear regret.

    ``g_1 = -0.5`` and then ``+1, -1, +1, ...``: the leader flips between the
    endpoints and always lands on the wrong side, losing 1 per round, while the best
    fixed point loses only ~0.5 in total.
    """
    gs = [-0.5] + [1.0 if i % 2 == 0 else -1.0 for i in range(T - 1)]
    return [LinearLoss(np.array([g])) for g in gs]


def expert_losses(T: int, n: int, seed: int = 0, gap: float = 0.1) -> list[Loss]:
    """Uniform [0,1] losses per expert; expert 0 has mean ``0.5 - gap``."""
    rng = np.random.default_rng(seed)
    L = rng.random((T, n))
    L[:, 0] = np.clip(L[:, 0] - gap, 0.0, 1.0)
    return [LinearLoss(row) for row in L]


def alternating_experts(T: int) -> list[Loss]:
    """Two experts with alternating losses: a classic deterministic-FTL trap.

    Round 1 favours expert 0 slightly; afterwards the currently-leading expert always
    suffers loss 1 while the other suffers 0.
    """
    out = []
    for t in range(T):
        if t == 0:
            row = np.array([0.0, 0.5])
        elif t % 2 == 1:
            row = np.array([1.0, 0.0])
        else:
            row = np.array([0.0, 1.0])
        out.append(LinearLoss(row))
    return out


def regression_stream(T: int, d: int, seed: int = 0, noise: float = 0.1,
                      lam: float = 0.0, radius: float = 1.0) -> list[Loss]:
    """Noisy linear regression ``b = a.w* + eps`` with ``|w*| = radius/2``.

    Features lie on the unit sphere, so residuals on the ball of radius ``R`` are
    bounded by ``R + |b|`` which is used for the exp-concavity constant.
    """
    rng = np.random.default_rng(seed)
    w = rng.normal(size=d)
    w *= 0.5 * radius / np.linalg.norm(w)
    out = []
    for _ in range(T):
        a = rng.normal(size=d)
        a /= np.linalg.norm(a)
        b = float(a @ w + noise * rng.normal())
        bound = radius + abs(b)
        out.append(SquaredLoss(a, b, lam=lam, residual_bound=bound))
    return out


def logistic_stream(T: int, d: int, seed: int = 0, flip: float = 0.05) -> list[Loss]:
    """Linearly separable-ish labels with a fraction ``flip`` of random flips."""
    rng = np.random.default_rng(seed)
    w = rng.normal(size=d)
    w /= np.linalg.norm(w)
    out = []
    for _ in range(T):
        a = rng.normal(size=d)
        a /= max(1.0, np.linalg.norm(a) / 1.0)
        y = 1.0 if a @ w >= 0 else -1.0
        if rng.random() < flip:
            y = -y
        out.append(LogisticLoss(a, y))
    return out
