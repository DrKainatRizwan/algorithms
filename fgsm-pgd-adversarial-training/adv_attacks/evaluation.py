"""Metrics and sanity checks for attacks and defences."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .attacks import fgsm, pgd
from .model import MLP

Attack = Callable[[MLP, np.ndarray, np.ndarray, float], np.ndarray]


def accuracy(model: MLP, X: np.ndarray, y: np.ndarray) -> float:
    return float((model.predict(X) == y).mean())


def robust_accuracy(model: MLP, X: np.ndarray, y: np.ndarray, eps: float,
                    attack: str | Attack = "pgd", **kw) -> float:
    """Accuracy on adversarial inputs generated with ``attack``."""
    if eps == 0:
        return accuracy(model, X, y)
    fn = _resolve(attack, **kw)
    return accuracy(model, fn(model, X, y, eps), y)


def _resolve(attack: str | Attack, **kw) -> Attack:
    if callable(attack):
        return attack
    if attack == "fgsm":
        return lambda m, x, y, e: fgsm(m, x, y, e, **kw)
    if attack == "pgd":
        return lambda m, x, y, e: pgd(m, x, y, e, **kw)
    raise ValueError(f"unknown attack {attack!r}")


def epsilon_sweep(model: MLP, X: np.ndarray, y: np.ndarray, epsilons,
                  attack: str | Attack = "pgd", **kw) -> list[tuple[float, float]]:
    """Robust accuracy for each epsilon: ``[(eps, acc), ...]``."""
    return [(float(e), robust_accuracy(model, X, y, float(e), attack, **kw))
            for e in epsilons]


def attack_success_rate(model: MLP, X: np.ndarray, y: np.ndarray, X_adv: np.ndarray) -> float:
    """Fraction of originally correct points that the attack flips."""
    ok = model.predict(X) == y
    if not ok.any():
        return 0.0
    return float((model.predict(X_adv)[ok] != y[ok]).mean())


def linear_margin_drop(w: np.ndarray, eps: float, norm: str = "linf") -> float:
    """Worst-case margin decrease of a *binary linear* score ``w.x + b`` under an
    eps perturbation: ``eps * ||w||_1`` for L-inf, ``eps * ||w||_2`` for L2
    (dual norms).  FGSM is exactly optimal in this setting."""
    w = np.asarray(w, dtype=float)
    return float(eps * (np.abs(w).sum() if norm == "linf" else np.linalg.norm(w)))


def gradient_check(model: MLP, X: np.ndarray, y: np.ndarray, h: float = 1e-5,
                   n_checks: int = 20, seed: int = 0) -> float:
    """Max relative error between analytic and finite-difference input gradients."""
    rng = np.random.default_rng(seed)
    g = model.input_gradient(X, y)
    worst = 0.0
    for _ in range(n_checks):
        i = int(rng.integers(len(y)))
        j = int(rng.integers(X.shape[1]))
        xp, xm = X[i:i + 1].copy(), X[i:i + 1].copy()
        xp[0, j] += h
        xm[0, j] -= h
        yi = y[i:i + 1]
        num = (model.loss(xp, yi) - model.loss(xm, yi)) / (2 * h)
        err = abs(num - g[i, j]) / max(1e-8, abs(num) + abs(g[i, j]))
        worst = max(worst, err)
    return worst
