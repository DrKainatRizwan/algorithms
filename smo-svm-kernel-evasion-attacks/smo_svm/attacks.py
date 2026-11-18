"""Evasion attacks against a fitted binary :class:`~smo_svm.smo.SVC`."""

from __future__ import annotations

import numpy as np

from .smo import SVC

Array = np.ndarray


def deepfool(model: SVC, x: Array, overshoot: float = 0.02, max_iter: int = 100) -> Array:
    """Minimum-norm (L2) adversarial example by iterated linearisation.

    For a linear kernel a single step lands exactly on the hyperplane; for
    non-linear kernels the decision surface is re-linearised each iteration.
    """
    x0 = np.asarray(x, dtype=float)
    f0 = model.decision_function(x0)[0]
    sign0 = 1.0 if f0 >= 0 else -1.0
    xa = x0.copy()
    r_tot = np.zeros_like(x0)
    for _ in range(max_iter):
        f = model.decision_function(xa)[0]
        if (f >= 0) != (sign0 > 0):
            break
        g = model.gradient(xa)
        gn = float(g @ g)
        if gn < 1e-18:
            break
        r_tot = r_tot - (f / gn) * g * 1.0
        xa = x0 + (1.0 + overshoot) * r_tot
    return xa


def minimum_distance(model: SVC, X: Array) -> Array:
    """L2 distance to the decision boundary found by :func:`deepfool` per row."""
    return np.array([np.linalg.norm(deepfool(model, x) - x) for x in np.atleast_2d(X)])


def pgd_linf(model: SVC, x: Array, label, eps: float, steps: int = 20,
             alpha: float | None = None, seed: int = 0, random_start: bool = False) -> Array:
    """Projected gradient ascent on the margin loss inside an L-inf ball.

    ``label`` is the true class (one of ``model.classes_``). The attack
    minimises ``t * f(x)`` where ``t = +-1`` encodes the true class.
    """
    x0 = np.asarray(x, dtype=float)
    t = 1.0 if label == model.classes_[1] else -1.0
    alpha = alpha if alpha is not None else 2.5 * eps / steps
    xa = x0.copy()
    if random_start:
        rng = np.random.default_rng(seed)
        xa = x0 + rng.uniform(-eps, eps, size=x0.shape)
    for _ in range(steps):
        g = t * model.gradient(xa)
        xa = xa - alpha * np.sign(g)
        xa = x0 + np.clip(xa - x0, -eps, eps)
    return xa


def robust_accuracy(model: SVC, X: Array, y: Array, eps: float, steps: int = 20) -> float:
    """Accuracy under an L-inf PGD attack of radius ``eps``."""
    X = np.asarray(X, dtype=float)
    adv = np.array([pgd_linf(model, x, lab, eps, steps) for x, lab in zip(X, y)])
    return float(np.mean(model.predict(adv) == np.asarray(y)))
