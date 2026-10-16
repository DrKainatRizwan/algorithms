"""Gradient-based evasion attacks: FGSM, PGD (L-inf / L2) and MI-FGSM.

All attacks operate on batches, return adversarial examples of the same shape
and guarantee ``||x_adv - x|| <= eps`` in the chosen norm (and optionally that
the point stays inside a box ``clip``).
"""

from __future__ import annotations

import numpy as np

from .model import MLP

Clip = tuple[float, float] | None


def project_linf(delta: np.ndarray, eps: float) -> np.ndarray:
    """Project perturbations onto the L-inf ball of radius eps."""
    return np.clip(delta, -eps, eps)


def project_l2(delta: np.ndarray, eps: float) -> np.ndarray:
    """Project perturbations (row-wise) onto the L2 ball of radius eps."""
    norms = np.linalg.norm(delta, axis=1, keepdims=True)
    scale = np.minimum(1.0, eps / np.maximum(norms, 1e-12))
    return delta * scale


def _project(delta: np.ndarray, eps: float, norm: str) -> np.ndarray:
    if norm == "linf":
        return project_linf(delta, eps)
    if norm == "l2":
        return project_l2(delta, eps)
    raise ValueError(f"unknown norm {norm!r}")


def _steepest(grad: np.ndarray, norm: str) -> np.ndarray:
    """Unit-norm ascent direction for the given norm."""
    if norm == "linf":
        return np.sign(grad)
    n = np.linalg.norm(grad, axis=1, keepdims=True)
    return grad / np.maximum(n, 1e-12)


def _clip(x: np.ndarray, clip: Clip) -> np.ndarray:
    return x if clip is None else np.clip(x, clip[0], clip[1])


def random_noise(x: np.ndarray, eps: float, norm: str = "linf", seed: int = 0,
                 clip: Clip = None) -> np.ndarray:
    """Uniform random perturbation baseline."""
    rng = np.random.default_rng(seed)
    if norm == "linf":
        d = rng.uniform(-eps, eps, x.shape)
    else:
        d = rng.standard_normal(x.shape)
        d = project_l2(d / np.linalg.norm(d, axis=1, keepdims=True) * eps, eps)
    return _clip(x + d, clip)


def fgsm(model: MLP, x: np.ndarray, y: np.ndarray, eps: float, norm: str = "linf",
         targeted: np.ndarray | None = None, clip: Clip = None) -> np.ndarray:
    """Fast Gradient Sign Method (Goodfellow et al., 2015).

    ``x_adv = x + eps * sign(grad_x L(x, y))``.  For ``norm='l2'`` the sign is
    replaced by the normalised gradient.  If ``targeted`` (target labels) is
    given, descends the loss of the target class instead.
    """
    if targeted is None:
        g = model.input_gradient(x, y)
        step = _steepest(g, norm)
    else:
        g = model.input_gradient(x, targeted)
        step = -_steepest(g, norm)
    return _clip(x + eps * step, clip)


def pgd(model: MLP, x: np.ndarray, y: np.ndarray, eps: float, alpha: float | None = None,
        steps: int = 10, norm: str = "linf", random_start: bool = True, seed: int = 0,
        targeted: np.ndarray | None = None, clip: Clip = None) -> np.ndarray:
    """Projected Gradient Descent attack (Madry et al., 2018).

    Iterates ``x <- Proj(x + alpha * steepest(grad))`` from a (optionally
    random) start inside the eps-ball.  Default step size is ``2.5 * eps / steps``.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    alpha = 2.5 * eps / steps if alpha is None else alpha
    rng = np.random.default_rng(seed)
    if random_start:
        if norm == "linf":
            delta = rng.uniform(-eps, eps, x.shape)
        else:
            delta = project_l2(rng.standard_normal(x.shape), eps)
    else:
        delta = np.zeros_like(x)
    sign = 1.0 if targeted is None else -1.0
    labels = y if targeted is None else targeted
    for _ in range(steps):
        xa = _clip(x + delta, clip)
        g = model.input_gradient(xa, labels)
        delta = delta + sign * alpha * _steepest(g, norm)
        delta = _project(delta, eps, norm)
        delta = _clip(x + delta, clip) - x
    return x + delta


def mi_fgsm(model: MLP, x: np.ndarray, y: np.ndarray, eps: float, steps: int = 10,
            decay: float = 1.0, clip: Clip = None) -> np.ndarray:
    """Momentum Iterative FGSM (Dong et al., 2018), L-inf."""
    alpha = eps / steps
    xa = x.copy()
    mom = np.zeros_like(x)
    for _ in range(steps):
        g = model.input_gradient(xa, y)
        g = g / np.maximum(np.abs(g).sum(axis=1, keepdims=True), 1e-12)
        mom = decay * mom + g
        xa = xa + alpha * np.sign(mom)
        xa = x + project_linf(xa - x, eps)
        xa = _clip(xa, clip)
    return xa
