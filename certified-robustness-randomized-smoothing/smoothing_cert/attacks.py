"""L2 attacks used to sanity-check certificates."""

from __future__ import annotations

import numpy as np

from .smoothing import SmoothedClassifier


def _project(delta: np.ndarray, eps: float) -> np.ndarray:
    norm = np.linalg.norm(delta)
    return delta if norm <= eps or norm == 0 else delta * (eps / norm)


def pgd_l2(model, x: np.ndarray, y: int, eps: float, steps: int = 30,
           step_size: float | None = None) -> np.ndarray:
    """L2 PGD against a base model exposing ``input_gradient``. Returns the adversarial point."""
    step_size = step_size or 2.5 * eps / steps
    delta = np.zeros_like(x)
    for _ in range(steps):
        g = model.input_gradient((x + delta)[None, :], np.array([y]))[0]
        gn = np.linalg.norm(g)
        if gn == 0:
            break
        delta = _project(delta + step_size * g / gn, eps)
        if model.predict((x + delta)[None, :])[0] != y:
            break
    return x + delta


def smoothed_pgd_l2(smoothed: SmoothedClassifier, model, x: np.ndarray, y: int, eps: float,
                    steps: int = 20, eot: int = 64, step_size: float | None = None) -> np.ndarray:
    """PGD with expectation-over-transformation: gradients averaged over Gaussian noise draws."""
    step_size = step_size or 2.5 * eps / steps
    delta = np.zeros_like(x)
    rng = smoothed.rng
    for _ in range(steps):
        noisy = (x + delta)[None, :] + smoothed.sigma * rng.normal(size=(eot, x.size))
        g = model.input_gradient(noisy, np.full(eot, y)).mean(axis=0)
        gn = np.linalg.norm(g)
        if gn == 0:
            break
        delta = _project(delta + step_size * g / gn, eps)
    return x + delta


def minimal_l2_boundary_distance(model, x: np.ndarray, y: int, hi: float = 20.0,
                                 directions: int = 200, seed: int = 0) -> float:
    """Upper bound on the distance to the decision boundary of ``model`` along gradient
    and random directions (bisection along each ray). ``inf`` if none flips within ``hi``."""
    rng = np.random.default_rng(seed)
    g = model.input_gradient(x[None, :], np.array([y]))[0]
    dirs = [g / (np.linalg.norm(g) + 1e-12)]
    R = rng.normal(size=(directions, x.size))
    dirs.extend(R / np.linalg.norm(R, axis=1, keepdims=True))
    best = np.inf
    for d in dirs:
        if model.predict((x + hi * d)[None, :])[0] == y:
            continue
        lo_r, hi_r = 0.0, hi
        for _ in range(30):
            mid = 0.5 * (lo_r + hi_r)
            if model.predict((x + mid * d)[None, :])[0] == y:
                lo_r = mid
            else:
                hi_r = mid
        best = min(best, hi_r)
    return best
