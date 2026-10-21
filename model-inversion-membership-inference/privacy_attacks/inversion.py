"""Model-inversion attacks (Fredrikson et al. 2015).

White-box inversion does gradient ascent on ``log p(c | x)`` with a Gaussian
prior pulling the reconstruction toward a (possibly public) mean.  Black-box
inversion uses natural-evolution-strategy gradient estimates from API queries
only, so it also works against output-perturbing defences.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import centered_correlation


@dataclass
class InversionResult:
    x: np.ndarray
    final_objective: float
    steps: int


def invert_whitebox(model, target: int, dim: int, steps: int = 300, lr: float = 0.1,
                    prior_weight: float = 0.01, prior_mean: np.ndarray | None = None,
                    clip: float | None = 6.0) -> InversionResult:
    """Gradient-ascent reconstruction of a representative of class ``target``."""
    mu = np.zeros(dim) if prior_mean is None else np.asarray(prior_mean, float)
    x = mu.copy()
    for _ in range(steps):
        g = model.input_gradient(x, target) - prior_weight * (x - mu)
        x = x + lr * g
        if clip is not None:
            x = np.clip(x, -clip, clip)
    obj = float(np.log(model.predict_proba(x)[0, target] + 1e-12) - 0.5 * prior_weight * ((x - mu) ** 2).sum())
    return InversionResult(x, obj, steps)


def invert_blackbox(api, target: int, dim: int, steps: int = 300, lr: float = 0.1, sigma: float = 0.3,
                    n_samples: int = 16, prior_weight: float = 0.01,
                    prior_mean: np.ndarray | None = None, seed: int = 0,
                    clip: float | None = 6.0) -> InversionResult:
    """NES (antithetic) gradient estimation using only ``api.predict_proba``."""
    rng = np.random.default_rng(seed)
    mu = np.zeros(dim) if prior_mean is None else np.asarray(prior_mean, float)

    def obj(X: np.ndarray) -> np.ndarray:
        p = api.predict_proba(X)[:, target]
        return np.log(np.clip(p, 1e-6, 1.0)) - 0.5 * prior_weight * ((X - mu) ** 2).sum(axis=1)

    x = mu.copy()
    for _ in range(steps):
        eps = rng.standard_normal((n_samples, dim))
        f_plus, f_minus = obj(x + sigma * eps), obj(x - sigma * eps)
        g = ((f_plus - f_minus)[:, None] * eps).mean(axis=0) / (2 * sigma)
        x = x + lr * g
        if clip is not None:
            x = np.clip(x, -clip, clip)
    return InversionResult(x, float(obj(x[None])[0]), steps)


def reconstruction_quality(recon: np.ndarray, prototypes: np.ndarray, target: int) -> dict[str, float]:
    """Compare a reconstruction against ground-truth prototypes.

    ``corr`` is the Pearson correlation with the true prototype; ``margin`` is
    that correlation minus the best correlation with any wrong prototype
    (positive means the reconstruction identifies the right class).
    """
    corrs = np.array([centered_correlation(recon, p) for p in prototypes])
    wrong = np.delete(corrs, target)
    return {"corr": float(corrs[target]), "margin": float(corrs[target] - wrong.max()),
            "identified": float(corrs.argmax() == target)}


def invert_all_classes(model, protos: np.ndarray, blackbox: bool = False, **kw) -> list[dict[str, float]]:
    """Invert every class and report :func:`reconstruction_quality` for each."""
    out = []
    for c in range(protos.shape[0]):
        r = (invert_blackbox if blackbox else invert_whitebox)(model, c, protos.shape[1], **kw)
        out.append(reconstruction_quality(r.x, protos, c))
    return out
