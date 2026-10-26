"""Defences against training-set poisoning.

Every defence has the signature ``f(X, y, l2=..., **opts) -> DefenseResult`` and
returns a fitted model plus the boolean mask of samples it kept.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .model import LogisticRegression


@dataclass
class DefenseResult:
    model: LogisticRegression
    kept: np.ndarray  # boolean mask over the (poisoned) training rows


def no_defense(X: np.ndarray, y: np.ndarray, l2: float = 1.0, **_: object) -> DefenseResult:
    """Plain fit on everything."""
    return DefenseResult(LogisticRegression(l2).fit(X, y), np.ones(len(y), dtype=bool))


def trim_fit(X: np.ndarray, y: np.ndarray, l2: float = 1.0, alpha: float = 0.15,
             max_iter: int = 50, **_: object) -> DefenseResult:
    """TRIM (Jagielski et al., 2018): alternate fitting and keeping the lowest-loss points.

    ``alpha`` is the assumed contamination fraction; ``(1 - alpha) * n`` points
    with the smallest residual are retained at each iteration until the kept
    set stabilises.
    """
    n = len(y)
    keep_n = max(2, int(round((1.0 - alpha) * n)))
    model = LogisticRegression(l2).fit(X, y)
    kept = np.zeros(n, dtype=bool)
    for _ in range(max_iter):
        losses = model.per_sample_loss(X, y)
        idx = np.argsort(losses, kind="stable")[:keep_n]
        new = np.zeros(n, dtype=bool)
        new[idx] = True
        if np.array_equal(new, kept):
            break
        kept = new
        model = LogisticRegression(l2).fit(X[kept], y[kept], w0=model.w)
    return DefenseResult(model, kept)


def sphere_slab_filter(X: np.ndarray, y: np.ndarray, l2: float = 1.0, remove_frac: float = 0.1,
                       **_: object) -> DefenseResult:
    """Sphere and slab outlier removal (Steinhardt et al., 2017).

    Class centroids are estimated with the coordinate-wise median.  A point is
    dropped if it lies in the top ``remove_frac`` of its class by distance to the
    centroid (sphere) or by |projection on the centroid-difference axis| after
    centring (slab).
    """
    n = len(y)
    c0 = np.median(X[y == 0], axis=0)
    c1 = np.median(X[y == 1], axis=0)
    axis = c1 - c0
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    drop = np.zeros(n, dtype=bool)
    for cls, c in ((0, c0), (1, c1)):
        m = y == cls
        if m.sum() == 0:
            continue
        diff = X[m] - c
        sphere = np.linalg.norm(diff, axis=1)
        slab = np.abs(diff @ axis)
        q = 1.0 - remove_frac
        bad = (sphere > np.quantile(sphere, q)) | (slab > np.quantile(slab, q))
        # keep the overall drop rate near remove_frac: rank by the larger normalised score
        score = np.maximum(sphere / (np.median(sphere) + 1e-12), slab / (np.median(slab) + 1e-12))
        k = int(round(remove_frac * m.sum()))
        order = np.argsort(-score)[:k]
        sel = np.zeros(m.sum(), dtype=bool)
        sel[order] = True
        drop[np.flatnonzero(m)[sel & bad]] = True
    kept = ~drop
    return DefenseResult(LogisticRegression(l2).fit(X[kept], y[kept]), kept)


def knn_sanitize(X: np.ndarray, y: np.ndarray, l2: float = 1.0, k: int = 9,
                 agree: float = 0.5, **_: object) -> DefenseResult:
    """Drop points whose label disagrees with most of their ``k`` nearest neighbours.

    Effective against scattered label flips; O(n^2 d) via a dense distance matrix.
    """
    n = len(y)
    D = ((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2)
    np.fill_diagonal(D, np.inf)
    kk = min(k, n - 1)
    nbr = np.argpartition(D, kk - 1, axis=1)[:, :kk]
    frac_same = (y[nbr] == y[:, None]).mean(axis=1)
    kept = frac_same >= agree
    if kept.sum() < 2 or len(np.unique(y[kept])) < 2:
        kept = np.ones(n, dtype=bool)
    return DefenseResult(LogisticRegression(l2).fit(X[kept], y[kept]), kept)


def trimmed_gradient_fit(X: np.ndarray, y: np.ndarray, l2: float = 1.0, beta: float = 0.15,
                         iters: int = 400, lr: float = 0.5, **_: object) -> DefenseResult:
    """Gradient descent with a coordinate-wise trimmed mean of per-sample gradients.

    A fraction ``beta`` of the largest and smallest values in each gradient
    coordinate is discarded before averaging (Yin et al., 2018), bounding the
    pull any small group of samples can exert on the update.
    """
    n = len(y)
    model = LogisticRegression(l2)
    w = np.zeros(X.shape[1] + 1)
    m = int(np.floor(beta * n))
    for t in range(iters):
        G = model.per_sample_grad(X, y, w)
        G.sort(axis=0)
        mean_g = G[m:n - m].mean(axis=0) if n - 2 * m > 0 else G.mean(axis=0)
        w = w - (lr / (1.0 + 0.01 * t)) * (mean_g + l2 * w / n)
    model.w = w
    return DefenseResult(model, np.ones(n, dtype=bool))


DEFENSES: dict[str, Callable[..., DefenseResult]] = {
    "none": no_defense,
    "trim": trim_fit,
    "sphere+slab": sphere_slab_filter,
    "knn": knn_sanitize,
    "trimmed-grad": trimmed_gradient_fit,
}
