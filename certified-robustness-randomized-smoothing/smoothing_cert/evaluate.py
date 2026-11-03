"""Certified-accuracy evaluation and closed-form checks for linear models."""

from __future__ import annotations

import numpy as np

from .attacks import smoothed_pgd_l2
from .models import LinearSoftmax
from .smoothing import ABSTAIN, CertifyResult, SmoothedClassifier


def certify_dataset(smoothed: SmoothedClassifier, X: np.ndarray, n0: int = 50, n: int = 500,
                    alpha: float = 0.001) -> list[CertifyResult]:
    return [smoothed.certify(x, n0=n0, n=n, alpha=alpha) for x in X]


def certified_accuracy(results: list[CertifyResult], y: np.ndarray, radius: float) -> float:
    """Fraction of points correctly classified with certified radius >= ``radius``."""
    ok = [r.label == t and r.label != ABSTAIN and r.radius >= radius for r, t in zip(results, y)]
    return float(np.mean(ok))


def certified_accuracy_curve(results: list[CertifyResult], y: np.ndarray,
                             radii: np.ndarray) -> np.ndarray:
    return np.array([certified_accuracy(results, y, r) for r in radii])


def exact_linear_radius(model: LinearSoftmax, x: np.ndarray) -> float:
    """Exact L2 distance from ``x`` to the decision boundary of a linear softmax model."""
    z = model.logits(x[None, :])[0]
    c = int(np.argmax(z))
    best = np.inf
    for j in range(model.num_classes):
        if j == c:
            continue
        w = model.W[:, c] - model.W[:, j]
        nw = np.linalg.norm(w)
        if nw > 0:
            best = min(best, (z[c] - z[j]) / nw)
    return float(best)


def empirical_robust_accuracy(smoothed: SmoothedClassifier, model, X: np.ndarray, y: np.ndarray,
                              eps: float, n: int = 300, steps: int = 10, eot: int = 32) -> float:
    """Accuracy of the smoothed classifier on EOT-PGD adversarial examples of size ``eps``."""
    hits = 0
    for x, t in zip(X, y):
        xa = smoothed_pgd_l2(smoothed, model, x, int(t), eps, steps=steps, eot=eot)
        hits += smoothed.predict(xa, n=n, alpha=0.01) == t
    return hits / len(X)
