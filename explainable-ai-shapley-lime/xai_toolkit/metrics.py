"""Evaluation metrics for attributions."""
from __future__ import annotations
from typing import Callable
import numpy as np


def efficiency_gap(phi: np.ndarray, base: float, full: float) -> float:
    """|sum(phi) - (f(x) - E f)|; zero for a perfect Shapley attribution."""
    return float(abs(np.sum(phi) - (full - base)))


def deletion_curve(f: Callable[[np.ndarray], np.ndarray], x: np.ndarray, attr: np.ndarray,
                   fill: np.ndarray, signed: bool = False) -> np.ndarray:
    """Model output as features are replaced by ``fill``.

    Order is decreasing |attr|, or decreasing signed attr when ``signed`` is True.
    """
    order = np.argsort(-(attr if signed else np.abs(attr)), kind="stable")
    z = np.array(x, dtype=float)
    out = [float(f(z[None, :])[0])]
    for i in order:
        z[i] = fill[i]
        out.append(float(f(z[None, :])[0]))
    return np.array(out)


def deletion_auc(curve: np.ndarray) -> float:
    """Normalised area under a deletion curve (trapezoid, x in [0,1])."""
    return float(np.trapezoid(curve, dx=1.0 / (len(curve) - 1)))


def rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman correlation of two score vectors (average ranks for ties)."""
    def rk(v):
        order = np.argsort(v, kind="stable")
        r = np.empty(len(v)); r[order] = np.arange(len(v))
        for u in np.unique(v):
            m = v == u
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r
    ra, rb = rk(np.asarray(a)), rk(np.asarray(b))
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def topk_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    """Fraction of shared indices among the top-k by magnitude."""
    ta = set(np.argsort(-np.abs(a))[:k].tolist())
    tb = set(np.argsort(-np.abs(b))[:k].tolist())
    return len(ta & tb) / k


def explanation_stability(explain: Callable[[int], np.ndarray], seeds: list[int]) -> float:
    """Mean pairwise Spearman correlation of explanations across random seeds."""
    E = [explain(s) for s in seeds]
    cs = [rank_correlation(E[i], E[j]) for i in range(len(E)) for j in range(i + 1, len(E))]
    return float(np.mean(cs))
