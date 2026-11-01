"""LIME for tabular data: local weighted ridge surrogate on perturbed samples."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np


@dataclass
class LimeExplanation:
    """Result of a LIME run: sparse coefficients in the original feature scale."""
    coef: np.ndarray
    intercept: float
    selected: np.ndarray
    r2: float
    local_pred: float

    def top(self, k: int = 3) -> list[tuple[int, float]]:
        order = np.argsort(-np.abs(self.coef))[:k]
        return [(int(i), float(self.coef[i])) for i in order]


class LimeTabular:
    """Tabular LIME with Gaussian perturbations and an exponential locality kernel.

    Perturbations are drawn in standardised space around the instance; the
    surrogate is a weighted ridge regression restricted to ``n_features``
    features chosen by forward selection (greedy residual reduction).
    """

    def __init__(self, training_data: np.ndarray, kernel_width: float | None = None,
                 n_samples: int = 2000, ridge: float = 1e-3, seed: int = 0):
        td = np.asarray(training_data, dtype=float)
        self.mu = td.mean(axis=0)
        sd = td.std(axis=0)
        self.sd = np.where(sd < 1e-12, 1.0, sd)
        self.d = td.shape[1]
        self.kernel_width = kernel_width if kernel_width else 0.75 * np.sqrt(self.d)
        self.n_samples, self.ridge, self.seed = n_samples, ridge, seed

    def kernel(self, dist: np.ndarray) -> np.ndarray:
        return np.sqrt(np.exp(-(dist ** 2) / self.kernel_width ** 2))

    @staticmethod
    def _wridge(A: np.ndarray, y: np.ndarray, w: np.ndarray, ridge: float):
        am = np.average(A, axis=0, weights=w)
        ym = np.average(y, weights=w)
        Ac, yc = A - am, y - ym
        G = Ac.T @ (w[:, None] * Ac) + ridge * np.eye(A.shape[1])
        beta = np.linalg.solve(G, Ac.T @ (w * yc))
        return beta, ym - am @ beta

    def _forward_select(self, A: np.ndarray, y: np.ndarray, w: np.ndarray, k: int) -> list[int]:
        chosen: list[int] = []
        for _ in range(min(k, A.shape[1])):
            best, best_err = -1, np.inf
            for j in range(A.shape[1]):
                if j in chosen:
                    continue
                cols = chosen + [j]
                beta, b0 = self._wridge(A[:, cols], y, w, self.ridge)
                err = np.sum(w * (y - A[:, cols] @ beta - b0) ** 2)
                if err < best_err:
                    best, best_err = j, err
            chosen.append(best)
        return chosen

    def explain(self, x: np.ndarray, f: Callable[[np.ndarray], np.ndarray],
                n_features: int | None = None) -> LimeExplanation:
        x = np.asarray(x, dtype=float)
        rng = np.random.default_rng(self.seed)
        Zs = rng.normal(size=(self.n_samples, self.d))
        Zs[0] = 0.0  # include the instance itself
        X = x + Zs * self.sd
        y = np.asarray(f(X), dtype=float)
        dist = np.linalg.norm(Zs, axis=1)
        w = self.kernel(dist)
        k = n_features or self.d
        sel = self._forward_select(Zs, y, w, k)
        beta, b0 = self._wridge(Zs[:, sel], y, w, self.ridge)
        pred = Zs[:, sel] @ beta + b0
        ss_res = np.sum(w * (y - pred) ** 2)
        ym = np.average(y, weights=w)
        ss_tot = np.sum(w * (y - ym) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
        coef = np.zeros(self.d)
        coef[sel] = beta / self.sd[sel]  # back to original units: d f / d x_i
        return LimeExplanation(coef, float(b0), np.array(sorted(sel)), float(r2), float(b0))
