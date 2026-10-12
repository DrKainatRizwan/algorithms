"""End-to-end intrusion detector: log-scaling, forest scoring, alert explanation."""

from __future__ import annotations

import numpy as np

from .extended import ExtendedIsolationForest
from .forest import IsolationForest


class IntrusionDetector:
    """Fit on (mostly) benign flows, then flag and explain anomalous flows.

    Heavy-tailed counters are compressed with ``log1p`` and every column is
    standardised with robust (median/MAD) statistics learned at fit time.
    """

    def __init__(self, extended: bool = False, n_trees: int = 100, max_samples: int = 256,
                 contamination: float = 0.05, seed: int = 0,
                 feature_names: list[str] | None = None) -> None:
        cls = ExtendedIsolationForest if extended else IsolationForest
        self.model = cls(n_trees=n_trees, max_samples=max_samples, contamination=contamination, seed=seed)
        self.feature_names = feature_names
        self.center_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def _prep(self, X: np.ndarray) -> np.ndarray:
        Z = np.log1p(np.clip(np.asarray(X, dtype=float), 0, None))
        return (Z - self.center_) / self.scale_

    def fit(self, X: np.ndarray) -> "IntrusionDetector":
        Z = np.log1p(np.clip(np.asarray(X, dtype=float), 0, None))
        self.center_ = np.median(Z, axis=0)
        mad = np.median(np.abs(Z - self.center_), axis=0) * 1.4826
        self.scale_ = np.where(mad > 1e-9, mad, 1.0)
        self.model.fit(self._prep(X))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return self.model.score_samples(self._prep(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.score(X) > self.model.threshold_).astype(int)

    def explain(self, x: np.ndarray, top: int = 3) -> list[tuple[str, float]]:
        """Features with the largest robust z-score for a single flow."""
        z = self._prep(np.asarray(x, dtype=float).reshape(1, -1))[0]
        names = self.feature_names or [f"f{i}" for i in range(len(z))]
        order = np.argsort(-np.abs(z))[:top]
        return [(names[i], float(z[i])) for i in order]
