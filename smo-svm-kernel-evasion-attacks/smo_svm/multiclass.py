"""Multiclass wrappers around the binary :class:`SVC`."""

from __future__ import annotations

from itertools import combinations

import numpy as np

from .smo import SVC

Array = np.ndarray


class OneVsRestSVC:
    """One binary SVM per class; prediction is the arg-max decision value."""

    def __init__(self, **svc_params) -> None:
        self.svc_params = svc_params

    def fit(self, X: Array, y: Array) -> "OneVsRestSVC":
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.models_ = [SVC(**self.svc_params).fit(X, (y == c).astype(int)) for c in self.classes_]
        return self

    def decision_function(self, X: Array) -> Array:
        return np.column_stack([m.decision_function(X) for m in self.models_])

    def predict(self, X: Array) -> Array:
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]

    def score(self, X: Array, y: Array) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))


class OneVsOneSVC:
    """One SVM per pair of classes; prediction by majority vote (ties by score)."""

    def __init__(self, **svc_params) -> None:
        self.svc_params = svc_params

    def fit(self, X: Array, y: Array) -> "OneVsOneSVC":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.models_: dict[tuple[int, int], SVC] = {}
        for i, j in combinations(range(len(self.classes_)), 2):
            mask = (y == self.classes_[i]) | (y == self.classes_[j])
            self.models_[(i, j)] = SVC(**self.svc_params).fit(
                X[mask], (y[mask] == self.classes_[j]).astype(int))
        return self

    def predict(self, X: Array) -> Array:
        X = np.atleast_2d(X)
        votes = np.zeros((len(X), len(self.classes_)))
        conf = np.zeros_like(votes)
        for (i, j), m in self.models_.items():
            d = m.decision_function(X)
            win_j = d >= 0
            votes[win_j, j] += 1
            votes[~win_j, i] += 1
            conf[:, j] += d
            conf[:, i] -= d
        return self.classes_[np.argmax(votes + 1e-6 * conf, axis=1)]

    def score(self, X: Array, y: Array) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))
