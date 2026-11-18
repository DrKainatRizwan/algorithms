"""Cross-validation and grid search for SVM hyper-parameters."""

from __future__ import annotations

from itertools import product
from typing import Callable

import numpy as np

from .smo import SVC

Array = np.ndarray


def kfold_indices(n: int, k: int = 5, seed: int = 0) -> list[tuple[Array, Array]]:
    """Return ``k`` shuffled (train, validation) index pairs."""
    if not 2 <= k <= n:
        raise ValueError("need 2 <= k <= n")
    perm = np.random.default_rng(seed).permutation(n)
    folds = np.array_split(perm, k)
    return [(np.concatenate(folds[:i] + folds[i + 1:]), folds[i]) for i in range(k)]


def cross_val_score(make_model: Callable[[], SVC], X: Array, y: Array, k: int = 5, seed: int = 0) -> Array:
    """Per-fold accuracy of freshly built models."""
    X, y = np.asarray(X), np.asarray(y)
    return np.array([make_model().fit(X[tr], y[tr]).score(X[va], y[va])
                     for tr, va in kfold_indices(len(X), k, seed)])


def grid_search(X: Array, y: Array, grid: dict[str, list], k: int = 3, seed: int = 0,
                **fixed) -> tuple[dict, float, list[tuple[dict, float]]]:
    """Exhaustive search over ``grid``; returns (best params, best score, all results)."""
    keys = list(grid)
    results = []
    for combo in product(*(grid[key] for key in keys)):
        params = dict(zip(keys, combo))
        score = float(cross_val_score(lambda p=params: SVC(**fixed, **p), X, y, k, seed).mean())
        results.append((params, score))
    best = max(results, key=lambda r: r[1])
    return best[0], best[1], results
