"""Training-set poisoning attacks.

* :func:`random_label_flip` - baseline, flips uniformly random labels.
* :func:`boundary_flip` - flips the points closest to the clean decision boundary.
* :func:`influence_label_flip` - greedy flips ranked by first-order validation-loss gain.
* :func:`gradient_poisoning` - injects new points optimised by projected gradient
  ascent on validation loss through the retraining map (Biggio et al., 2012).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .influence import flip_influence, poison_input_gradient
from .model import LogisticRegression


@dataclass
class AttackResult:
    """A poisoned training set plus bookkeeping."""

    X: np.ndarray
    y: np.ndarray
    poisoned: np.ndarray  # boolean mask over rows of X: True for corrupted / injected samples
    history: list[float] = field(default_factory=list)

    @property
    def n_poisoned(self) -> int:
        return int(self.poisoned.sum())


def _budget(n: int, frac: float) -> int:
    return int(round(frac * n))


def random_label_flip(X: np.ndarray, y: np.ndarray, frac: float,
                      rng: np.random.Generator) -> AttackResult:
    """Flip the labels of a random ``frac`` of the training points."""
    k = _budget(len(y), frac)
    idx = rng.choice(len(y), size=k, replace=False)
    y2 = y.copy()
    y2[idx] = 1 - y2[idx]
    mask = np.zeros(len(y), dtype=bool)
    mask[idx] = True
    return AttackResult(X.copy(), y2, mask)


def boundary_flip(X: np.ndarray, y: np.ndarray, frac: float, l2: float = 1.0) -> AttackResult:
    """Flip the points with the smallest margin under the clean model."""
    k = _budget(len(y), frac)
    model = LogisticRegression(l2).fit(X, y)
    idx = np.argsort(np.abs(model.decision_function(X)))[:k]
    y2 = y.copy()
    y2[idx] = 1 - y2[idx]
    mask = np.zeros(len(y), dtype=bool)
    mask[idx] = True
    return AttackResult(X.copy(), y2, mask)


def influence_label_flip(X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
                         frac: float, l2: float = 1.0, rounds: int = 4) -> AttackResult:
    """Greedy label-flip attack using first-order influence scores.

    The budget is spent in ``rounds`` batches; the model is refitted on the
    partially flipped data between batches so the linearisation stays accurate.
    Already flipped points are never selected again.
    """
    n = len(y)
    k = _budget(n, frac)
    y2 = y.copy()
    mask = np.zeros(n, dtype=bool)
    history: list[float] = []
    remaining = k
    for r in range(rounds):
        if remaining <= 0:
            break
        take = int(np.ceil(remaining / (rounds - r)))
        model = LogisticRegression(l2).fit(X, y2)
        history.append(model.loss(X_val, y_val, reg=False))
        scores = flip_influence(model, X, y2, X_val, y_val)
        scores[mask] = -np.inf
        idx = np.argsort(-scores)[:take]
        y2[idx] = 1 - y2[idx]
        mask[idx] = True
        remaining -= take
    final = LogisticRegression(l2).fit(X, y2)
    history.append(final.loss(X_val, y_val, reg=False))
    return AttackResult(X.copy(), y2, mask, history)


def gradient_poisoning(X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
                       frac: float, rng: np.random.Generator, l2: float = 1.0,
                       steps: int = 40, step_size: float = 0.4, margin: float = 1.0
                       ) -> AttackResult:
    """Inject ``frac * n`` new points optimised to maximise validation loss.

    Each poison point starts as a label-flipped copy of a random training
    point.  At every iteration the model is refitted on clean + poison data and
    each poison point takes a normalised gradient-ascent step, projected back
    to the (slightly enlarged) bounding box of the clean features.  The best
    iterate (largest validation loss) is returned.
    """
    n, d = X.shape
    k = _budget(n, frac)
    if k == 0:
        return AttackResult(X.copy(), y.copy(), np.zeros(n, dtype=bool))
    lo, hi = X.min(axis=0) - margin, X.max(axis=0) + margin
    src = rng.choice(n, size=k, replace=False)
    Xp = X[src].copy()
    yp = 1 - y[src]
    best_Xp, best_loss = Xp.copy(), -np.inf
    history: list[float] = []
    model = LogisticRegression(l2)
    w0 = None
    for it in range(steps):
        X_all = np.vstack([X, Xp])
        y_all = np.concatenate([y, yp])
        model.fit(X_all, y_all, w0=w0)
        w0 = model.w.copy()
        val = model.loss(X_val, y_val, reg=False)
        history.append(val)
        if val > best_loss:
            best_loss, best_Xp = val, Xp.copy()
        lr = step_size * (1.0 - 0.8 * it / max(1, steps - 1))
        for j in range(k):
            g = poison_input_gradient(model, X_all, Xp[j], float(yp[j]), X_val, y_val)
            norm = np.linalg.norm(g)
            if norm > 1e-12:
                Xp[j] = np.clip(Xp[j] + lr * g / norm, lo, hi)
    X_all = np.vstack([X, Xp])
    model.fit(X_all, np.concatenate([y, yp]), w0=w0)
    val = model.loss(X_val, y_val, reg=False)
    history.append(val)
    if val > best_loss:
        best_Xp = Xp.copy()
    mask = np.concatenate([np.zeros(n, dtype=bool), np.ones(k, dtype=bool)])
    return AttackResult(np.vstack([X, best_Xp]), np.concatenate([y, yp]), mask, history)
