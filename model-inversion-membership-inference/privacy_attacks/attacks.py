"""Membership-inference attacks.

Every attack exposes ``score(target, X, y) -> ndarray`` where a larger value
means "more likely a training member", so all attacks share the metrics in
:mod:`privacy_attacks.metrics`.

Implemented:
  * loss / confidence / entropy / modified-entropy threshold scores
    (Yeom et al. 2018, Salem et al. 2019, Song & Mittal 2021),
  * shadow-model attack with a logistic attack classifier (Shokri et al. 2017),
  * difficulty-calibrated reference attack (Watson et al. 2022, Ye et al. 2022),
  * label-only perturbation-robustness attack (Choquette-Choo et al. 2021).
"""
from __future__ import annotations

import numpy as np

from .models import MLP, TrainConfig, train

EPS = 1e-12


# ---------------------------------------------------------------- scores
def true_class_prob(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    return P[np.arange(len(y)), y]


def loss_score(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Negative cross-entropy: members tend to have lower loss."""
    return np.log(np.clip(true_class_prob(P, y), EPS, 1.0))


def confidence_score(P: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
    return P.max(axis=1)


def entropy_score(P: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
    """Negative predictive entropy."""
    return (P * np.log(np.clip(P, EPS, 1.0))).sum(axis=1)


def modified_entropy_score(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Negative Song-Mittal modified entropy, which uses the true label.

    ``-(1-p_y) log p_y - sum_{k != y} p_k log(1-p_k)``.
    """
    Pc = np.clip(P, EPS, 1 - EPS)
    n = len(y)
    py = Pc[np.arange(n), y]
    mod = -(1 - py) * np.log(py)
    other = -(Pc * np.log(1 - Pc)).sum(axis=1) + py * np.log(1 - py)
    return -(mod + other)


SCORERS = {
    "loss": loss_score,
    "confidence": confidence_score,
    "entropy": entropy_score,
    "mod_entropy": modified_entropy_score,
}


def score_attack(target, X: np.ndarray, y: np.ndarray, kind: str = "loss") -> np.ndarray:
    """Threshold-style attack score computed from the target's API outputs."""
    return SCORERS[kind](target.predict_proba(X), y)


# ---------------------------------------------------------- shadow models
def train_shadow_models(
    X: np.ndarray, y: np.ndarray, n_classes: int, n_shadows: int, set_size: int,
    hidden: tuple[int, ...], cfg: TrainConfig, seed: int = 0,
) -> list[tuple[MLP, np.ndarray]]:
    """Train shadow models on random subsets; return ``(model, member_index_mask)``."""
    rng = np.random.default_rng(seed)
    out = []
    for s in range(n_shadows):
        idx = rng.choice(len(y), size=set_size, replace=False)
        m = MLP(X.shape[1], n_classes, hidden, seed=seed * 1000 + s)
        c = TrainConfig(**{**cfg.__dict__, "seed": seed * 1000 + s})
        train(m, X[idx], y[idx], c)
        mask = np.zeros(len(y), dtype=bool)
        mask[idx] = True
        out.append((m, mask))
    return out


def _features(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Attack features: logit-scaled true prob, entropy, top-1 prob, correctness."""
    py = np.clip(true_class_prob(P, y), 1e-6, 1 - 1e-6)
    return np.stack([
        np.log(py / (1 - py)),
        -entropy_score(P),
        P.max(axis=1),
        (P.argmax(axis=1) == y).astype(float),
    ], axis=1)


class LogisticRegression:
    """Tiny binary logistic regression (standardised inputs, L2, Newton-free GD)."""

    def __init__(self, l2: float = 1e-2, lr: float = 0.1, iters: int = 500) -> None:
        self.l2, self.lr, self.iters = l2, lr, iters

    def fit(self, X: np.ndarray, t: np.ndarray) -> "LogisticRegression":
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = (X - self.mu) / self.sd
        self.w, self.b = np.zeros(Z.shape[1]), 0.0
        for _ in range(self.iters):
            p = 1 / (1 + np.exp(-(Z @ self.w + self.b)))
            self.w -= self.lr * (Z.T @ (p - t) / len(t) + self.l2 * self.w)
            self.b -= self.lr * (p - t).mean()
        return self

    def decision(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mu) / self.sd) @ self.w + self.b


class ShadowAttack:
    """Shokri-style attack: learn ``member vs non-member`` from shadow models."""

    def __init__(self, n_shadows: int = 6, set_size: int = 100, hidden: tuple[int, ...] = (64,),
                 cfg: TrainConfig | None = None, seed: int = 0) -> None:
        self.n_shadows, self.set_size, self.hidden = n_shadows, set_size, hidden
        self.cfg = cfg or TrainConfig()
        self.seed = seed
        self.shadows: list[tuple[MLP, np.ndarray]] = []
        self.clf: LogisticRegression | None = None

    def fit(self, X_pool: np.ndarray, y_pool: np.ndarray, n_classes: int) -> "ShadowAttack":
        self.shadows = train_shadow_models(X_pool, y_pool, n_classes, self.n_shadows, self.set_size,
                                           self.hidden, self.cfg, self.seed)
        feats, labels = [], []
        for m, mask in self.shadows:
            P = m.predict_proba(X_pool)
            F = _features(P, y_pool)
            n_in = int(mask.sum())
            out_idx = np.flatnonzero(~mask)[:n_in]  # balanced
            feats += [F[mask], F[out_idx]]
            labels += [np.ones(n_in), np.zeros(len(out_idx))]
        self.clf = LogisticRegression().fit(np.concatenate(feats), np.concatenate(labels))
        return self

    def score(self, target, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.clf is None:
            raise RuntimeError("call fit() first")
        return self.clf.decision(_features(target.predict_proba(X), y))


class CalibratedAttack:
    """Difficulty-calibrated loss attack.

    Score = (mean shadow loss on the example, over shadows that did *not* train
    on it) - (target loss).  Intrinsically hard examples have high reference
    loss, which cancels their naturally high target loss.
    """

    def __init__(self, shadow_models: list[tuple[MLP, np.ndarray]]) -> None:
        self.shadows = shadow_models

    def score(self, target, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        ref = np.mean([m.loss(X, y) for m, _ in self.shadows], axis=0)
        t_loss = -loss_score(target.predict_proba(X), y)
        return ref - t_loss


class LabelOnlyAttack:
    """Fraction of noisy copies of ``x`` still classified as ``y``.

    Needs only hard labels, so it still works when confidences are hidden.
    Members lie farther from the decision boundary.
    """

    def __init__(self, n_queries: int = 30, sigma: float = 1.0, seed: int = 0) -> None:
        self.n_queries, self.sigma, self.seed = n_queries, sigma, seed

    def score(self, target, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        hits = np.zeros(len(y))
        for _ in range(self.n_queries):
            Xn = X + self.sigma * rng.standard_normal(X.shape)
            hits += (target.predict_proba(Xn).argmax(axis=1) == y)
        base = (target.predict_proba(X).argmax(axis=1) == y).astype(float)
        return (hits + base) / (self.n_queries + 1)
