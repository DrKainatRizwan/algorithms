"""Output-side defences: wrappers that post-process a model's probability vector.

All wrappers expose ``predict_proba`` / ``predict`` so attacks can treat a
raw model and a defended API identically.  Training-time defences (L2, early
stopping, DP-SGD) live in :class:`privacy_attacks.models.TrainConfig`.
"""
from __future__ import annotations

import zlib
from typing import Protocol

import numpy as np


class Predictor(Protocol):
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


class _Wrapper:
    def __init__(self, base: Predictor) -> None:
        self.base = base

    def predict_proba(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def loss(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        p = np.clip(self.predict_proba(X)[np.arange(len(y)), y], 1e-12, 1.0)
        return -np.log(p)


class LabelOnly(_Wrapper):
    """Release only the predicted label (as a one-hot vector)."""

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = self.base.predict_proba(X)
        out = np.zeros_like(p)
        out[np.arange(len(p)), p.argmax(axis=1)] = 1.0
        return out


class TopK(_Wrapper):
    """Keep only the ``k`` largest probabilities (renormalised), zero the rest."""

    def __init__(self, base: Predictor, k: int = 1) -> None:
        super().__init__(base)
        self.k = k

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = self.base.predict_proba(X)
        keep = np.argsort(-p, axis=1)[:, : self.k]
        out = np.zeros_like(p)
        rows = np.arange(len(p))[:, None]
        out[rows, keep] = p[rows, keep]
        return out / out.sum(axis=1, keepdims=True)


class Rounding(_Wrapper):
    """Round probabilities to a fixed number of decimals."""

    def __init__(self, base: Predictor, decimals: int = 1) -> None:
        super().__init__(base)
        self.decimals = decimals

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = np.round(self.base.predict_proba(X), self.decimals)
        s = p.sum(axis=1, keepdims=True)
        return np.where(s > 0, p / np.where(s > 0, s, 1.0), 1.0 / p.shape[1])


class Temperature(_Wrapper):
    """Soften (T>1) or sharpen (T<1) outputs; requires access to logits."""

    def __init__(self, base, temperature: float = 2.0) -> None:
        super().__init__(base)
        self.temperature = temperature

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = self.base.logits(X) / self.temperature
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)


class OutputNoise(_Wrapper):
    """Add i.i.d. Laplace noise to every probability, clip and renormalise.

    Noise is seeded per call so repeated identical queries are *not*
    averaged away only if ``fresh=True``; with ``fresh=False`` the noise is a
    deterministic function of the input (cannot be averaged out).
    """

    def __init__(self, base: Predictor, scale: float = 0.1, seed: int = 0, fresh: bool = True) -> None:
        super().__init__(base)
        self.scale, self.fresh = scale, fresh
        self.rng = np.random.default_rng(seed)
        self.seed = seed

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = self.base.predict_proba(X)
        if self.fresh:
            noise = self.rng.laplace(0.0, self.scale, p.shape)
        else:
            noise = np.stack([
                np.random.default_rng(zlib.crc32(row.tobytes()) ^ self.seed).laplace(0.0, self.scale, p.shape[1])
                for row in np.ascontiguousarray(np.atleast_2d(X))
            ])
        q = np.clip(p + noise, 1e-6, None)
        return q / q.sum(axis=1, keepdims=True)
