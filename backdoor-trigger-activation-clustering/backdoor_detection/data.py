"""Synthetic image data and BadNets-style trigger poisoning."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

IMG = 8  # images are IMG x IMG, flattened to IMG*IMG features


def _prototypes(n_classes: int, rng: np.random.Generator) -> np.ndarray:
    """Random smooth class prototypes; the bottom-right corner is kept empty
    so a corner trigger does not coincide with class evidence."""
    protos = rng.normal(0.0, 1.0, size=(n_classes, IMG, IMG))
    # light smoothing for image-like structure
    k = np.array([0.25, 0.5, 0.25])
    for axis in (1, 2):
        protos = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), axis, protos)
    protos *= 2.5
    protos[:, -3:, -3:] = 0.0
    return protos.reshape(n_classes, -1)


def make_dataset(n_per_class: int, n_classes: int = 4, noise: float = 0.6,
                 seed: int = 0, proto_seed: int = 123) -> tuple[np.ndarray, np.ndarray]:
    """Generate (X, y). Prototypes depend only on ``proto_seed`` so that train
    and test sets share the same underlying classes."""
    protos = _prototypes(n_classes, np.random.default_rng(proto_seed))
    rng = np.random.default_rng(seed)
    X = np.concatenate([p + noise * rng.normal(size=(n_per_class, protos.shape[1]))
                        for p in protos])
    y = np.repeat(np.arange(n_classes), n_per_class)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def apply_trigger(X: np.ndarray, size: int = 3, value: float = 4.0) -> np.ndarray:
    """Stamp a square patch of ``value`` in the bottom-right corner."""
    imgs = np.array(X, dtype=float, copy=True).reshape(-1, IMG, IMG)
    imgs[:, -size:, -size:] = value
    return imgs.reshape(len(imgs), -1)


@dataclass
class PoisonedData:
    """Training set with a known poison mask (ground truth for evaluation)."""
    X: np.ndarray
    y: np.ndarray
    is_poison: np.ndarray
    target: int


def poison_dataset(X: np.ndarray, y: np.ndarray, target: int, rate: float,
                   seed: int = 0, size: int = 3, value: float = 4.0) -> PoisonedData:
    """Add triggered copies of non-target samples relabelled as ``target``.

    ``rate`` is the poison fraction of the *final* dataset. Poison samples are
    appended (clean originals are kept) which mimics an attacker injecting data.
    """
    if not 0.0 <= rate < 1.0:
        raise ValueError("rate must be in [0, 1)")
    rng = np.random.default_rng(seed)
    n_poison = int(round(rate * len(y) / (1.0 - rate)))
    cand = np.flatnonzero(y != target)
    idx = rng.choice(cand, size=n_poison, replace=n_poison > len(cand))
    Xp = apply_trigger(X[idx], size=size, value=value)
    yp = np.full(n_poison, target, dtype=y.dtype)
    Xall = np.concatenate([X, Xp])
    yall = np.concatenate([y, yp])
    mask = np.concatenate([np.zeros(len(y), bool), np.ones(n_poison, bool)])
    perm = rng.permutation(len(yall))
    return PoisonedData(Xall[perm], yall[perm], mask[perm], target)
