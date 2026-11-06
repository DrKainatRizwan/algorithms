"""Synthetic image-like dataset: smooth class prototypes on a small grid."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DatasetSpec:
    """Shape and size parameters of the synthetic dataset."""
    side: int = 8
    n_classes: int = 4
    n_samples: int = 200
    noise: float = 0.08

    @property
    def dim(self) -> int:
        return self.side * self.side


def _prototype(side: int, label: int) -> np.ndarray:
    """Deterministic stroke pattern for a class (bar, cross, ring, diagonal...)."""
    img = np.zeros((side, side))
    c = side // 2
    yy, xx = np.mgrid[0:side, 0:side]
    k = label % 4
    if k == 0:
        img[c - 1:c + 1, 1:side - 1] = 1.0
    elif k == 1:
        img[1:side - 1, c - 1:c + 1] = 1.0
    elif k == 2:
        r = np.hypot(yy - (side - 1) / 2, xx - (side - 1) / 2)
        img[(r > side / 4) & (r < side / 2.3)] = 1.0
    else:
        img[np.abs(yy - xx) <= 1] = 1.0
    if label >= 4:
        img = img[::-1]
    return img


def make_dataset(spec: DatasetSpec, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(X, y)`` with X in [0, 1], shape ``(n, side*side)``."""
    y = rng.integers(0, spec.n_classes, size=spec.n_samples)
    X = np.empty((spec.n_samples, spec.dim))
    for i, lab in enumerate(y):
        shift = rng.integers(-1, 2, size=2)
        img = np.roll(_prototype(spec.side, int(lab)), tuple(shift), axis=(0, 1))
        X[i] = (img + spec.noise * rng.standard_normal(img.shape)).ravel()
    return np.clip(X, 0.0, 1.0), y
