"""Synthetic image-like classification worlds for privacy experiments.

Every class has a smooth ``side x side`` prototype; samples are the prototype
plus isotropic Gaussian noise.  Because the prototypes are known, model
inversion can be scored against ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _smooth(field: np.ndarray, passes: int) -> np.ndarray:
    """Repeated 3x3 box blur with edge replication."""
    f = field
    for _ in range(passes):
        p = np.pad(f, 1, mode="edge")
        f = sum(p[i : i + f.shape[0], j : j + f.shape[1]] for i in range(3) for j in range(3)) / 9.0
    return f


def make_prototypes(n_classes: int, side: int, rng: np.random.Generator, separation: float = 1.0) -> np.ndarray:
    """Return ``(n_classes, side*side)`` smooth, zero-mean, unit-scale prototypes."""
    protos = []
    for _ in range(n_classes):
        f = _smooth(rng.standard_normal((side, side)), 2)
        f = (f - f.mean()) / (f.std() + 1e-12)
        protos.append(separation * f.ravel())
    return np.stack(protos)


def sample(n: int, protos: np.ndarray, noise: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Draw ``n`` class-balanced samples."""
    k = protos.shape[0]
    y = np.arange(n) % k
    rng.shuffle(y)
    X = protos[y] + noise * rng.standard_normal((n, protos.shape[1]))
    return X, y.astype(int)


@dataclass
class World:
    """Container with member / non-member target data plus a shadow pool."""

    protos: np.ndarray
    X_in: np.ndarray
    y_in: np.ndarray
    X_out: np.ndarray
    y_out: np.ndarray
    X_shadow: np.ndarray
    y_shadow: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray

    @property
    def n_classes(self) -> int:
        return self.protos.shape[0]

    @property
    def dim(self) -> int:
        return self.protos.shape[1]


def make_world(
    seed: int = 0,
    n_classes: int = 5,
    side: int = 8,
    n_members: int = 100,
    n_nonmembers: int = 100,
    n_shadow: int = 800,
    n_test: int = 500,
    noise: float = 2.0,
    separation: float = 1.0,
) -> World:
    """Build a reproducible :class:`World`."""
    rng = np.random.default_rng(seed)
    protos = make_prototypes(n_classes, side, rng, separation)
    Xi, yi = sample(n_members, protos, noise, rng)
    Xo, yo = sample(n_nonmembers, protos, noise, rng)
    Xs, ys = sample(n_shadow, protos, noise, rng)
    Xt, yt = sample(n_test, protos, noise, rng)
    return World(protos, Xi, yi, Xo, yo, Xs, ys, Xt, yt)
