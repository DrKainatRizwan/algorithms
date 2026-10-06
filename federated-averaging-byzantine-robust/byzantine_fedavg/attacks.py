"""Byzantine attacks that craft malicious client updates.

Each attack takes the honest updates ``(h, dim)`` (what an omniscient
adversary can see) and returns ``(n_byz, dim)`` malicious updates.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def sign_flip_attack(honest: np.ndarray, n_byz: int,
                     rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
    """Send the negated honest mean."""
    return np.tile(-scale * honest.mean(axis=0), (n_byz, 1))


def scaled_negative_attack(honest: np.ndarray, n_byz: int,
                           rng: np.random.Generator, scale: float = 10.0) -> np.ndarray:
    """Send a heavily amplified opposite of the honest mean."""
    return sign_flip_attack(honest, n_byz, rng, scale=scale)


def gaussian_attack(honest: np.ndarray, n_byz: int,
                    rng: np.random.Generator, std: float = 5.0) -> np.ndarray:
    """Send isotropic Gaussian noise."""
    return rng.normal(0.0, std, size=(n_byz, honest.shape[1]))


def little_is_enough(honest: np.ndarray, n_byz: int,
                     rng: np.random.Generator, z: float = 1.5) -> np.ndarray:
    """Shift the mean by ``z`` std-devs per coordinate (Baruch et al., 2019).

    The perturbation is small enough to evade distance-based filters.
    """
    mu = honest.mean(axis=0)
    sd = honest.std(axis=0)
    return np.tile(mu - z * sd, (n_byz, 1))


ATTACKS: dict[str, Callable[..., np.ndarray]] = {
    "sign_flip": sign_flip_attack,
    "scaled_negative": scaled_negative_attack,
    "gaussian": gaussian_attack,
    "lie": little_is_enough,
}
