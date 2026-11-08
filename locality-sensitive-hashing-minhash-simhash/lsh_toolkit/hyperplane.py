"""Vector LSH families: random hyperplanes (cosine) and p-stable (Euclidean)."""
from __future__ import annotations

import numpy as np


class HyperplaneHasher:
    """Sign random projections: ``P[h(x)=h(y)] = 1 - angle(x,y)/pi`` per bit."""

    def __init__(self, dim: int, num_bits: int, seed: int = 0) -> None:
        self.dim, self.num_bits = dim, num_bits
        self.planes = np.random.default_rng(seed).standard_normal((num_bits, dim))

    def bits(self, x: np.ndarray) -> np.ndarray:
        """Boolean hash bits for one vector (dim,) or a batch (n, dim)."""
        return (np.asarray(x) @ self.planes.T) > 0

    def keys(self, x: np.ndarray) -> list[bytes] | bytes:
        """Hashable bucket keys (packed bits) for a vector or batch."""
        b = self.bits(x)
        if b.ndim == 1:
            return np.packbits(b).tobytes()
        return [np.packbits(row).tobytes() for row in b]

    @staticmethod
    def bit_collision_probability(x: np.ndarray, y: np.ndarray) -> float:
        """Theoretical per-bit collision probability ``1 - theta/pi``."""
        c = float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))
        return 1.0 - float(np.arccos(np.clip(c, -1, 1))) / np.pi

    def estimate_cosine(self, x: np.ndarray, y: np.ndarray) -> float:
        """Cosine estimate from the Hamming distance of the two codes."""
        frac = float(np.mean(self.bits(x) != self.bits(y)))
        return float(np.cos(np.pi * frac))


class PStableHasher:
    """Gaussian (2-stable) LSH for L2: ``h(x) = floor((a.x + b) / w)``."""

    def __init__(self, dim: int, num_hashes: int, width: float = 4.0, seed: int = 0) -> None:
        if width <= 0:
            raise ValueError("width must be positive")
        rng = np.random.default_rng(seed)
        self.width = width
        self.a = rng.standard_normal((num_hashes, dim))
        self.b = rng.uniform(0, width, size=num_hashes)

    def codes(self, x: np.ndarray) -> np.ndarray:
        """Integer hash codes for a vector (m,) or batch (n, m)."""
        return np.floor((np.asarray(x) @ self.a.T + self.b) / self.width).astype(np.int64)

    def keys(self, x: np.ndarray) -> list[bytes] | bytes:
        c = self.codes(x)
        if c.ndim == 1:
            return c.tobytes()
        return [row.tobytes() for row in c]

    def collision_probability(self, dist: float) -> float:
        """Per-hash collision probability at L2 distance ``dist`` (Datar et al.)."""
        if dist <= 0:
            return 1.0
        from math import erf, exp, pi, sqrt
        r = self.width / dist
        cdf = 0.5 * (1 + erf((-r) / sqrt(2)))  # Phi(-r)
        return 1 - 2 * cdf - 2 / (sqrt(2 * pi) * r) * (1 - exp(-r * r / 2))
