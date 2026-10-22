"""Secret-key rate bounds and Toeplitz-hash privacy amplification."""

from __future__ import annotations

import math

import numpy as np


def binary_entropy(p: float) -> float:
    """Shannon entropy h(p) in bits."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def secret_fraction(qber: float) -> float:
    """Asymptotic Shor-Preskill rate ``max(0, 1 - 2 h(e))``."""
    return max(0.0, 1.0 - 2.0 * binary_entropy(qber))


def secure_key_length(n: int, qber: float, leaked: int, eps: float = 1e-6) -> int:
    """Number of key bits that can be distilled from ``n`` reconciled bits.

    ``n * (1 - h(e_u))`` minus the bits disclosed during reconciliation and a
    finite-size security term ``2 log2(1/eps)``, where ``e_u`` is the QBER
    plus a statistical (Hoeffding) margin.
    """
    if n <= 0:
        return 0
    e_u = min(0.5, qber + math.sqrt(math.log(1 / eps) / (2 * n)))
    m = n * (1.0 - binary_entropy(e_u)) - leaked - 2.0 * math.log2(1.0 / eps)
    return max(0, int(math.floor(m)))


def toeplitz_hash(bits: np.ndarray, m: int, seed: int) -> np.ndarray:
    """Hash ``bits`` to ``m`` bits with a random Toeplitz matrix over GF(2).

    The matrix is defined by ``n + m - 1`` random bits derived from ``seed``,
    forming a 2-universal family.
    """
    bits = np.asarray(bits, dtype=np.int64)
    n = bits.size
    if m <= 0:
        return np.zeros(0, dtype=np.int64)
    r = np.random.default_rng(seed).integers(0, 2, n + m - 1)
    out = np.empty(m, dtype=np.int64)
    for i in range(m):
        # Row i is r[i : i + n] reversed so that T[i, j] = r[i - j + n - 1].
        row = r[i:i + n][::-1]
        out[i] = int(np.dot(row, bits) & 1)
    return out


def privacy_amplify(bits: np.ndarray, m: int, seed: int) -> np.ndarray:
    """Compress a partially known key to a shorter, near-uniform one."""
    return toeplitz_hash(bits, min(m, bits.size), seed)
