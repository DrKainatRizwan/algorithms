"""Fixed-point encoding of real vectors into the ring ``Z_{2**bits}``.

Secure aggregation sums masked integers modulo ``2**bits``.  Real-valued model
updates are therefore clipped to a known L2 radius, scaled, rounded and mapped
into the ring using two's-complement style wrap-around so that negative values
survive the modular sum.
"""

from __future__ import annotations

import numpy as np


def clip_l2(x: np.ndarray, bound: float) -> np.ndarray:
    """Scale ``x`` down so that its L2 norm is at most ``bound``."""
    norm = float(np.linalg.norm(x))
    if norm <= bound or norm == 0.0:
        return x.copy()
    return x * (bound / norm)


def max_safe_clients(clip: float, scale: float, bits: int = 32) -> int:
    """Largest number of summands guaranteed not to overflow the ring.

    Each coordinate of a clipped vector has magnitude at most ``clip``; the
    signed sum must stay below ``2**(bits-1)``.
    """
    return int((2 ** (bits - 1) - 1) // (np.ceil(clip * scale) + 1))


def encode(x: np.ndarray, scale: float, bits: int = 32) -> np.ndarray:
    """Round ``x * scale`` to integers and wrap them into ``[0, 2**bits)``."""
    mod = 1 << bits
    q = np.rint(np.asarray(x, dtype=np.float64) * scale).astype(np.int64)
    if np.any(np.abs(q) >= mod // 2):
        raise OverflowError("value does not fit in the ring; reduce scale/clip")
    return (q % mod).astype(np.uint64)


def decode(v: np.ndarray, scale: float, bits: int = 32) -> np.ndarray:
    """Inverse of :func:`encode` for a (possibly summed) ring element."""
    mod = 1 << bits
    s = v.astype(np.int64) % mod
    s = np.where(s >= mod // 2, s - mod, s)
    return s.astype(np.float64) / scale


def ring_add(a: np.ndarray, b: np.ndarray, bits: int = 32) -> np.ndarray:
    """Element-wise addition modulo ``2**bits``."""
    return (a + b) & np.uint64((1 << bits) - 1)


def ring_sub(a: np.ndarray, b: np.ndarray, bits: int = 32) -> np.ndarray:
    """Element-wise subtraction modulo ``2**bits``."""
    mod = np.uint64(1 << bits)
    return (a + mod - b) & (mod - np.uint64(1))
