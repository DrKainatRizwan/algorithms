"""MinHash signatures for Jaccard similarity estimation."""
from __future__ import annotations

from typing import Iterable

import numpy as np

_PRIME = (1 << 31) - 1  # Mersenne prime; products of two residues fit in uint64


class MinHasher:
    """Estimate Jaccard similarity with ``num_perm`` universal hash functions.

    Each permutation is ``h_i(x) = (a_i * x + b_i) mod p``; the signature entry
    is the minimum over the set, so ``P[sig_i(A) == sig_i(B)] = J(A, B)``.
    """

    def __init__(self, num_perm: int = 128, seed: int = 0) -> None:
        if num_perm < 1:
            raise ValueError("num_perm must be positive")
        rng = np.random.default_rng(seed)
        self.num_perm = num_perm
        self.a = rng.integers(1, _PRIME, size=num_perm, dtype=np.uint64)
        self.b = rng.integers(0, _PRIME, size=num_perm, dtype=np.uint64)

    def signature(self, items: Iterable[int]) -> np.ndarray:
        """Signature (uint64 array of length ``num_perm``) of a set of ints."""
        arr = np.fromiter((int(x) % _PRIME for x in set(items)), dtype=np.uint64)
        if arr.size == 0:
            return np.full(self.num_perm, _PRIME, dtype=np.uint64)
        sig = np.full(self.num_perm, np.iinfo(np.uint64).max, dtype=np.uint64)
        # chunk the set to bound the (num_perm x chunk) intermediate matrix
        for start in range(0, arr.size, 4096):
            chunk = arr[start:start + 4096]
            h = (self.a[:, None] * chunk[None, :] + self.b[:, None]) % np.uint64(_PRIME)
            sig = np.minimum(sig, h.min(axis=1))
        return sig

    @staticmethod
    def estimate(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
        """Estimated Jaccard similarity: fraction of agreeing signature slots."""
        if sig_a.shape != sig_b.shape:
            raise ValueError("signature length mismatch")
        return float(np.mean(sig_a == sig_b))

    @staticmethod
    def merge(sig_a: np.ndarray, sig_b: np.ndarray) -> np.ndarray:
        """Signature of the union of two sets (element-wise minimum)."""
        return np.minimum(sig_a, sig_b)

    def standard_error(self, j: float) -> float:
        """Std. deviation of the estimator at true similarity ``j``."""
        return float(np.sqrt(j * (1 - j) / self.num_perm))
