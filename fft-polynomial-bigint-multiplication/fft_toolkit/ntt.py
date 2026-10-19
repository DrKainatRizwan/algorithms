"""Number-theoretic transform and exact convolution via the CRT.

The NTT is an FFT over the finite field Z_p for an NTT-friendly prime
``p = c * 2^k + 1``.  Three such primes are combined with Garner's algorithm
to recover exact convolutions of integers up to roughly 7.8e25 in magnitude.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .fft import bit_reversal_permutation, is_power_of_two, next_power_of_two

PRIMES: tuple[tuple[int, int], ...] = (
    (998244353, 3),   # 119 * 2^23 + 1
    (167772161, 3),   # 5 * 2^25 + 1
    (469762049, 3),   # 7 * 2^26 + 1
)
DEFAULT_MOD, DEFAULT_ROOT = PRIMES[0]
CRT_LIMIT = PRIMES[0][0] * PRIMES[1][0] * PRIMES[2][0]


@lru_cache(maxsize=64)
def _twiddles(n: int, mod: int, root: int, inverse: bool) -> np.ndarray:
    """Powers ``w^0 .. w^(n/2-1)`` of a primitive n-th root of unity."""
    if (mod - 1) % n:
        raise ValueError(f"length {n} does not divide p-1 for p={mod}")
    w = pow(root, (mod - 1) // n, mod)
    if inverse:
        w = pow(w, mod - 2, mod)
    tw = np.ones(1, dtype=np.int64)
    while len(tw) < n // 2:
        step = pow(w, len(tw), mod)
        tw = np.concatenate([tw, tw * step % mod])
    return tw[: max(n // 2, 1)]


def ntt(a: np.ndarray, mod: int = DEFAULT_MOD, root: int = DEFAULT_ROOT,
        inverse: bool = False) -> np.ndarray:
    """Iterative radix-2 NTT of a power-of-two length int64 array."""
    a = np.asarray(a, dtype=np.int64) % mod
    n = a.shape[0]
    if not is_power_of_two(n):
        raise ValueError("length must be a power of two")
    if n == 1:
        return a.copy()
    a = a[bit_reversal_permutation(n)]
    tw = _twiddles(n, mod, root, inverse)
    length = 2
    while length <= n:
        half = length // 2
        w = tw[:: n // length][:half]
        blocks = a.reshape(n // length, 2, half)
        u = blocks[:, 0, :]
        v = blocks[:, 1, :] * w % mod
        a = np.stack([(u + v) % mod, (u - v) % mod], axis=1).reshape(n)
        length *= 2
    if inverse:
        a = a * pow(n, mod - 2, mod) % mod
    return a


def convolve_mod_prime(a, b, mod: int = DEFAULT_MOD, root: int = DEFAULT_ROOT) -> np.ndarray:
    """Linear convolution of two integer sequences modulo an NTT prime."""
    a = np.array([int(v) % mod for v in a], dtype=np.int64)
    b = np.array([int(v) % mod for v in b], dtype=np.int64)
    if a.size == 0 or b.size == 0:
        return np.zeros(0, dtype=np.int64)
    out_len = a.size + b.size - 1
    n = next_power_of_two(out_len)
    fa = ntt(np.pad(a % mod, (0, n - a.size)), mod, root)
    fb = ntt(np.pad(b % mod, (0, n - b.size)), mod, root)
    return ntt(fa * fb % mod, mod, root, inverse=True)[:out_len]


def garner(residues: list[int], moduli: list[int]) -> int:
    """Chinese remainder reconstruction in mixed-radix form (Garner)."""
    coeffs: list[int] = []
    for i, (r, m) in enumerate(zip(residues, moduli)):
        t = r
        for j in range(i):
            t = (t - coeffs[j]) * pow(moduli[j], -1, m) % m
        coeffs.append(t)
    value, prod = 0, 1
    for c, m in zip(coeffs, moduli):
        value += c * prod
        prod *= m
    return value


def convolve_exact(a, b) -> list[int]:
    """Exact integer convolution (signed allowed) using three NTT primes.

    Correct whenever every output coefficient has magnitude below ``M/2`` with
    ``M ~ 7.8e25`` (the product of the three primes).
    """
    a = [int(v) for v in a]
    b = [int(v) for v in b]
    if not a or not b:
        return []
    mods = [p for p, _ in PRIMES]
    if max(abs(v) for v in a) * max(abs(v) for v in b) * min(len(a), len(b)) >= CRT_LIMIT // 2:
        raise OverflowError("coefficients too large for three-prime reconstruction")
    res = [convolve_mod_prime(a, b, p, g).tolist() for p, g in PRIMES]
    big = mods[0] * mods[1] * mods[2]
    out = []
    for r in zip(*res):
        v = garner(list(r), mods)
        out.append(v - big if v > big // 2 else v)
    return out


def convolve_mod(a, b, mod: int) -> list[int]:
    """Convolution modulo an arbitrary ``mod`` (< 2^31) via three-prime CRT."""
    a = [int(v) % mod for v in a]
    b = [int(v) % mod for v in b]
    return [v % mod for v in convolve_exact(a, b)]
