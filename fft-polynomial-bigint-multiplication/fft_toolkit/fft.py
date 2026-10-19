"""Complex FFT routines written from scratch on top of NumPy array arithmetic.

Provides a naive DFT, a recursive radix-2 transform, an iterative
(bit-reversal, stage-vectorised) radix-2 transform, Bluestein's chirp-z
algorithm for arbitrary lengths, and helpers for real input.  All transforms
use the convention ``X_k = sum_n x_n exp(-2*pi*i*k*n/N)``.
"""
from __future__ import annotations

import numpy as np


def is_power_of_two(n: int) -> bool:
    """Return True when ``n`` is a positive power of two."""
    return n > 0 and (n & (n - 1)) == 0


def next_power_of_two(n: int) -> int:
    """Smallest power of two that is >= ``n`` (and >= 1)."""
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def bit_reversal_permutation(n: int) -> np.ndarray:
    """Index array ``p`` with ``p[i]`` equal to ``i`` with its bits reversed."""
    if not is_power_of_two(n):
        raise ValueError("n must be a power of two")
    bits = n.bit_length() - 1
    rev = np.zeros(n, dtype=np.int64)
    for b in range(bits):
        rev |= ((np.arange(n) >> b) & 1) << (bits - 1 - b)
    return rev


def dft_naive(x: np.ndarray, inverse: bool = False) -> np.ndarray:
    """O(N^2) discrete Fourier transform, used as a reference implementation."""
    x = np.asarray(x, dtype=np.complex128)
    n = x.shape[-1]
    k = np.arange(n)
    sign = 1 if inverse else -1
    # reduce k*j modulo n first so the angle stays small and accurate
    mat = np.exp(sign * 2j * np.pi * ((np.outer(k, k) % n) / n))
    out = x @ mat.T
    return out / n if inverse else out


def fft_recursive(x: np.ndarray, inverse: bool = False) -> np.ndarray:
    """Cooley-Tukey decimation-in-time FFT for 1-D input of power-of-two length."""
    x = np.asarray(x, dtype=np.complex128)
    n = x.shape[0]
    if not is_power_of_two(n):
        raise ValueError("length must be a power of two")
    out = _rec(x, -1.0 if not inverse else 1.0)
    return out / n if inverse else out


def _rec(x: np.ndarray, sign: float) -> np.ndarray:
    n = x.shape[0]
    if n == 1:
        return x.copy()
    even = _rec(x[0::2], sign)
    odd = _rec(x[1::2], sign)
    tw = np.exp(sign * 2j * np.pi * np.arange(n // 2) / n) * odd
    return np.concatenate([even + tw, even - tw])


def fft_iterative(x: np.ndarray, inverse: bool = False) -> np.ndarray:
    """In-order radix-2 FFT along the last axis (supports batched input)."""
    a = np.asarray(x, dtype=np.complex128)
    n = a.shape[-1]
    if not is_power_of_two(n):
        raise ValueError("length must be a power of two")
    lead = a.shape[:-1]
    a = a[..., bit_reversal_permutation(n)]
    sign = 1.0 if inverse else -1.0
    length = 2
    while length <= n:
        half = length // 2
        w = np.exp(sign * 2j * np.pi * np.arange(half) / length)
        blocks = a.reshape(lead + (n // length, 2, half))
        u = blocks[..., 0, :]
        v = blocks[..., 1, :] * w
        a = np.stack([u + v, u - v], axis=-2).reshape(lead + (n,))
        length *= 2
    return a / n if inverse else a


def bluestein(x: np.ndarray, inverse: bool = False) -> np.ndarray:
    """Chirp-z FFT for arbitrary 1-D length ``N`` in O(N log N).

    Uses ``nk = (n^2 + k^2 - (k-n)^2) / 2`` to turn the DFT into a linear
    convolution with a chirp, evaluated with power-of-two FFTs.
    """
    x = np.asarray(x, dtype=np.complex128)
    n = x.shape[0]
    if n == 0:
        return x.copy()
    sign = 1.0 if inverse else -1.0
    idx = np.arange(n, dtype=np.int64)
    # n^2 mod 2N keeps the chirp angle exact for large N
    chirp = np.exp(sign * 1j * np.pi * ((idx * idx) % (2 * n)) / n)
    m = next_power_of_two(2 * n - 1)
    a = np.zeros(m, dtype=np.complex128)
    a[:n] = x * chirp
    b = np.zeros(m, dtype=np.complex128)
    b[:n] = np.conj(chirp)
    b[m - n + 1:] = np.conj(chirp[1:][::-1])
    conv = fft_iterative(fft_iterative(a) * fft_iterative(b), inverse=True)
    out = conv[:n] * chirp
    return out / n if inverse else out


def fft(x: np.ndarray, inverse: bool = False) -> np.ndarray:
    """FFT of any length along the last axis (Bluestein if not a power of two)."""
    a = np.asarray(x, dtype=np.complex128)
    n = a.shape[-1]
    if is_power_of_two(n):
        return fft_iterative(a, inverse)
    flat = a.reshape(-1, n)
    out = np.stack([bluestein(row, inverse) for row in flat]) if len(flat) else flat
    return out.reshape(a.shape)


def ifft(x: np.ndarray) -> np.ndarray:
    """Inverse FFT of any length."""
    return fft(x, inverse=True)


def fft2(x: np.ndarray, inverse: bool = False) -> np.ndarray:
    """Two-dimensional FFT via row-column decomposition."""
    a = np.asarray(x, dtype=np.complex128)
    a = fft(a, inverse)
    a = np.swapaxes(fft(np.swapaxes(a, -1, -2), inverse), -1, -2)
    return a


def two_real_ffts(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Transform two real sequences with one complex FFT (``a + i*b`` trick)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    z = fft(a + 1j * b)
    zr = np.conj(np.roll(z[::-1], 1))  # conj(Z[-k])
    return (z + zr) / 2, (z - zr) / (2j)


def rfft(x: np.ndarray) -> np.ndarray:
    """FFT of real input returning only the ``N//2 + 1`` non-redundant bins."""
    x = np.asarray(x, dtype=np.float64)
    return fft(x)[..., : x.shape[-1] // 2 + 1]


def irfft(spec: np.ndarray, n: int) -> np.ndarray:
    """Inverse of :func:`rfft` for an output length ``n``."""
    spec = np.asarray(spec, dtype=np.complex128)
    full = np.zeros(spec.shape[:-1] + (n,), dtype=np.complex128)
    h = spec.shape[-1]
    full[..., :h] = spec
    k = np.arange(h, n)
    full[..., h:] = np.conj(spec[..., (n - k)])
    return ifft(full).real
