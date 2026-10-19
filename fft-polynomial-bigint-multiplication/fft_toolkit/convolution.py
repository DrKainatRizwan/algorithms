"""Convolution and correlation built on the FFT and NTT, with applications."""
from __future__ import annotations

import numpy as np

from .fft import fft, fft2, ifft, next_power_of_two
from .ntt import convolve_exact


def convolve_naive(a, b) -> np.ndarray:
    """Direct O(nm) linear convolution (reference)."""
    a = np.asarray(a)
    b = np.asarray(b)
    out = np.zeros(a.size + b.size - 1, dtype=np.result_type(a, b))
    for i, v in enumerate(a):
        out[i:i + b.size] += v * b
    return out


def convolve_fft(a, b) -> np.ndarray:
    """Linear convolution of real sequences using zero-padded FFTs."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return np.zeros(0)
    n = next_power_of_two(a.size + b.size - 1)
    fa = fft(np.pad(a, (0, n - a.size)))
    fb = fft(np.pad(b, (0, n - b.size)))
    return ifft(fa * fb).real[: a.size + b.size - 1]


def circular_convolve(a, b) -> np.ndarray:
    """Circular convolution of two equal-length real sequences."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("circular convolution needs equal lengths")
    return ifft(fft(a) * fft(b)).real


def cross_correlate(a, b) -> np.ndarray:
    """Full cross-correlation ``c[k] = sum_i a[i+k-(m-1)] * b[i]`` via FFT."""
    b = np.asarray(b, dtype=np.float64)
    return convolve_fft(a, b[::-1])


def convolve_2d(img, kernel) -> np.ndarray:
    """Full 2-D linear convolution using row-column FFTs."""
    img = np.asarray(img, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    shape = (img.shape[0] + kernel.shape[0] - 1, img.shape[1] + kernel.shape[1] - 1)
    ph, pw = next_power_of_two(shape[0]), next_power_of_two(shape[1])
    fa = fft2(np.pad(img, ((0, ph - img.shape[0]), (0, pw - img.shape[1]))))
    fb = fft2(np.pad(kernel, ((0, ph - kernel.shape[0]), (0, pw - kernel.shape[1]))))
    return fft2(fa * fb, inverse=True).real[: shape[0], : shape[1]]


def convolve_2d_naive(img, kernel) -> np.ndarray:
    """Direct 2-D convolution (reference)."""
    img = np.asarray(img, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    out = np.zeros((img.shape[0] + kernel.shape[0] - 1, img.shape[1] + kernel.shape[1] - 1))
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            out[i:i + img.shape[0], j:j + img.shape[1]] += kernel[i, j] * img
    return out


def wildcard_match(text: str, pattern: str, wildcard: str = "?") -> list[int]:
    """All start positions where ``pattern`` matches ``text``; ``?`` matches anything.

    Uses the identity ``sum_j p_j t_{i+j} (p_j - t_{i+j})^2 = 0`` (wildcards
    encoded as 0) which expands into three exact convolutions.
    """
    n, m = len(text), len(pattern)
    if m == 0 or m > n:
        return [] if m else list(range(n + 1))
    t = [0 if c == wildcard else ord(c) - 31 for c in text]
    p = [0 if c == wildcard else ord(c) - 31 for c in pattern]
    pr = p[::-1]
    t1, t2, t3 = t, [v * v for v in t], [v ** 3 for v in t]
    p1, p2, p3 = pr, [v * v for v in pr], [v ** 3 for v in pr]
    c1 = convolve_exact(t3, p1)
    c2 = convolve_exact(t2, p2)
    c3 = convolve_exact(t1, p3)
    return [i for i in range(n - m + 1)
            if c1[i + m - 1] - 2 * c2[i + m - 1] + c3[i + m - 1] == 0]


def wildcard_match_naive(text: str, pattern: str, wildcard: str = "?") -> list[int]:
    """Quadratic-time reference for :func:`wildcard_match`."""
    m = len(pattern)
    return [i for i in range(len(text) - m + 1)
            if all(pc == wildcard or tc == wildcard or pc == tc
                   for pc, tc in zip(pattern, text[i:i + m]))]


def count_matches(text: str, pattern: str) -> list[int]:
    """Number of equal characters between ``pattern`` and each window of ``text``."""
    alphabet = sorted(set(text) | set(pattern))
    n, m = len(text), len(pattern)
    total = np.zeros(max(n - m + 1, 0))
    for ch in alphabet:
        ta = np.array([c == ch for c in text], dtype=float)
        pa = np.array([c == ch for c in pattern], dtype=float)
        total += convolve_fft(ta, pa[::-1])[m - 1: n]
    return [int(round(v)) for v in total]
