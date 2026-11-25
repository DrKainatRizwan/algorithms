"""Statistical randomness tests (NIST SP 800-22 flavoured) on bit arrays."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------- special functions
def gammaincc(a: float, x: float) -> float:
    """Regularised upper incomplete gamma function Q(a, x)."""
    if x <= 0:
        return 1.0
    if x < a + 1:
        term = total = 1.0 / a
        n = a
        for _ in range(1000):
            n += 1
            term *= x / n
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return max(0.0, 1.0 - total * math.exp(-x + a * math.log(x) - math.lgamma(a)))
    tiny = 1e-300
    b = x + 1 - a
    c = 1 / tiny
    d = 1 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-15:
            break
    return h * math.exp(-x + a * math.log(x) - math.lgamma(a))


def chi2_sf(chi2: float, dof: float) -> float:
    """Survival function of the chi-square distribution."""
    return gammaincc(dof / 2.0, chi2 / 2.0)


@dataclass(frozen=True)
class TestResult:
    name: str
    p_value: float
    statistic: float

    def passed(self, alpha: float = 1e-3) -> bool:
        return self.p_value >= alpha


def _bits(bits) -> np.ndarray:
    return np.asarray(bits, dtype=np.uint8).reshape(-1)


# ---------------------------------------------------------------- tests
def monobit_test(bits) -> TestResult:
    b = _bits(bits)
    n = b.size
    s = int(2 * int(b.sum()) - n)
    stat = abs(s) / math.sqrt(n)
    return TestResult("monobit", math.erfc(stat / math.sqrt(2)), stat)


def runs_test(bits) -> TestResult:
    b = _bits(bits)
    n = b.size
    pi = b.mean()
    if abs(pi - 0.5) >= 2 / math.sqrt(n):
        return TestResult("runs", 0.0, float("inf"))
    v = 1 + int(np.count_nonzero(b[1:] != b[:-1]))
    num = abs(v - 2 * n * pi * (1 - pi))
    den = 2 * math.sqrt(2 * n) * pi * (1 - pi)
    return TestResult("runs", math.erfc(num / den), num / den)


def block_frequency_test(bits, block: int = 128) -> TestResult:
    b = _bits(bits)
    nb = b.size // block
    if nb < 1:
        raise ValueError("sequence shorter than one block")
    frac = b[: nb * block].reshape(nb, block).mean(axis=1)
    chi2 = 4.0 * block * float(((frac - 0.5) ** 2).sum())
    return TestResult("block_frequency", chi2_sf(chi2, nb), chi2)


def byte_chi_square_test(bits) -> TestResult:
    b = _bits(bits)
    nbytes = b.size // 8
    vals = np.packbits(b[: nbytes * 8])
    counts = np.bincount(vals, minlength=256).astype(float)
    exp = nbytes / 256.0
    chi2 = float(((counts - exp) ** 2 / exp).sum())
    return TestResult("byte_chi_square", chi2_sf(chi2, 255), chi2)


def _psi2(b: np.ndarray, m: int) -> float:
    if m == 0:
        return 0.0
    n = b.size
    ext = np.concatenate([b, b[: m - 1]]).astype(np.int64)
    idx = np.zeros(n, dtype=np.int64)
    for k in range(m):
        idx = (idx << 1) | ext[k:k + n]
    counts = np.bincount(idx, minlength=1 << m).astype(float)
    return float((counts ** 2).sum()) * (1 << m) / n - n


def serial_test(bits, m: int = 3) -> TestResult:
    """NIST serial test (first statistic) with overlapping m-bit patterns."""
    b = _bits(bits)
    d1 = _psi2(b, m) - _psi2(b, m - 1)
    return TestResult("serial", gammaincc(2 ** (m - 2), d1 / 2.0), d1)


def autocorrelation_test(bits, lag: int = 1) -> TestResult:
    b = _bits(bits)
    n = b.size - lag
    a = int(np.count_nonzero(b[:n] != b[lag:]))
    z = 2.0 * (a - n / 2.0) / math.sqrt(n)
    return TestResult(f"autocorr_lag{lag}", math.erfc(abs(z) / math.sqrt(2)), abs(z))


def spectral_test(bits) -> TestResult:
    b = _bits(bits)
    n = b.size
    mags = np.abs(np.fft.fft(2.0 * b - 1.0))[: n // 2]
    thr = math.sqrt(math.log(1 / 0.05) * n)
    n0 = 0.95 * n / 2.0
    n1 = int((mags < thr).sum())
    d = (n1 - n0) / math.sqrt(n * 0.95 * 0.05 / 4.0)
    return TestResult("spectral", math.erfc(abs(d) / math.sqrt(2)), abs(d))


def berlekamp_massey(bits) -> int:
    """Linear complexity of a GF(2) sequence (bitset Berlekamp-Massey)."""
    s = [int(x) for x in bits]
    n = len(s)
    c, bpoly = 1, 1          # connection polynomials as bitmasks
    l, m = 0, -1
    for i in range(n):
        d = s[i]
        # discrepancy: sum_{j=1..l} c_j * s[i-j]
        cc = c >> 1
        j = 1
        while cc:
            if cc & 1:
                d ^= s[i - j]
            cc >>= 1
            j += 1
        if d:
            t = c
            c ^= bpoly << (i - m)
            if 2 * l <= i:
                l, m, bpoly = i + 1 - l, i, t
    return l


def linear_complexity_test(bits, n: int = 2000) -> TestResult:
    """Flag sequences whose linear complexity falls well below ``n/2``.

    For a random sequence P(L <= n/2 - d) is roughly 2^(-2d).
    """
    b = _bits(bits)[:n]
    length = b.size
    lc = berlekamp_massey(b)
    d = length / 2.0 - lc
    p = 1.0 if d <= 0 else min(1.0, 2.0 ** (-2.0 * d))
    return TestResult("linear_complexity", p, float(lc))


def run_battery(bits, alpha: float = 1e-3) -> list[TestResult]:
    """Run the full battery and return all results."""
    return [
        monobit_test(bits),
        runs_test(bits),
        block_frequency_test(bits),
        byte_chi_square_test(bits),
        serial_test(bits),
        autocorrelation_test(bits, 1),
        autocorrelation_test(bits, 2),
        spectral_test(bits),
        linear_complexity_test(bits),
    ]
