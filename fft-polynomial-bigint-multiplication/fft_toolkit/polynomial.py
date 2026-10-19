"""Dense univariate polynomials with FFT/NTT-accelerated arithmetic."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .convolution import convolve_fft, convolve_naive
from .ntt import CRT_LIMIT, convolve_exact

NAIVE_CUTOFF = 24


def _trim(c: list) -> list:
    while len(c) > 1 and c[-1] == 0:
        c.pop()
    return c or [0]


def multiply_coeffs(a: Sequence, b: Sequence) -> list:
    """Multiply coefficient lists (low degree first), choosing the best method.

    Integer inputs use the exact three-prime NTT, floats use the FFT, and short
    operands fall back to schoolbook multiplication.
    """
    if len(a) == 0 or len(b) == 0:
        return []
    exact = all(isinstance(v, (int, np.integer)) for v in list(a) + list(b))
    if exact:
        bound = max(abs(int(v)) for v in a) * max(abs(int(v)) for v in b) * min(len(a), len(b))
        too_big = bound >= CRT_LIMIT // 2
    else:
        too_big = False
    if too_big or min(len(a), len(b)) <= NAIVE_CUTOFF:
        if exact:
            out = [0] * (len(a) + len(b) - 1)
            for i, x in enumerate(a):
                for j, y in enumerate(b):
                    out[i + j] += int(x) * int(y)
            return out
        return convolve_naive(np.asarray(a, float), np.asarray(b, float)).tolist()
    if exact:
        return convolve_exact(a, b)
    return convolve_fft(a, b).tolist()


def inverse_series(c: Sequence[float], n: int) -> np.ndarray:
    """Power-series inverse ``g`` with ``f*g = 1 (mod x^n)`` by Newton iteration.

    Each step doubles the precision: ``g <- g * (2 - f * g) mod x^{2k}``.
    """
    c = np.asarray(c, dtype=np.float64)
    if c[0] == 0:
        raise ZeroDivisionError("constant term must be non-zero")
    g = np.array([1.0 / c[0]])
    k = 1
    while k < n:
        k *= 2
        fg = convolve_fft(c[:k], g)[:k]
        two_minus = -fg
        two_minus[0] += 2.0
        g = convolve_fft(g, two_minus)[:k]
    return g[:n]


class Polynomial:
    """Polynomial ``sum c[i] x^i`` stored low-degree first."""

    def __init__(self, coeffs: Sequence):
        self.coeffs = _trim(list(coeffs))

    @property
    def degree(self) -> int:
        """Degree of the polynomial (0 for the zero polynomial)."""
        return len(self.coeffs) - 1

    @classmethod
    def from_roots(cls, roots: Sequence) -> "Polynomial":
        """Monic polynomial with the given roots by balanced product tree."""
        if len(roots) == 0:
            return cls([1])
        level = [[-r, 1] for r in roots]
        while len(level) > 1:
            nxt = [multiply_coeffs(level[i], level[i + 1])
                   for i in range(0, len(level) - 1, 2)]
            if len(level) % 2:
                nxt.append(level[-1])
            level = nxt
        return cls(level[0])

    def __call__(self, x):
        """Evaluate by Horner's rule (works on scalars and numpy arrays)."""
        acc = 0 * x if isinstance(x, np.ndarray) else 0
        for c in reversed(self.coeffs):
            acc = acc * x + c
        return acc

    def __add__(self, other: "Polynomial") -> "Polynomial":
        n = max(len(self.coeffs), len(other.coeffs))
        a = self.coeffs + [0] * (n - len(self.coeffs))
        b = other.coeffs + [0] * (n - len(other.coeffs))
        return Polynomial([x + y for x, y in zip(a, b)])

    def __neg__(self) -> "Polynomial":
        return Polynomial([-c for c in self.coeffs])

    def __sub__(self, other: "Polynomial") -> "Polynomial":
        return self + (-other)

    def __mul__(self, other):
        if not isinstance(other, Polynomial):
            return Polynomial([c * other for c in self.coeffs])
        return Polynomial(multiply_coeffs(self.coeffs, other.coeffs))

    def __pow__(self, e: int) -> "Polynomial":
        """Exponentiation by squaring."""
        result, base = Polynomial([1]), self
        while e:
            if e & 1:
                result = result * base
            e >>= 1
            if e:
                base = base * base
        return result

    def __eq__(self, other) -> bool:
        return isinstance(other, Polynomial) and self.coeffs == other.coeffs

    def __repr__(self) -> str:
        return f"Polynomial({self.coeffs})"

    def derivative(self) -> "Polynomial":
        """Formal derivative."""
        return Polynomial([i * c for i, c in enumerate(self.coeffs)][1:] or [0])

    def divmod(self, other: "Polynomial") -> tuple["Polynomial", "Polynomial"]:
        """Quotient and remainder (float arithmetic) via reversed-series inversion.

        With ``rev(p)(x) = x^deg p(1/x)`` we have
        ``rev(q) = rev(a) * rev(b)^-1 mod x^{deg a - deg b + 1}``.
        """
        if other.coeffs == [0]:
            raise ZeroDivisionError("division by zero polynomial")
        da, db = self.degree, other.degree
        if da < db:
            return Polynomial([0]), Polynomial(self.coeffs)
        m = da - db + 1
        ra = np.asarray(self.coeffs[::-1], dtype=float)
        rb = np.asarray(other.coeffs[::-1], dtype=float)
        inv = inverse_series(rb, m)
        rq = convolve_fft(ra[:m], inv)[:m]
        q = Polynomial(rq[::-1].tolist())
        r = self - q * other
        return q, Polynomial(r.coeffs[:db] if db else [0])

    def divmod_naive(self, other: "Polynomial") -> tuple["Polynomial", "Polynomial"]:
        """Schoolbook long division (reference)."""
        rem = [float(c) for c in self.coeffs]
        db = other.degree
        if db > self.degree:
            return Polynomial([0]), Polynomial(self.coeffs)
        q = [0.0] * (self.degree - db + 1)
        for i in range(self.degree - db, -1, -1):
            f = rem[i + db] / other.coeffs[-1]
            q[i] = f
            for j, c in enumerate(other.coeffs):
                rem[i + j] -= f * c
        return Polynomial(q), Polynomial(rem[:db] if db else [0.0])
