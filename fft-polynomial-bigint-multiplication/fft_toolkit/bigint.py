"""Arbitrary-precision non-negative integers with FFT/NTT multiplication.

Numbers are stored as little-endian limbs in base ``10**LIMB_DIGITS``.  Three
multiplication back-ends are available: schoolbook, Karatsuba, and
convolution-based (floating FFT with rounding, or exact three-prime NTT).
"""
from __future__ import annotations

import numpy as np

from .convolution import convolve_fft
from .ntt import convolve_exact

LIMB_DIGITS = 4
BASE = 10 ** LIMB_DIGITS


def _carry(vals: list[int], base: int = BASE) -> list[int]:
    out, c = [], 0
    for v in vals:
        c, d = divmod(int(v) + c, base)
        out.append(d)
    while c:
        c, d = divmod(c, base)
        out.append(d)
    while len(out) > 1 and out[-1] == 0:
        out.pop()
    return out or [0]


def schoolbook_limbs(a: list[int], b: list[int]) -> list[int]:
    """O(nm) limb multiplication."""
    out = [0] * (len(a) + len(b))
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] += x * y
    return _carry(out)


def fft_limbs(a: list[int], b: list[int]) -> list[int]:
    """Limb multiplication through the floating-point FFT (rounded)."""
    conv = convolve_fft(np.array(a, float), np.array(b, float))
    return _carry(np.rint(conv).astype(np.int64).tolist())


def ntt_limbs(a: list[int], b: list[int]) -> list[int]:
    """Limb multiplication through the exact three-prime NTT."""
    return _carry(convolve_exact(a, b))


def fft_rounding_error(a: list[int], b: list[int]) -> float:
    """Largest distance of an FFT convolution coefficient from an integer."""
    conv = convolve_fft(np.array(a, float), np.array(b, float))
    return float(np.max(np.abs(conv - np.rint(conv))))


def karatsuba_int(x: int, y: int, cutoff_bits: int = 256) -> int:
    """Karatsuba multiplication of Python ints (baseline, 3 half-size products)."""
    if x.bit_length() <= cutoff_bits or y.bit_length() <= cutoff_bits:
        return _school_int(x, y)
    half = max(x.bit_length(), y.bit_length()) // 2
    mask = (1 << half) - 1
    x0, x1 = x & mask, x >> half
    y0, y1 = y & mask, y >> half
    z0 = karatsuba_int(x0, y0, cutoff_bits)
    z2 = karatsuba_int(x1, y1, cutoff_bits)
    z1 = karatsuba_int(x0 + x1, y0 + y1, cutoff_bits) - z0 - z2
    return (z2 << (2 * half)) + (z1 << half) + z0


def _school_int(x: int, y: int) -> int:
    """Shift-and-add product on 64-bit chunks (keeps Karatsuba leaves explicit)."""
    out, shift = 0, 0
    while y:
        out += (x * (y & 0xFFFFFFFFFFFFFFFF)) << shift
        y >>= 64
        shift += 64
    return out


class BigNum:
    """Non-negative big integer with selectable multiplication algorithm."""

    METHODS = ("school", "fft", "ntt")

    def __init__(self, limbs: list[int]):
        self.limbs = _carry(limbs)

    @classmethod
    def from_str(cls, s: str) -> "BigNum":
        """Parse a decimal string."""
        s = s.strip().lstrip("0") or "0"
        if not s.isdigit():
            raise ValueError("expected a non-negative decimal string")
        limbs = [int(s[max(i - LIMB_DIGITS, 0):i]) for i in range(len(s), 0, -LIMB_DIGITS)]
        return cls(limbs)

    @classmethod
    def from_int(cls, n: int) -> "BigNum":
        """Convert a non-negative Python int (O(n^2) limb extraction)."""
        if n < 0:
            raise ValueError("negative values are not supported")
        limbs = []
        while n:
            n, d = divmod(n, BASE)
            limbs.append(d)
        return cls(limbs or [0])

    def to_int(self) -> int:
        """Convert to a Python int by Horner evaluation."""
        v = 0
        for d in reversed(self.limbs):
            v = v * BASE + d
        return v

    def __str__(self) -> str:
        head = str(self.limbs[-1])
        return head + "".join(f"{d:0{LIMB_DIGITS}d}" for d in reversed(self.limbs[:-1]))

    def __repr__(self) -> str:
        return f"BigNum({str(self)[:30]}{'...' if len(str(self)) > 30 else ''})"

    def __eq__(self, other) -> bool:
        return isinstance(other, BigNum) and self.limbs == other.limbs

    def __lt__(self, other: "BigNum") -> bool:
        if len(self.limbs) != len(other.limbs):
            return len(self.limbs) < len(other.limbs)
        return self.limbs[::-1] < other.limbs[::-1]

    def __add__(self, other: "BigNum") -> "BigNum":
        n = max(len(self.limbs), len(other.limbs))
        a = self.limbs + [0] * (n - len(self.limbs))
        b = other.limbs + [0] * (n - len(other.limbs))
        return BigNum([x + y for x, y in zip(a, b)])

    def __sub__(self, other: "BigNum") -> "BigNum":
        if self < other:
            raise ValueError("result would be negative")
        out, borrow = [], 0
        for i, x in enumerate(self.limbs):
            y = other.limbs[i] if i < len(other.limbs) else 0
            d = x - y - borrow
            borrow = 1 if d < 0 else 0
            out.append(d + BASE * borrow)
        return BigNum(out)

    def multiply(self, other: "BigNum", method: str = "auto") -> "BigNum":
        """Multiply using ``school``, ``fft``, ``ntt`` or ``auto`` selection."""
        if method == "auto":
            small = min(len(self.limbs), len(other.limbs))
            method = "school" if small < 32 else ("fft" if len(self.limbs) + len(other.limbs) < 200_000 else "ntt")
        fn = {"school": schoolbook_limbs, "fft": fft_limbs, "ntt": ntt_limbs}.get(method)
        if fn is None:
            raise ValueError(f"unknown method {method!r}")
        return BigNum(fn(self.limbs, other.limbs))

    __mul__ = multiply

    def __pow__(self, e: int) -> "BigNum":
        result, base = BigNum([1]), self
        while e:
            if e & 1:
                result = result * base
            e >>= 1
            if e:
                base = base * base
        return result

    def divmod_small(self, d: int) -> tuple["BigNum", int]:
        """Divide by a small positive int, returning quotient and remainder."""
        if d <= 0:
            raise ZeroDivisionError
        out, r = [], 0
        for limb in reversed(self.limbs):
            q, r = divmod(r * BASE + limb, d)
            out.append(q)
        return BigNum(out[::-1]), r


def factorial(n: int, method: str = "fft") -> BigNum:
    """``n!`` via a balanced product tree so large operands dominate the cost."""
    def prod(lo: int, hi: int) -> BigNum:
        if hi - lo <= 8:
            v = 1
            for k in range(lo, hi + 1):
                v *= k
            return BigNum.from_int(v)
        mid = (lo + hi) // 2
        return prod(lo, mid).multiply(prod(mid + 1, hi), method)
    return BigNum([1]) if n < 2 else prod(2, n)


def fibonacci(n: int, method: str = "fft") -> BigNum:
    """``F(n)`` by fast doubling; ``F(2k)=F(k)(2F(k+1)-F(k))``, ``F(2k+1)=F(k)^2+F(k+1)^2``."""
    def fd(k: int) -> tuple[BigNum, BigNum]:
        if k == 0:
            return BigNum([0]), BigNum([1])
        a, b = fd(k // 2)
        c = a.multiply(b + b - a, method)
        d = a.multiply(a, method) + b.multiply(b, method)
        return (c, d) if k % 2 == 0 else (d, c + d)
    return fd(n)[0]
