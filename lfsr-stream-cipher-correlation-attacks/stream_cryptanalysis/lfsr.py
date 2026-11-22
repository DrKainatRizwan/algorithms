"""Linear feedback shift registers, Berlekamp-Massey and state recovery.

Convention
----------
An LFSR of length ``n`` with tap set ``T`` (a subset of ``0..n-1``) emits the
sequence ``s_0, s_1, ...`` with ``s_{t+n} = XOR_{i in T} s_{t+i}``.  The initial
state is ``(s_0, ..., s_{n-1})``; bit ``i`` of the integer state is ``s_i``.
The characteristic polynomial is ``x^n + sum_{i in T} x^i``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .gf2 import parity, solve

# Primitive feedback polynomials (tap sets, constant term included) so that the
# register has the maximal period 2^n - 1 for every non-zero state.
PRIMITIVE_TAPS: dict[int, tuple[int, ...]] = {
    3: (0, 1),
    4: (0, 1),
    5: (0, 2),
    7: (0, 1),
    8: (0, 2, 3, 4),
    9: (0, 4),
    10: (0, 3),
    11: (0, 2),
    12: (0, 1, 4, 6),
    13: (0, 1, 3, 4),
    14: (0, 1, 3, 5),
    15: (0, 1),
    16: (0, 2, 3, 5),
    17: (0, 3),
    19: (0, 1, 2, 5),
    20: (0, 3),
    23: (0, 5),
    31: (0, 3),
}


@dataclass(frozen=True)
class LFSR:
    """A Fibonacci-style binary LFSR."""

    length: int
    taps: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.length < 1:
            raise ValueError("length must be positive")
        if any(not 0 <= t < self.length for t in self.taps):
            raise ValueError("tap outside register")

    @classmethod
    def primitive(cls, length: int) -> "LFSR":
        """Return an LFSR of the given length with a primitive feedback polynomial."""
        if length not in PRIMITIVE_TAPS:
            raise KeyError(f"no primitive polynomial tabulated for length {length}")
        return cls(length, PRIMITIVE_TAPS[length])

    @property
    def tap_mask(self) -> int:
        return sum(1 << t for t in self.taps)

    def poly_exponents(self) -> list[int]:
        """Exponents present in the characteristic polynomial, ascending."""
        return sorted(set(self.taps) | {self.length})

    def stream(self, state: int, nbits: int) -> np.ndarray:
        """Return ``nbits`` output bits (uint8) starting from ``state``."""
        if not 0 <= state < (1 << self.length):
            raise ValueError("state out of range")
        n = self.length
        out = np.zeros(nbits, dtype=np.uint8)
        tm = self.tap_mask
        s = state
        for t in range(nbits):
            out[t] = s & 1
            fb = parity(s & tm)
            s = (s >> 1) | (fb << (n - 1))
        return out

    def period(self, state: int = 1) -> int:
        """Length of the cycle through ``state`` (brute force, small n only)."""
        n, tm = self.length, self.tap_mask
        s = start = state
        count = 0
        while True:
            fb = parity(s & tm)
            s = (s >> 1) | (fb << (n - 1))
            count += 1
            if s == start:
                return count

    def masks(self, nbits: int) -> list[int]:
        """Linear dependence of each output bit on the initial state.

        ``masks(N)[t]`` is an int ``m`` with ``s_t = parity(m & state)``.
        """
        n = self.length
        ms = [1 << t for t in range(min(n, nbits))]
        for t in range(n, nbits):
            m = 0
            for i in self.taps:
                m ^= ms[t - n + i]
            ms.append(m)
        return ms[:nbits]


def berlekamp_massey(bits: Sequence[int]) -> tuple[int, list[int]]:
    """Berlekamp-Massey over GF(2).

    Returns ``(L, C)`` where ``L`` is the linear complexity of ``bits`` and
    ``C`` the connection polynomial coefficients ``c_0=1, c_1, ..., c_L`` with
    ``sum_j c_j s_{t-j} = 0`` for all ``t >= L``.
    """
    n = len(bits)
    c = [1] + [0] * n
    b = [1] + [0] * n
    L, m = 0, 1
    for i in range(n):
        d = bits[i]
        for j in range(1, L + 1):
            d ^= c[j] & bits[i - j]
        if d == 0:
            m += 1
            continue
        t = c[:]
        for j in range(n + 1 - m):
            c[j + m] ^= b[j]
        if 2 * L <= i:
            L, b, m = i + 1 - L, t, 1
        else:
            m += 1
    return L, c[: L + 1]


def linear_complexity_profile(bits: Sequence[int]) -> list[int]:
    """Linear complexity of every prefix of ``bits``."""
    return [berlekamp_massey(bits[: i + 1])[0] for i in range(len(bits))]


def lfsr_from_connection(L: int, c: Sequence[int]) -> LFSR:
    """Build the LFSR (in this module's convention) from a BM connection polynomial.

    ``s_t = sum_{j=1..L} c_j s_{t-j}`` becomes ``s_{t+L} = sum c_j s_{t+L-j}``,
    i.e. tap index ``L - j``.
    """
    taps = tuple(sorted(L - j for j in range(1, L + 1) if c[j]))
    return LFSR(L, taps)


def recover_state(lfsr: LFSR, bits: Sequence[int], offset: int = 0) -> int | None:
    """Recover the initial state from observed output bits.

    ``bits[k]`` is taken as ``s_{offset+k}``.  Solves the linear system
    ``parity(mask_t & x) = bits`` over GF(2).  With at least ``n`` consecutive
    bits the solution is unique; returns ``None`` if the data is inconsistent.
    """
    ms = lfsr.masks(offset + len(bits))[offset:]
    res = solve(ms, [int(b) for b in bits], lfsr.length)
    if res is None:
        return None
    return res[0]


def sequence_from_taps(taps: Iterable[int], length: int, init: Sequence[int], nbits: int) -> list[int]:
    """Plain-list helper used for cross-checking the optimised generator."""
    s = list(init)
    tl = list(taps)
    while len(s) < nbits:
        t = len(s) - length
        s.append(sum(s[t + i] for i in tl) & 1)
    return s[:nbits]
