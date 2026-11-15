"""Binary primitive narrow-sense BCH codes.

A ``BCH(n = 2^m - 1, t)`` code has generator polynomial equal to the lcm of
the minimal polynomials of ``alpha, alpha^2, ..., alpha^(2t)``.  Binary
polynomials are bit-packed into Python integers (bit ``i`` = coefficient of
``x^i``), so encoding is carry-less long division with shifts and XORs.
"""

from __future__ import annotations

from collections.abc import Sequence

from .berlekamp import berlekamp_massey
from .galois import GF2m
from .reed_solomon import DecodingFailure


def _gf2_mul(a: int, b: int) -> int:
    """Carry-less product of bit-packed polynomials."""
    out = 0
    while b:
        if b & 1:
            out ^= a
        a <<= 1
        b >>= 1
    return out


def _gf2_mod(a: int, g: int) -> int:
    """Remainder of ``a`` modulo ``g`` over GF(2)."""
    dg = g.bit_length()
    while a.bit_length() >= dg:
        a ^= g << (a.bit_length() - dg)
    return a


class BCH:
    """Binary BCH code of length ``2^m - 1`` correcting ``t`` bit errors."""

    def __init__(self, m: int, t: int) -> None:
        self.field = GF2m(m)
        self.m, self.t = m, t
        self.n = self.field.order
        if not 1 <= t or 2 * t >= self.n:
            raise ValueError("invalid error-correction capability")
        seen: set[int] = set()
        g = 1
        for i in range(1, 2 * t + 1):
            coset = frozenset(self.field.cyclotomic_coset(i))
            if coset in seen:
                continue
            seen.add(coset)
            g = _gf2_mul(g, self.field.minimal_poly(i))
        self.generator = g
        self.k = self.n - (g.bit_length() - 1)
        if self.k <= 0:
            raise ValueError("t too large: no information symbols remain")
        self.nparity = self.n - self.k

    @property
    def designed_distance(self) -> int:
        return 2 * self.t + 1

    # ------------------------------------------------------------ encoding
    def encode(self, message: Sequence[int]) -> list[int]:
        """Systematic encode; bit ``i`` of the result is the coefficient of
        ``x^i`` with parity in the low ``nparity`` positions."""
        if len(message) != self.k:
            raise ValueError(f"message must have {self.k} bits")
        if any(b not in (0, 1) for b in message):
            raise ValueError("message must be binary")
        msg = sum(b << i for i, b in enumerate(message))
        shifted = msg << self.nparity
        cw = shifted | _gf2_mod(shifted, self.generator)
        return [(cw >> i) & 1 for i in range(self.n)]

    def extract(self, codeword: Sequence[int]) -> list[int]:
        return list(codeword[self.nparity:])

    def is_codeword(self, word: Sequence[int]) -> bool:
        return not any(self.syndromes(word))

    # ------------------------------------------------------------ decoding
    def syndromes(self, received: Sequence[int]) -> list[int]:
        """``S_j = r(alpha^j)`` for ``j = 1..2t`` (returned 0-indexed)."""
        f = self.field
        ones = [i for i, b in enumerate(received) if b]
        out = []
        for j in range(1, 2 * self.t + 1):
            s = 0
            for i in ones:
                s ^= f.alpha_pow(i * j)
            out.append(s)
        return out

    def decode_full(self, received: Sequence[int]) -> tuple[list[int], list[int]]:
        """Return ``(corrected_codeword, flipped_positions)``."""
        if len(received) != self.n:
            raise ValueError(f"received word must have {self.n} bits")
        f = self.field
        word = [int(b) for b in received]
        synd = self.syndromes(word)
        if not any(synd):
            return word, []
        sigma, nu = berlekamp_massey(f, synd)
        if nu > self.t:
            raise DecodingFailure("more than t errors")
        positions = [p for p in range(self.n)
                     if f.poly_eval(sigma, f.alpha_pow(-p)) == 0]
        if len(positions) != nu:
            raise DecodingFailure("locator polynomial does not split")
        for p in positions:
            word[p] ^= 1
        if not self.is_codeword(word):
            raise DecodingFailure("residual syndrome after correction")
        return word, positions

    def decode(self, received: Sequence[int]) -> list[int]:
        return self.extract(self.decode_full(received)[0])

    def minimum_distance(self) -> int:
        """Exact minimum distance by enumerating codewords (small codes only)."""
        if self.k > 16:
            raise ValueError("code too large for exhaustive enumeration")
        best = self.n
        for msg in range(1, 1 << self.k):
            cw = self.encode([(msg >> i) & 1 for i in range(self.k)])
            best = min(best, sum(cw))
        return best
