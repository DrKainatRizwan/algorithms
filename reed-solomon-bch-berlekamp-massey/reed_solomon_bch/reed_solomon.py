"""Systematic Reed-Solomon codes with errors-and-erasures decoding.

An ``RS(n, k)`` code over GF(2^m) has ``n <= 2^m - 1`` symbols, ``k`` data
symbols and ``nsym = n - k`` parity symbols.  The generator polynomial is
``g(x) = prod_{i=0}^{nsym-1} (x - alpha^(fcr+i))``.  Any pattern with
``2 * errors + erasures <= nsym`` is corrected.

Codewords are lists of length ``n`` where index ``i`` holds the coefficient of
``x**i``; the ``nsym`` parity symbols occupy indices ``0..nsym-1`` and the
message occupies indices ``nsym..n-1``.
"""

from __future__ import annotations

from collections.abc import Sequence

from .berlekamp import berlekamp_massey, pgz_locator
from .galois import GF2m, Poly


class DecodingFailure(Exception):
    """Raised when the received word is uncorrectable (detected failure)."""


class ReedSolomon:
    """Reed-Solomon encoder/decoder over GF(2^m)."""

    def __init__(self, n: int, k: int, m: int = 8, fcr: int = 1,
                 method: str = "bm") -> None:
        if not 0 < k < n <= (1 << m) - 1:
            raise ValueError("need 0 < k < n <= 2^m - 1")
        if method not in ("bm", "pgz"):
            raise ValueError("method must be 'bm' or 'pgz'")
        self.field = GF2m(m)
        self.n, self.k, self.m = n, k, m
        self.nsym = n - k
        self.t = self.nsym // 2
        self.fcr = fcr
        self.method = method
        g: Poly = [1]
        for i in range(self.nsym):
            g = self.field.poly_mul(g, [self.field.alpha_pow(fcr + i), 1])
        self.generator = g

    # ------------------------------------------------------------ encoding
    def encode(self, message: Sequence[int]) -> list[int]:
        """Systematic encoding: ``c(x) = m(x) x^nsym + (m(x) x^nsym mod g)``."""
        if len(message) != self.k:
            raise ValueError(f"message must have {self.k} symbols")
        if any(not 0 <= s < self.field.size for s in message):
            raise ValueError("symbol out of field range")
        shifted = [0] * self.nsym + list(message)
        _, rem = self.field.poly_divmod(shifted, self.generator)
        parity = rem + [0] * (self.nsym - len(rem))
        return parity[: self.nsym] + list(message)

    def extract(self, codeword: Sequence[int]) -> list[int]:
        """Message part of a (corrected) codeword."""
        return list(codeword[self.nsym:])

    # ------------------------------------------------------------ syndromes
    def syndromes(self, received: Sequence[int]) -> list[int]:
        f = self.field
        return [f.poly_eval(list(received), f.alpha_pow(self.fcr + j))
                for j in range(self.nsym)]

    def is_codeword(self, word: Sequence[int]) -> bool:
        return not any(self.syndromes(word))

    # ------------------------------------------------------------- decoding
    def _erasure_locator(self, erasures: Sequence[int]) -> Poly:
        f = self.field
        gamma: Poly = [1]
        for p in erasures:
            gamma = f.poly_mul(gamma, [1, f.alpha_pow(p)])
        return gamma

    def _error_locator(self, synd: list[int], gamma: Poly, ne: int) -> Poly:
        """Locator for the unknown errors using Forney (modified) syndromes."""
        f = self.field
        # T(x) = S(x) * Gamma(x) mod x^nsym; entries ne.. are error-only
        t_poly = f.poly_mul(synd, gamma)[: self.nsym]
        t_poly += [0] * (self.nsym - len(t_poly))
        forney = t_poly[ne:]
        budget = (self.nsym - ne) // 2
        if not forney:
            return [1]
        if self.method == "bm":
            sigma, nu = berlekamp_massey(f, forney)
        else:
            sigma, nu = pgz_locator(f, forney, budget)
        if nu > budget:
            raise DecodingFailure("too many errors")
        return sigma

    def decode_full(self, received: Sequence[int],
                    erasures: Sequence[int] = ()) -> tuple[list[int], list[int]]:
        """Correct errors and erasures.

        Returns ``(codeword, error_positions)`` where positions include the
        erasures that were repaired.  Raises :class:`DecodingFailure` when the
        pattern exceeds the code's capability and is detected as such.
        """
        f = self.field
        if len(received) != self.n:
            raise ValueError(f"received word must have {self.n} symbols")
        erasures = sorted(set(erasures))
        if any(not 0 <= p < self.n for p in erasures):
            raise ValueError("erasure position out of range")
        if len(erasures) > self.nsym:
            raise DecodingFailure("too many erasures")
        word = list(received)
        for p in erasures:
            word[p] = 0
        synd = self.syndromes(word)
        if not any(synd):
            return word, []
        gamma = self._erasure_locator(erasures)
        sigma = self._error_locator(synd, gamma, len(erasures))
        psi = f.trim(f.poly_mul(sigma, gamma))
        deg = len(psi) - 1
        # Chien search: roots of psi at alpha^(-p)
        positions = [p for p in range(self.n)
                     if f.poly_eval(psi, f.alpha_pow(-p)) == 0]
        if len(positions) != deg:
            raise DecodingFailure("locator polynomial does not split")
        omega = f.poly_mul(synd, psi)[: self.nsym]
        dpsi = f.poly_deriv(psi)
        for p in positions:
            x = f.alpha_pow(p)
            xinv = f.alpha_pow(-p)
            den = f.poly_eval(dpsi, xinv)
            if den == 0:
                raise DecodingFailure("zero derivative in Forney formula")
            mag = f.div(f.poly_eval(omega, xinv), den)
            mag = f.mul(mag, f.pow(x, 1 - self.fcr) if self.fcr != 1 else 1)
            word[p] ^= mag
        if any(self.syndromes(word)):
            raise DecodingFailure("residual syndrome after correction")
        return word, positions

    def decode(self, received: Sequence[int],
               erasures: Sequence[int] = ()) -> list[int]:
        """Return the recovered message symbols."""
        return self.extract(self.decode_full(received, erasures)[0])

    # ---------------------------------------------------------- byte helpers
    def encode_bytes(self, data: bytes) -> list[int]:
        """Encode arbitrary bytes (requires ``m == 8``); pads the last block."""
        if self.m != 8:
            raise ValueError("byte interface requires m = 8")
        padded = data + bytes((-len(data)) % self.k)
        out: list[int] = []
        for i in range(0, len(padded), self.k):
            out.extend(self.encode(list(padded[i:i + self.k])))
        return out

    def decode_bytes(self, symbols: Sequence[int], length: int) -> bytes:
        """Inverse of :meth:`encode_bytes`; ``length`` strips the padding."""
        if len(symbols) % self.n:
            raise ValueError("symbol stream is not a multiple of n")
        out = bytearray()
        for i in range(0, len(symbols), self.n):
            out.extend(self.decode(symbols[i:i + self.n]))
        return bytes(out[:length])
