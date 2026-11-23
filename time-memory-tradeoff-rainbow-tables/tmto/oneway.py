"""One-way functions on a finite search space of size N = 2**bits.

The attacks in this package invert ``f: [0, N) -> [0, N)``.  ``f`` is a keyed
multiply/xorshift mixer (a stand-in for a real hash truncated to ``bits``
bits); it behaves like a random function, which is the model assumed in the
analyses of Hellman (1980) and Oechslin (2003).  Evaluations are counted so
that online cost can be reported exactly.
"""

from __future__ import annotations

import string
from dataclasses import dataclass, field

_M64 = (1 << 64) - 1


def _mix64(x: int) -> int:
    """SplitMix64 finaliser: a fast, well-distributed 64-bit bijection."""
    x = (x + 0x9E3779B97F4A7C15) & _M64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _M64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _M64
    return x ^ (x >> 31)


@dataclass
class OneWayFunction:
    """Random-looking function on ``bits``-bit integers with an evaluation counter."""

    bits: int
    salt: int = 0x5DEECE66D
    evaluations: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if not 4 <= self.bits <= 48:
            raise ValueError("bits must be in [4, 48]")
        self.size = 1 << self.bits
        self.mask = self.size - 1

    def __call__(self, x: int) -> int:
        self.evaluations += 1
        return _mix64(x ^ self.salt) & self.mask

    def reset_counter(self) -> int:
        """Return the evaluation count and set it back to zero."""
        n, self.evaluations = self.evaluations, 0
        return n


class PasswordSpace:
    """Bijection between integers and fixed-length lowercase/digit passwords.

    Lets the demo present the attack as cracking unsalted password hashes.
    """

    ALPHABET = string.ascii_lowercase + string.digits

    def __init__(self, length: int) -> None:
        if length < 1:
            raise ValueError("length must be positive")
        self.length = length
        self.size = len(self.ALPHABET) ** length

    def encode(self, index: int) -> str:
        """Integer in ``[0, size)`` to password string."""
        if not 0 <= index < self.size:
            raise ValueError("index out of range")
        base = len(self.ALPHABET)
        chars = []
        for _ in range(self.length):
            index, r = divmod(index, base)
            chars.append(self.ALPHABET[r])
        return "".join(reversed(chars))

    def decode(self, password: str) -> int:
        """Password string to its integer index."""
        if len(password) != self.length:
            raise ValueError("wrong password length")
        value = 0
        for ch in password:
            value = value * len(self.ALPHABET) + self.ALPHABET.index(ch)
        return value
