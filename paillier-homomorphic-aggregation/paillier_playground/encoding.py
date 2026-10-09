"""Fixed-point encoding and an ``EncryptedNumber`` wrapper with operator overloads."""
from __future__ import annotations

import random
from dataclasses import dataclass

from .paillier import PrivateKey, PublicKey

SCALE = 10**6


def encode(value: float | int, scale: int = SCALE) -> int:
    """Encode a number as a signed fixed-point integer."""
    if isinstance(value, int):
        return value * scale
    return int(round(value * scale))


@dataclass
class EncryptedNumber:
    """Ciphertext tagged with a fixed-point exponent: value = m / SCALE**exponent."""

    public: PublicKey
    ciphertext: int
    exponent: int = 1

    @classmethod
    def encrypt(cls, public: PublicKey, value: float | int, rng: random.Random) -> "EncryptedNumber":
        m = encode(value)
        if abs(m) > public.max_int:
            raise OverflowError("value exceeds plaintext space")
        return cls(public, public.raw_encrypt(m, rng), 1)

    def decrypt(self, private: PrivateKey) -> float:
        return private.decrypt_signed(self.ciphertext) / SCALE**self.exponent

    def _lift(self, exponent: int) -> int:
        if exponent < self.exponent:
            raise ValueError("cannot lower exponent")
        return self.public.mul_plain(self.ciphertext, SCALE ** (exponent - self.exponent))

    def __add__(self, other: "EncryptedNumber | float | int") -> "EncryptedNumber":
        if not isinstance(other, EncryptedNumber):
            m = encode(other, SCALE**self.exponent)
            ct = self.public.add(self.ciphertext, self.public.raw_encrypt(m, random.Random(0), r=1))
            return EncryptedNumber(self.public, ct, self.exponent)
        if other.public != self.public:
            raise ValueError("public keys differ")
        e = max(self.exponent, other.exponent)
        return EncryptedNumber(self.public, self.public.add(self._lift(e), other._lift(e)), e)

    __radd__ = __add__

    def __neg__(self) -> "EncryptedNumber":
        return EncryptedNumber(self.public, self.public.mul_plain(self.ciphertext, -1), self.exponent)

    def __sub__(self, other: "EncryptedNumber | float | int") -> "EncryptedNumber":
        return self + (-other)

    def __mul__(self, k: float | int) -> "EncryptedNumber":
        """Multiply by a plaintext scalar (ciphertext * ciphertext is not supported)."""
        if isinstance(k, EncryptedNumber):
            raise TypeError("Paillier is only additively homomorphic")
        if isinstance(k, int):
            return EncryptedNumber(self.public, self.public.mul_plain(self.ciphertext, k), self.exponent)
        return EncryptedNumber(
            self.public, self.public.mul_plain(self.ciphertext, encode(k)), self.exponent + 1
        )

    __rmul__ = __mul__

    def rerandomize(self, rng: random.Random) -> "EncryptedNumber":
        return EncryptedNumber(self.public, self.public.rerandomize(self.ciphertext, rng), self.exponent)
