"""Minimal Paillier additively homomorphic cryptosystem."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .numtheory import modinv, random_prime


@dataclass(frozen=True)
class PublicKey:
    """Paillier public key (``g = n + 1``)."""

    n: int

    @property
    def n2(self) -> int:
        return self.n * self.n

    def encrypt(self, m: int, rng: random.Random) -> int:
        """Encrypt ``0 <= m < n``."""
        if not 0 <= m < self.n:
            raise ValueError("plaintext out of range")
        while True:
            r = rng.randrange(2, self.n)
            if math.gcd(r, self.n) == 1:
                break
        return (1 + m * self.n) * pow(r, self.n, self.n2) % self.n2

    def add(self, c1: int, c2: int) -> int:
        """Homomorphic addition of plaintexts."""
        return c1 * c2 % self.n2

    def scalar_mul(self, c: int, k: int) -> int:
        """Homomorphic multiplication of the plaintext by a non-negative integer."""
        return pow(c, k, self.n2)


@dataclass(frozen=True)
class PrivateKey:
    """Paillier private key."""

    public: PublicKey
    lam: int
    mu: int

    def decrypt(self, c: int) -> int:
        n = self.public.n
        x = pow(c, self.lam, self.public.n2)
        return (x - 1) // n * self.mu % n


def keygen(bits: int, rng: random.Random) -> PrivateKey:
    """Generate a key whose modulus has about ``bits`` bits."""
    while True:
        p = random_prime(bits // 2, rng)
        q = random_prime(bits // 2, rng)
        if p != q and math.gcd(p * q, (p - 1) * (q - 1)) == 1:
            break
    n = p * q
    lam = math.lcm(p - 1, q - 1)
    # L(g^lam mod n^2) = lam mod n for g = n + 1
    return PrivateKey(PublicKey(n), lam, modinv(lam % n, n))
