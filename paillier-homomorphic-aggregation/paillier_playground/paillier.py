"""The Paillier cryptosystem (g = n + 1 variant) with CRT-accelerated decryption."""
from __future__ import annotations

import random
from dataclasses import dataclass

from .numtheory import crt, lcm, modinv, random_coprime, random_prime


@dataclass(frozen=True)
class PublicKey:
    """Public key: modulus ``n``; plaintexts live in Z_n, ciphertexts in Z_{n^2}."""

    n: int

    @property
    def n_sq(self) -> int:
        return self.n * self.n

    @property
    def max_int(self) -> int:
        """Largest magnitude of a signed plaintext that decodes unambiguously."""
        return self.n // 2 - 1

    def raw_encrypt(self, m: int, rng: random.Random, r: int | None = None) -> int:
        """Encrypt ``m`` (reduced mod n): c = (1 + m n) r^n mod n^2."""
        r = random_coprime(self.n, rng) if r is None else r
        return (1 + (m % self.n) * self.n) * pow(r, self.n, self.n_sq) % self.n_sq

    def add(self, c1: int, c2: int) -> int:
        """Homomorphic addition of plaintexts."""
        return c1 * c2 % self.n_sq

    def mul_plain(self, c: int, k: int) -> int:
        """Homomorphic multiplication by a plaintext integer (may be negative)."""
        if k < 0:
            c, k = modinv(c, self.n_sq), -k
        return pow(c, k, self.n_sq)

    def rerandomize(self, c: int, rng: random.Random) -> int:
        """Fresh ciphertext of the same plaintext (unlinkable to ``c``)."""
        return c * pow(random_coprime(self.n, rng), self.n, self.n_sq) % self.n_sq


@dataclass(frozen=True)
class PrivateKey:
    """Private key holding the factorisation of n."""

    public: PublicKey
    p: int
    q: int

    @property
    def lam(self) -> int:
        return lcm(self.p - 1, self.q - 1)

    def decrypt_textbook(self, c: int) -> int:
        """Decrypt with lambda = lcm(p-1, q-1); returns value in [0, n)."""
        n, n_sq = self.public.n, self.public.n_sq
        lam = self.lam
        mu = modinv(lam, n)  # since L(g^lam) = lam for g = n + 1
        return (pow(c, lam, n_sq) - 1) // n * mu % n

    def decrypt(self, c: int) -> int:
        """CRT decryption (about 3-4x faster); returns value in [0, n)."""
        p, q = self.p, self.q
        mp = self._half(c, p)
        mq = self._half(c, q)
        return crt(mp, p, mq, q)

    def _half(self, c: int, prime: int) -> int:
        n = self.public.n
        psq = prime * prime
        l_val = (pow(c % psq, prime - 1, psq) - 1) // prime
        h = modinv((pow(n + 1, prime - 1, psq) - 1) // prime % prime, prime)
        return l_val * h % prime

    def decrypt_signed(self, c: int) -> int:
        """Decrypt and map to the signed range (-n/2, n/2)."""
        m = self.decrypt(c)
        return m - self.public.n if m > self.public.n // 2 else m


def generate_keypair(bits: int = 512, seed: int | None = None) -> tuple[PublicKey, PrivateKey]:
    """Generate a key pair with an ``bits``-bit modulus (deterministic if seeded)."""
    if bits < 16:
        raise ValueError("modulus too small")
    rng = random.Random(seed)
    half = bits // 2
    while True:
        p = random_prime(half, rng)
        q = random_prime(bits - half, rng)
        if p != q and (p * q).bit_length() == bits:
            break
    pub = PublicKey(p * q)
    return pub, PrivateKey(pub, p, q)
