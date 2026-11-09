"""Textbook-RSA weaknesses: Wiener, shared primes, common modulus and Hastad broadcast."""
from __future__ import annotations

import random
from dataclasses import dataclass
from math import gcd, isqrt

from .primes import crt, egcd, modinv, random_prime


@dataclass(frozen=True)
class RSAKey:
    """RSA key material (n = p*q, public exponent e, private exponent d)."""
    p: int
    q: int
    e: int
    d: int

    @property
    def n(self) -> int:
        return self.p * self.q


def generate_rsa(bits: int, rng: random.Random, e: int = 65537, small_d: bool = False,
                 close_primes: bool = False) -> RSAKey:
    """Generate an RSA key; optionally deliberately weak (small d or close primes)."""
    half = bits // 2
    while True:
        p = random_prime(half, rng)
        if close_primes:
            from .primes import next_prime
            q = next_prime(p + rng.randrange(2, 2000))
        else:
            q = random_prime(bits - half, rng)
        if p == q:
            continue
        phi = (p - 1) * (q - 1)
        if small_d:
            dbits = max(8, (p * q).bit_length() // 4 - 3)
            d = rng.getrandbits(dbits) | (1 << (dbits - 1)) | 1
            if gcd(d, phi) != 1:
                continue
            return RSAKey(max(p, q), min(p, q), modinv(d, phi), d)
        if gcd(e, phi) != 1:
            continue
        return RSAKey(max(p, q), min(p, q), e, modinv(e, phi))


def _convergents(a: int, b: int):
    """Yield continued-fraction convergents (num, den) of a/b."""
    h0, h1, k0, k1 = 0, 1, 1, 0
    while b:
        q = a // b
        a, b = b, a - q * b
        h0, h1 = h1, q * h1 + h0
        k0, k1 = k1, q * k1 + k0
        yield h1, k1


def wiener_attack(e: int, n: int) -> tuple[int, int, int] | None:
    """Wiener's attack: recovers (d, p, q) when d < n^(1/4)/3.

    Some convergent k/d of e/n satisfies e*d - 1 = k*phi(n); phi gives p+q and hence
    the roots of x^2 - (p+q)x + n.
    """
    for k, d in _convergents(e, n):
        if k == 0 or (e * d - 1) % k:
            continue
        phi = (e * d - 1) // k
        s = n - phi + 1
        disc = s * s - 4 * n
        if disc < 0:
            continue
        r = isqrt(disc)
        if r * r == disc and (s + r) % 2 == 0:
            p, q = (s + r) // 2, (s - r) // 2
            if p * q == n and q > 1:
                return d, p, q
    return None


def shared_prime_attack(moduli: list[int]) -> dict[int, tuple[int, int]]:
    """Pairwise-GCD scan: for each modulus sharing a prime with another, return (p, q)."""
    found: dict[int, tuple[int, int]] = {}
    for i, a in enumerate(moduli):
        for b in moduli[i + 1:]:
            g = gcd(a, b)
            if 1 < g < a and a != b:
                found.setdefault(a, (g, a // g))
            if 1 < g < b and a != b:
                found.setdefault(b, (g, b // g))
    return found


def common_modulus_attack(n: int, e1: int, e2: int, c1: int, c2: int) -> int:
    """Recover m from two encryptions under coprime exponents with the same modulus."""
    g, a, b = egcd(e1, e2)
    if g != 1:
        raise ValueError("exponents must be coprime")
    if a < 0:
        c1, a = modinv(c1, n), -a
    if b < 0:
        c2, b = modinv(c2, n), -b
    return pow(c1, a, n) * pow(c2, b, n) % n


def integer_root(x: int, k: int) -> int:
    """Floor of the k-th root of ``x`` via Newton iteration."""
    if x < 2:
        return x
    r = 1 << ((x.bit_length() + k - 1) // k)
    while True:
        nr = ((k - 1) * r + x // r ** (k - 1)) // k
        if nr >= r:
            return r
        r = nr


def hastad_broadcast(ciphertexts: list[int], moduli: list[int], e: int) -> int:
    """Hastad's attack: the same message sent to >= e recipients with exponent e."""
    c, _ = crt(ciphertexts[:e], moduli[:e])
    return integer_root(c, e)
