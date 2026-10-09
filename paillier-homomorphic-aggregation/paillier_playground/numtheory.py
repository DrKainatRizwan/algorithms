"""Number-theoretic primitives: modular arithmetic, primality, prime generation."""
from __future__ import annotations

import random

_SMALL_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71)


def egcd(a: int, b: int) -> tuple[int, int, int]:
    """Extended Euclid: return (g, x, y) with a*x + b*y == g == gcd(a, b)."""
    x0, y0, x1, y1 = 1, 0, 0, 1
    while b:
        q, a, b = a // b, b, a % b
        x0, x1 = x1, x0 - q * x1
        y0, y1 = y1, y0 - q * y1
    return a, x0, y0


def modinv(a: int, m: int) -> int:
    """Modular inverse of ``a`` modulo ``m``; raises ValueError if none exists."""
    g, x, _ = egcd(a % m, m)
    if g != 1:
        raise ValueError(f"{a} has no inverse modulo {m}")
    return x % m


def lcm(a: int, b: int) -> int:
    """Least common multiple of two positive integers."""
    return a // egcd(a, b)[0] * b


def crt(r1: int, m1: int, r2: int, m2: int) -> int:
    """Solve x = r1 (mod m1), x = r2 (mod m2) for coprime moduli."""
    inv = modinv(m1, m2)
    return (r1 + m1 * (((r2 - r1) * inv) % m2)) % (m1 * m2)


def is_probable_prime(n: int, rounds: int = 24, rng: random.Random | None = None) -> bool:
    """Miller-Rabin primality test with ``rounds`` random bases."""
    if n < 2:
        return False
    for p in _SMALL_PRIMES:
        if n == p:
            return True
        if n % p == 0:
            return False
    rng = rng or random.Random(n)
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for _ in range(rounds):
        a = rng.randrange(2, n - 1)
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def random_prime(bits: int, rng: random.Random) -> int:
    """Random prime with exactly ``bits`` bits (top two bits set)."""
    if bits < 3:
        raise ValueError("bits must be >= 3")
    while True:
        cand = rng.getrandbits(bits) | (1 << (bits - 1)) | (1 << (bits - 2)) | 1
        if is_probable_prime(cand, rng=rng):
            return cand


def random_coprime(n: int, rng: random.Random) -> int:
    """Uniform element of Z_n^* ."""
    while True:
        r = rng.randrange(1, n)
        if egcd(r, n)[0] == 1:
            return r
