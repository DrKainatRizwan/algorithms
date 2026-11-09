"""Elementary number theory: primality, modular arithmetic and square roots mod p."""
from __future__ import annotations

import random
from math import gcd

_MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41)  # deterministic below 3.3e24


def sieve(limit: int) -> list[int]:
    """Return all primes <= ``limit`` using the sieve of Eratosthenes (O(n log log n))."""
    if limit < 2:
        return []
    flags = bytearray([1]) * (limit + 1)
    flags[0:2] = b"\x00\x00"
    for i in range(2, int(limit ** 0.5) + 1):
        if flags[i]:
            flags[i * i::i] = bytearray(len(range(i * i, limit + 1, i)))
    return [i for i, f in enumerate(flags) if f]


def is_prime(n: int) -> bool:
    """Miller-Rabin primality test with fixed bases (deterministic for n < 3.3e24)."""
    if n < 2:
        return False
    for p in _MR_BASES:
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in _MR_BASES:
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


def next_prime(n: int) -> int:
    """Smallest prime strictly greater than ``n``."""
    c = n + 1
    if c <= 2:
        return 2
    if c % 2 == 0:
        c += 1
    while not is_prime(c):
        c += 2
    return c


def random_prime(bits: int, rng: random.Random) -> int:
    """Random prime with exactly ``bits`` bits."""
    if bits < 2:
        raise ValueError("bits must be >= 2")
    while True:
        c = rng.getrandbits(bits) | (1 << (bits - 1)) | 1
        if is_prime(c):
            return c


def egcd(a: int, b: int) -> tuple[int, int, int]:
    """Extended Euclid: return (g, x, y) with a*x + b*y = g = gcd(a, b)."""
    x0, y0, x1, y1 = 1, 0, 0, 1
    while b:
        q, a, b = a // b, b, a % b
        x0, x1 = x1, x0 - q * x1
        y0, y1 = y1, y0 - q * y1
    return a, x0, y0


def modinv(a: int, m: int) -> int:
    """Modular inverse of ``a`` mod ``m``; raises ValueError if not invertible."""
    g, x, _ = egcd(a % m, m)
    if g != 1:
        raise ValueError(f"{a} has no inverse modulo {m}")
    return x % m


def crt(residues: list[int], moduli: list[int]) -> tuple[int, int]:
    """Chinese remainder theorem for pairwise coprime moduli: returns (x, M)."""
    x, m = 0, 1
    for r, n in zip(residues, moduli):
        t = ((r - x) * modinv(m, n)) % n
        x += m * t
        m *= n
    return x % m, m


def jacobi(a: int, n: int) -> int:
    """Jacobi symbol (a/n) for odd positive ``n``."""
    if n <= 0 or n % 2 == 0:
        raise ValueError("n must be odd and positive")
    a %= n
    result = 1
    while a:
        while a % 2 == 0:
            a //= 2
            if n % 8 in (3, 5):
                result = -result
        a, n = n, a
        if a % 4 == 3 and n % 4 == 3:
            result = -result
        a %= n
    return result if n == 1 else 0


def tonelli_shanks(a: int, p: int) -> int:
    """Square root of ``a`` modulo odd prime ``p`` (raises if ``a`` is a non-residue)."""
    a %= p
    if a == 0:
        return 0
    if jacobi(a, p) != 1:
        raise ValueError("not a quadratic residue")
    if p % 4 == 3:
        return pow(a, (p + 1) // 4, p)
    q, s = p - 1, 0
    while q % 2 == 0:
        q //= 2
        s += 1
    z = 2
    while jacobi(z, p) != -1:
        z += 1
    m, c, t, r = s, pow(z, q, p), pow(a, q, p), pow(a, (q + 1) // 2, p)
    while t != 1:
        i, t2 = 0, t
        while t2 != 1:
            t2 = t2 * t2 % p
            i += 1
        b = pow(c, 1 << (m - i - 1), p)
        m, c = i, b * b % p
        t, r = t * c % p, r * b % p
    return r


def gcd_all(values: list[int]) -> int:
    """GCD of a list of integers."""
    g = 0
    for v in values:
        g = gcd(g, v)
    return g
