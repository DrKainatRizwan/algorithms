"""Modular arithmetic helpers: inverses, square roots and primality testing."""

from __future__ import annotations

_SMALL_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def inv_mod(a: int, m: int) -> int:
    """Return the inverse of ``a`` modulo ``m`` using the extended Euclidean algorithm."""
    a %= m
    if a == 0:
        raise ZeroDivisionError("0 has no modular inverse")
    old_r, r = a, m
    old_s, s = 1, 0
    while r:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
    if old_r != 1:
        raise ValueError("value is not invertible modulo m")
    return old_s % m


def is_probable_prime(n: int) -> bool:
    """Miller-Rabin test; deterministic for n < 3.3e24 and overwhelmingly reliable beyond."""
    if n < 2:
        return False
    for p in _SMALL_PRIMES:
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in _SMALL_PRIMES:
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


def legendre(a: int, p: int) -> int:
    """Legendre symbol (a|p) for an odd prime p, returned as -1, 0 or 1."""
    r = pow(a % p, (p - 1) // 2, p)
    return -1 if r == p - 1 else r


def sqrt_mod(a: int, p: int) -> int | None:
    """Tonelli-Shanks square root of ``a`` modulo an odd prime ``p``; ``None`` if non-residue."""
    a %= p
    if a == 0:
        return 0
    if legendre(a, p) != 1:
        return None
    if p % 4 == 3:
        return pow(a, (p + 1) // 4, p)
    q, s = p - 1, 0
    while q % 2 == 0:
        q //= 2
        s += 1
    z = 2
    while legendre(z, p) != -1:
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
