"""Classical number-theory helpers used by the order-finding pipeline.

Everything here is implemented from scratch on Python integers: Euclid's
algorithm, modular exponentiation, Miller-Rabin, perfect-power detection and
continued-fraction expansions.
"""
from __future__ import annotations

from math import isqrt


def gcd(a: int, b: int) -> int:
    """Greatest common divisor via Euclid's algorithm."""
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a


def egcd(a: int, b: int) -> tuple[int, int, int]:
    """Extended Euclid: return ``(g, x, y)`` with ``a*x + b*y == g``."""
    x0, y0, x1, y1 = 1, 0, 0, 1
    while b:
        q, r = divmod(a, b)
        a, b = b, r
        x0, x1 = x1, x0 - q * x1
        y0, y1 = y1, y0 - q * y1
    return a, x0, y0


def modinv(a: int, n: int) -> int:
    """Modular inverse of ``a`` modulo ``n``; raises if it does not exist."""
    g, x, _ = egcd(a % n, n)
    if g != 1:
        raise ValueError(f"{a} has no inverse modulo {n}")
    return x % n


def modpow(base: int, exp: int, mod: int) -> int:
    """Square-and-multiply modular exponentiation."""
    if mod == 1:
        return 0
    result, base = 1, base % mod
    while exp > 0:
        if exp & 1:
            result = result * base % mod
        base = base * base % mod
        exp >>= 1
    return result


def is_probable_prime(n: int) -> bool:
    """Deterministic Miller-Rabin for n < 3.3e24 (first 12 prime bases)."""
    if n < 2:
        return False
    small = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for p in small:
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in small:
        x = modpow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def integer_root(n: int, k: int) -> int:
    """Floor of the k-th root of ``n`` by binary search."""
    if n < 2 or k == 1:
        return n
    lo, hi = 1, 1 << (n.bit_length() // k + 1)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if mid**k <= n:
            lo = mid
        else:
            hi = mid - 1
    return lo


def perfect_power(n: int) -> tuple[int, int] | None:
    """Return ``(base, exponent)`` with ``base**exponent == n`` and exponent>1."""
    for k in range(2, n.bit_length() + 1):
        r = integer_root(n, k)
        if r > 1 and r**k == n:
            return r, k
    return None


def multiplicative_order(a: int, n: int) -> int:
    """Smallest r > 0 with a**r = 1 (mod n), by brute force (reference)."""
    if gcd(a, n) != 1:
        raise ValueError("a and n must be coprime")
    if n == 1:
        return 1
    x, r = a % n, 1
    while x != 1:
        x = x * a % n
        r += 1
    return r


def euler_phi(n: int) -> int:
    """Euler's totient by trial factorisation."""
    result, m, p = n, n, 2
    while p * p <= m:
        if m % p == 0:
            while m % p == 0:
                m //= p
            result -= result // p
        p += 1
    if m > 1:
        result -= result // m
    return result


def continued_fraction(num: int, den: int) -> list[int]:
    """Continued-fraction coefficients of ``num/den``."""
    out: list[int] = []
    while den:
        q, r = divmod(num, den)
        out.append(q)
        num, den = den, r
    return out


def convergents(num: int, den: int) -> list[tuple[int, int]]:
    """All convergents ``(p, q)`` of ``num/den`` in increasing order."""
    h0, h1, k0, k1 = 0, 1, 1, 0
    result = []
    for a in continued_fraction(num, den):
        h0, h1 = h1, a * h1 + h0
        k0, k1 = k1, a * k1 + k0
        result.append((h1, k1))
    return result


def recover_denominator(y: int, q: int, bound: int) -> int:
    """Largest convergent denominator of ``y/q`` that is below ``bound``.

    Returns 0 when the measurement is zero (carries no information).
    """
    best = 0
    for _, d in convergents(y, q):
        if 0 < d < bound:
            best = d
    return best if y else 0
