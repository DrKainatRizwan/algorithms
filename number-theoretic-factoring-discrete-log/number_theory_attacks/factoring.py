"""Integer factorisation: trial division, Fermat, Pollard rho (Brent) and Pollard p-1."""
from __future__ import annotations

import random
from math import gcd, isqrt

from .primes import is_prime, sieve


def trial_division(n: int, bound: int = 10_000) -> tuple[list[int], int]:
    """Strip prime factors <= ``bound``. Returns (factors found, cofactor)."""
    found: list[int] = []
    for p in sieve(bound):
        if p * p > n:
            break
        while n % p == 0:
            found.append(p)
            n //= p
    return found, n


def fermat_factor(n: int, max_steps: int = 1_000_000) -> tuple[int, int] | None:
    """Fermat's method: effective when n = p*q with p, q close together."""
    if n % 2 == 0:
        return 2, n // 2
    a = isqrt(n)
    if a * a < n:
        a += 1
    for _ in range(max_steps):
        b2 = a * a - n
        b = isqrt(b2)
        if b * b == b2:
            p, q = a - b, a + b
            return (p, q) if p > 1 else None
        a += 1
    return None


def pollard_rho_brent(n: int, rng: random.Random | None = None,
                      max_tries: int = 20) -> int | None:
    """Brent's variant of Pollard's rho; returns a non-trivial factor of composite ``n``.

    Expected O(n^(1/4)) multiplications; batches gcds over blocks of ``m`` steps.
    """
    if n % 2 == 0:
        return 2
    rng = rng or random.Random(0)
    for _ in range(max_tries):
        y, c, m = rng.randrange(1, n), rng.randrange(1, n), 128
        g = r = q = 1
        x = ys = y
        while g == 1:
            x = y
            for _ in range(r):
                y = (y * y + c) % n
            k = 0
            while k < r and g == 1:
                ys = y
                for _ in range(min(m, r - k)):
                    y = (y * y + c) % n
                    q = q * abs(x - y) % n
                g = gcd(q, n)
                k += m
            r *= 2
        if g == n:
            g = 1
            while g == 1:  # backtrack one step at a time
                ys = (ys * ys + c) % n
                g = gcd(abs(x - ys), n)
        if g != n:
            return g
    return None


def pollard_p_minus_1(n: int, bound: int = 10_000) -> int | None:
    """Pollard p-1 (stage 1): finds p | n when p-1 is ``bound``-smooth."""
    a = 2
    for p in sieve(bound):
        pk = p
        while pk * p <= bound:
            pk *= p
        a = pow(a, pk, n)
    g = gcd(a - 1, n)
    return g if 1 < g < n else None


def factorize(n: int, rng: random.Random | None = None) -> dict[int, int]:
    """Complete factorisation as {prime: exponent}, combining the methods above."""
    if n < 1:
        raise ValueError("n must be positive")
    rng = rng or random.Random(0)
    out: dict[int, int] = {}

    def add(p: int) -> None:
        out[p] = out.get(p, 0) + 1

    small, rest = trial_division(n, 1000)
    for p in small:
        add(p)
    stack = [rest] if rest > 1 else []
    while stack:
        m = stack.pop()
        if is_prime(m):
            add(m)
            continue
        r = isqrt(m)
        if r * r == m:
            stack += [r, r]
            continue
        d = pollard_p_minus_1(m, 2000) or pollard_rho_brent(m, rng)
        if d is None:
            raise RuntimeError(f"failed to factor {m}")
        stack += [d, m // d]
    return dict(sorted(out.items()))
