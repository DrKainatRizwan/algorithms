"""Discrete logarithm algorithms in (Z/pZ)*: BSGS, Pollard rho and Pohlig-Hellman."""
from __future__ import annotations

import random
from math import gcd, isqrt

from .factoring import factorize
from .primes import crt, modinv


def baby_step_giant_step(g: int, h: int, p: int, order: int | None = None) -> int | None:
    """Solve g^x = h (mod p) with 0 <= x < order in O(sqrt(order)) time and space."""
    order = order or p - 1
    m = isqrt(order) + 1
    table: dict[int, int] = {}
    cur = 1
    for j in range(m):
        table.setdefault(cur, j)
        cur = cur * g % p
    factor = pow(modinv(g, p), m, p)
    gamma = h % p
    for i in range(m + 1):
        j = table.get(gamma)
        if j is not None:
            x = i * m + j
            if x < order or pow(g, x, p) == h % p:
                return x % order
        gamma = gamma * factor % p
    return None


def pollard_rho_dlog(g: int, h: int, p: int, order: int,
                     rng: random.Random | None = None, max_tries: int = 50) -> int | None:
    """Pollard's rho for discrete logs: O(sqrt(order)) time, O(1) memory.

    A pseudo-random walk x -> {h*x, x^2, g*x} (chosen by x mod 3) tracks the exponents
    (a, b) with x = g^a h^b. Floyd cycle detection yields a collision from which x is
    solved modulo ``order`` (resolving the gcd ambiguity by trying all lifts).
    """
    rng = rng or random.Random(0)
    h %= p

    def step(x: int, a: int, b: int) -> tuple[int, int, int]:
        s = x % 3
        if s == 0:
            return x * h % p, a, (b + 1) % order
        if s == 1:
            return x * x % p, 2 * a % order, 2 * b % order
        return x * g % p, (a + 1) % order, b

    for _ in range(max_tries):
        a0, b0 = rng.randrange(order), rng.randrange(order)
        x0 = pow(g, a0, p) * pow(h, b0, p) % p
        tort, hare = (x0, a0, b0), (x0, a0, b0)
        while True:
            tort = step(*tort)
            hare = step(*step(*hare))
            if tort[0] == hare[0]:
                break
        db = (tort[2] - hare[2]) % order
        da = (hare[1] - tort[1]) % order
        d = gcd(db, order)
        if d == order:
            continue
        if da % d:
            continue
        o2 = order // d
        base = da // d * modinv(db // d, o2) % o2 if o2 > 1 else 0
        for k in range(d):
            x = base + k * o2
            if pow(g, x, p) == h:
                return x
    return None


def pohlig_hellman(g: int, h: int, p: int, order: int | None = None) -> int | None:
    """Solve g^x = h (mod p) when the order of g is smooth.

    Reduces to prime-power subgroups, lifts digit by digit with BSGS in each prime-order
    subgroup, then recombines with CRT. Cost O(sum e_i (log n + sqrt(q_i))).
    """
    order = order or p - 1
    if pow(h, order, p) != 1:
        return None
    residues, moduli = [], []
    for q, e in factorize(order).items():
        qe = q ** e
        gq = pow(g, order // qe, p)
        hq = pow(h, order // qe, p)
        gamma = pow(gq, q ** (e - 1), p)  # element of order q
        x = 0
        for k in range(e):
            target = pow(hq * pow(modinv(gq, p), x, p) % p, q ** (e - 1 - k), p)
            dk = baby_step_giant_step(gamma, target, p, q)
            if dk is None:
                return None
            x += dk * q ** k
        residues.append(x)
        moduli.append(qe)
    x, _ = crt(residues, moduli)
    return x if pow(g, x, p) == h % p else None
