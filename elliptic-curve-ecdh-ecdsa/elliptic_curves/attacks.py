"""Attacks on weak parameters: discrete-log solvers on toy curves and ECDSA nonce reuse."""

from __future__ import annotations

import random

from .curves import Curve, Point
from .ecdsa import bits2int
from .field import inv_mod, is_probable_prime, sqrt_mod


def baby_step_giant_step(curve: Curve, P: Point, Q: Point, order: int | None = None) -> int | None:
    """Solve ``k*P = Q`` in O(sqrt(n)) time and memory; returns ``None`` if no solution."""
    n = order or curve.n
    m = int(n ** 0.5) + 1
    table: dict[Point, int] = {}
    R: Point = None
    for j in range(m):
        table.setdefault(R, j)
        R = curve.add(R, P)
    step = curve.neg(curve.mul_naive(m, P))
    R = Q
    for i in range(m + 1):
        if R in table:
            return (i * m + table[R]) % n
        R = curve.add(R, step)
    return None


def pollard_rho(curve: Curve, P: Point, Q: Point, order: int | None = None,
                seed: int = 0, max_restarts: int = 50) -> int | None:
    """Pollard's rho with Floyd cycle detection and a 16-way additive walk; O(sqrt(n)) time, O(1) memory."""
    n = order or curve.n
    rng = random.Random(seed)
    for _ in range(max_restarts):
        steps = []
        for _i in range(16):
            a, b = rng.randrange(n), rng.randrange(n)
            steps.append((a, b, curve.add(curve.mul_naive(a, P), curve.mul_naive(b, Q))))

        def walk(state):
            X, a, b = state
            if X is None:
                return state
            sa, sb, S = steps[X[0] % 16]
            return curve.add(X, S), (a + sa) % n, (b + sb) % n

        a0, b0 = rng.randrange(n), rng.randrange(n)
        start = (curve.add(curve.mul_naive(a0, P), curve.mul_naive(b0, Q)), a0, b0)
        tort, hare = walk(start), walk(walk(start))
        limit = 20 * int(n ** 0.5) + 100
        for _step in range(limit):
            if tort[0] == hare[0]:
                break
            tort, hare = walk(tort), walk(walk(hare))
        else:
            continue
        da, db = (tort[1] - hare[1]) % n, (hare[2] - tort[2]) % n
        if db == 0:
            continue
        # da*P = db*k*P  =>  k = da / db (mod n); n prime so db is invertible
        k = da * inv_mod(db, n) % n
        if curve.mul_naive(k, P) == Q:
            return k
    return None


def recover_key_from_nonce_reuse(curve: Curve, digest1: bytes, sig1: tuple[int, int],
                                 digest2: bytes, sig2: tuple[int, int]) -> int:
    """Recover the private key from two signatures sharing a nonce (identical ``r``)."""
    (r1, s1), (r2, s2) = sig1, sig2
    if r1 != r2:
        raise ValueError("signatures do not share a nonce")
    n = curve.n
    z1, z2 = bits2int(curve, digest1), bits2int(curve, digest2)
    if (s1 - s2) % n == 0:
        raise ValueError("identical signatures carry no information")
    k = (z1 - z2) * inv_mod(s1 - s2, n) % n
    return (s1 * k - z1) * inv_mod(r1, n) % n


def count_points(p: int, a: int, b: int) -> int:
    """Naive O(p) point count of y^2 = x^3 + ax + b over F_p (includes infinity)."""
    squares: dict[int, int] = {}
    for y in range(p):
        squares[y * y % p] = squares.get(y * y % p, 0) + 1
    return 1 + sum(squares.get((x ** 3 + a * x + b) % p, 0) for x in range(p))


def find_prime_order_curve(p_min: int, seed: int = 0) -> Curve:
    """Search for a curve over a prime p >= p_min whose group order is prime (small p only)."""
    rng = random.Random(seed)
    p = p_min | 1
    while True:
        if p % 4 == 3 and is_probable_prime(p):
            for _ in range(40):
                a, b = rng.randrange(p), rng.randrange(1, p)
                if (4 * a ** 3 + 27 * b * b) % p == 0:
                    continue
                n = count_points(p, a, b)
                if n > 3 and is_probable_prime(n):
                    while True:
                        x = rng.randrange(p)
                        y = sqrt_mod(x ** 3 + a * x + b, p)
                        if y is not None and y != 0:
                            return Curve(f"toy{p}", p, a, b, x, y, n)
        p += 2
