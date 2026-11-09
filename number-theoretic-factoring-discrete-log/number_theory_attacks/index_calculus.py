"""Index calculus for discrete logs modulo a prime (sub-exponential, small-prime factor base)."""
from __future__ import annotations

import random
from math import exp, log, sqrt

from .factoring import factorize
from .primes import crt, sieve


def _smooth_exponents(v: int, base: list[int]) -> list[int] | None:
    exps = []
    for q in base:
        e = 0
        while v % q == 0:
            v //= q
            e += 1
        exps.append(e)
    return exps if v == 1 else None


def _solve_mod_prime_power(rows: list[list[int]], rhs: list[int], q: int, e: int,
                           nvars: int) -> list[int] | None:
    """Gauss-Jordan over Z/q^e pivoting only on units; None if under-determined."""
    m = q ** e
    A = [[v % m for v in r] + [b % m] for r, b in zip(rows, rhs)]
    row = 0
    pivot_cols = []
    for col in range(nvars):
        sel = next((i for i in range(row, len(A)) if A[i][col] % q), None)
        if sel is None:
            return None
        A[row], A[sel] = A[sel], A[row]
        inv = pow(A[row][col], -1, m)
        A[row] = [v * inv % m for v in A[row]]
        for i in range(len(A)):
            if i != row and A[i][col]:
                f = A[i][col]
                A[i] = [(a - f * b) % m for a, b in zip(A[i], A[row])]
        pivot_cols.append(col)
        row += 1
    return [A[i][nvars] for i in range(nvars)]


def index_calculus(g: int, h: int, p: int, rng: random.Random | None = None,
                   bound: int | None = None, max_tries: int = 2_000_000) -> int | None:
    """Solve g^x = h (mod p) for a generator ``g`` of (Z/pZ)*.

    Phase 1: collect relations g^k = prod q_i^e_i over a factor base, giving linear
    equations k = sum e_i log(q_i) mod (p-1). Phase 2: solve them modulo each prime
    power of p-1 and combine with CRT. Phase 3: find s with h*g^s smooth and read off x.
    """
    rng = rng or random.Random(0)
    ln = log(p)
    bound = bound or max(20, int(exp(0.5 * sqrt(ln * log(ln))) * 1.5))
    base = sieve(bound)
    n = len(base)
    order = p - 1
    rows: list[list[int]] = []
    rhs: list[int] = []
    tries = 0
    logs: list[int] | None = None
    while logs is None:
        while len(rows) < n + 10 and tries < max_tries:
            tries += 1
            k = rng.randrange(1, order)
            ex = _smooth_exponents(pow(g, k, p), base)
            if ex is not None:
                rows.append(ex)
                rhs.append(k)
        if len(rows) < n + 10:
            return None
        res, mods = [], []
        for q, e in factorize(order).items():
            sol = _solve_mod_prime_power(rows, rhs, q, e, n)
            if sol is None:
                break
            res.append(sol)
            mods.append(q ** e)
        else:
            logs = [crt([r[i] for r in res], mods)[0] for i in range(n)]
            continue
        rows_needed = len(rows) + 10  # collect more relations and retry
        while len(rows) < rows_needed and tries < max_tries:
            tries += 1
            k = rng.randrange(1, order)
            ex = _smooth_exponents(pow(g, k, p), base)
            if ex is not None:
                rows.append(ex)
                rhs.append(k)
        if tries >= max_tries:
            return None
    for _ in range(max_tries):
        s = rng.randrange(order)
        ex = _smooth_exponents(h * pow(g, s, p) % p, base)
        if ex is not None:
            x = (sum(e * l for e, l in zip(ex, logs)) - s) % order
            if pow(g, x, p) == h % p:
                return x
    return None
