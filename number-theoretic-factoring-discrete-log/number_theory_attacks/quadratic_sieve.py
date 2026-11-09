"""Single-polynomial quadratic sieve with GF(2) dependency search."""
from __future__ import annotations

from math import exp, gcd, isqrt, log, sqrt

from .primes import jacobi, sieve, tonelli_shanks


def _smooth_bound(n: int) -> int:
    ln = log(n)
    return max(200, int(exp(0.5 * sqrt(ln * log(ln))) * 1.5))


def _factor_over_base(v: int, base: list[int]) -> dict[int, int] | None:
    exps: dict[int, int] = {}
    for p in base:
        while v % p == 0:
            v //= p
            exps[p] = exps.get(p, 0) + 1
    return exps if v == 1 else None


def quadratic_sieve(n: int, bound: int | None = None, block: int = 20_000,
                    max_blocks: int = 200) -> int | None:
    """Return a non-trivial factor of odd composite ``n`` (not a prime power).

    Collects relations x^2 - n = prod(p^e) over a factor base, finds subsets whose
    exponent vectors sum to zero mod 2 (online Gaussian elimination on bitmasks) and
    turns each dependency into a congruence of squares X^2 = Y^2 (mod n).
    """
    if n % 2 == 0:
        return 2
    r = isqrt(n)
    if r * r == n:
        return r
    bound = bound or _smooth_bound(n)
    base = [2] + [p for p in sieve(bound)[1:] if jacobi(n, p) == 1]
    for p in base:
        if n % p == 0:
            return p
    index = {p: i for i, p in enumerate(base)}
    roots = {p: [tonelli_shanks(n, p)] for p in base[1:]}
    for p in roots:
        roots[p].append(p - roots[p][0])
    logs = {p: log(p) for p in base}
    x0 = r + 1
    pivots: dict[int, tuple[int, int]] = {}  # leading bit -> (vector, combination mask)
    rels: list[tuple[int, dict[int, int]]] = []

    for blk in range(max_blocks):
        lo = x0 + blk * block
        acc = [0.0] * block
        for p in base:
            if p == 2:
                start = (1 - lo) % 2
                for i in range(start, block, 2):
                    acc[i] += logs[2]
                continue
            for rt in roots[p]:
                for i in range((rt - lo) % p, block, p):
                    acc[i] += logs[p]
        for i in range(block):
            x = lo + i
            q = x * x - n
            if acc[i] < log(q) - 2.0 * log(bound):
                continue
            exps = _factor_over_base(q, base)
            if exps is None:
                continue
            vec = 0
            for p, e in exps.items():
                if e & 1:
                    vec |= 1 << index[p]
            k = len(rels)
            rels.append((x, exps))
            comb = 1 << k
            while vec:
                lead = vec.bit_length() - 1
                if lead in pivots:
                    pv, pc = pivots[lead]
                    vec ^= pv
                    comb ^= pc
                else:
                    pivots[lead] = (vec, comb)
                    break
            if vec == 0:
                f = _try_dependency(n, rels, comb)
                if f:
                    return f
    return None


def _try_dependency(n: int, rels: list[tuple[int, dict[int, int]]], comb: int) -> int | None:
    X, total = 1, {}
    k = 0
    while comb:
        if comb & 1:
            x, exps = rels[k]
            X = X * x % n
            for p, e in exps.items():
                total[p] = total.get(p, 0) + e
        comb >>= 1
        k += 1
    Y = 1
    for p, e in total.items():
        Y = Y * pow(p, e // 2, n) % n
    g = gcd(X - Y, n)
    return g if 1 < g < n else None
