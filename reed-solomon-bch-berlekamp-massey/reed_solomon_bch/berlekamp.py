"""Solvers for the error-locator polynomial from a syndrome sequence.

Two algorithms are provided and cross-checked in the tests:

* :func:`berlekamp_massey` -- finds the shortest LFSR generating the
  syndromes in ``O(N^2)`` field operations.
* :func:`pgz_locator` -- the Peterson-Gorenstein-Zierler method, which solves
  a Hankel system and costs ``O(t^4)`` with naive elimination.
"""

from __future__ import annotations

from .galois import GF2m, Poly


def berlekamp_massey(field: GF2m, s: list[int]) -> tuple[Poly, int]:
    """Berlekamp-Massey over GF(2^m).

    Returns ``(C, L)`` where ``C(x) = 1 + C_1 x + ... + C_L x^L`` satisfies
    ``s[n] + sum_{i=1..L} C_i s[n-i] = 0`` for all ``L <= n < len(s)`` and
    ``L`` is the linear complexity of the sequence.
    """
    c: Poly = [1]
    b: Poly = [1]
    big_l = 0
    m = 1
    bd = 1  # last non-zero discrepancy
    for n in range(len(s)):
        d = s[n]
        for i in range(1, big_l + 1):
            if i < len(c):
                d ^= field.mul(c[i], s[n - i])
        if d == 0:
            m += 1
            continue
        coef = field.div(d, bd)
        shifted = [0] * m + field.poly_scale(b, coef)
        if 2 * big_l <= n:
            prev = c[:]
            c = field.poly_add(c, shifted)
            big_l = n + 1 - big_l
            b, bd, m = prev, d, 1
        else:
            c = field.poly_add(c, shifted)
            m += 1
    c = field.trim(c)
    return c + [0] * (big_l + 1 - len(c)), big_l


def pgz_locator(field: GF2m, s: list[int], t: int) -> tuple[Poly, int]:
    """Peterson-Gorenstein-Zierler locator for up to ``t`` errors.

    Tries ``nu = t, t-1, ..., 1`` and returns the locator for the first
    non-singular Hankel system.  If every syndrome is zero, ``([1], 0)``.
    """
    if not any(s[: 2 * t]):
        return [1], 0
    for nu in range(t, 0, -1):
        a = [[s[j + c] for c in range(nu)] for j in range(nu)]
        rhs = [s[j + nu] for j in range(nu)]
        x = field.solve(a, rhs)
        if x is not None:
            # x[c] = Lambda_{nu-c}
            return [1] + [x[nu - i] for i in range(1, nu + 1)], nu
    return [1], 0
