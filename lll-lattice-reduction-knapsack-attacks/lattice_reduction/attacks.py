"""Cryptanalytic applications of lattice reduction.

* low-density subset-sum (knapsack) attack of Coster et al. (CJLOSS)
* integer-relation detection / minimal polynomial recovery
* state recovery for a truncated linear congruential generator
"""
from __future__ import annotations

import random
from decimal import Decimal
from fractions import Fraction
from math import log2
from typing import Sequence

from .bkz import bkz_reduce
from .enumeration import babai_nearest_plane, cvp
from .linalg import int_sqrt_ceil
from .lll import lll_reduce


# ----------------------------------------------------------------- knapsack
def knapsack_density(weights: Sequence[int]) -> float:
    """Density ``n / log2(max a_i)`` of a subset-sum instance."""
    return len(weights) / log2(max(weights))


def random_knapsack(n: int, density: float, rng: random.Random) -> tuple[list[int], list[int], int]:
    """Random instance with the requested density.

    Returns ``(weights, secret_bits, target)``.
    """
    bits = max(2, round(n / density))
    weights = [rng.randrange(2 ** (bits - 1), 2 ** bits) for _ in range(n)]
    secret = [rng.randrange(2) for _ in range(n)]
    return weights, secret, sum(w for w, s in zip(weights, secret) if s)


def knapsack_lattice(weights: Sequence[int], target: int) -> list[list[int]]:
    """CJLOSS basis: rows ``(2 e_i | N a_i)`` and ``(1,...,1 | N s)``."""
    n = len(weights)
    big_n = int_sqrt_ceil(n) + 1
    rows = []
    for i, a in enumerate(weights):
        row = [0] * (n + 1)
        row[i] = 2
        row[n] = big_n * a
        rows.append(row)
    rows.append([1] * n + [big_n * target])
    return rows


def _extract_solution(rows: Sequence[Sequence[int]], weights: Sequence[int], target: int) -> list[int] | None:
    n = len(weights)
    for row in rows:
        if row[n] != 0 or any(abs(v) != 1 for v in row[:n]):
            continue
        for sign in (1, -1):
            x = [(1 - sign * v) // 2 for v in row[:n]]
            if sum(a * xi for a, xi in zip(weights, x)) == target:
                return x
    return None


def solve_knapsack(weights: Sequence[int], target: int, block_size: int = 0) -> list[int] | None:
    """Recover ``x in {0,1}^n`` with ``sum a_i x_i = target`` or return None.

    LLL is tried first; if that fails and ``block_size >= 3`` BKZ is run.
    """
    rows = knapsack_lattice(weights, target)
    red = lll_reduce(rows, Fraction(99, 100))
    sol = _extract_solution(red, weights, target)
    if sol is None and block_size >= 3:
        red = bkz_reduce(red, block_size)
        sol = _extract_solution(red, weights, target)
    return sol


# ----------------------------------------------------------- integer relations
def integer_relation(values: Sequence[Decimal | Fraction | int], digits: int,
                     tol_digits: int | None = None,
                     max_coeff: int | None = None) -> list[int] | None:
    """Find small integers ``c`` with ``sum c_i v_i ~ 0``.

    The values are scaled by ``10**digits`` and the lattice with rows
    ``(e_i | round(C v_i))`` is LLL-reduced; a short row with a tiny last
    coordinate is a relation.  Returns None if no relation passes the
    residual test ``|sum c_i v_i| < 10**-(tol_digits)`` and have all
    ``|c_i| <= max_coeff``.  By pigeonhole, spurious relations exist with
    coefficients near ``10**(digits/n)``, so the default bound is
    ``10**(digits // (2n))``.
    """
    n = len(values)
    if max_coeff is None:
        max_coeff = 10 ** max(1, digits // (2 * n))
    tol_digits = tol_digits if tol_digits is not None else digits // 2
    scale = 10 ** digits
    fv = [Fraction(v) for v in values]
    rows = []
    for i, v in enumerate(fv):
        row = [0] * (n + 1)
        row[i] = 1
        row[n] = round(v * scale)
        rows.append(row)
    red = lll_reduce(rows, Fraction(99, 100))
    for row in red:
        coeffs = row[:n]
        if not any(coeffs) or max(abs(c) for c in coeffs) > max_coeff:
            continue
        resid = abs(sum(c * v for c, v in zip(coeffs, fv)))
        if resid < Fraction(1, 10 ** tol_digits):
            return list(coeffs)
    return None


def minimal_polynomial(alpha: Decimal, degree: int, digits: int) -> list[int] | None:
    """Integer polynomial of degree ``degree`` vanishing at ``alpha``.

    Coefficients are returned from the constant term upward with a
    positive leading coefficient.
    """
    powers = [Decimal(1)]
    for _ in range(degree):
        powers.append(powers[-1] * alpha)
    rel = integer_relation(powers, digits)
    if rel is None:
        return None
    lead = next(c for c in reversed(rel) if c != 0)
    return [c if lead > 0 else -c for c in rel]


# --------------------------------------------------------- truncated LCG
class TruncatedLCG:
    """``x <- a x + c (mod m)`` that reveals only the top bits ``x >> hidden_bits``."""

    def __init__(self, a: int, c: int, m: int, hidden_bits: int, seed: int) -> None:
        self.a, self.c, self.m, self.k = a, c, m, hidden_bits
        self.state = seed % m

    def next(self) -> tuple[int, int]:
        """Advance and return ``(full_state, visible_output)``."""
        self.state = (self.a * self.state + self.c) % self.m
        return self.state, self.state >> self.k


def recover_lcg_states(outputs: Sequence[int], a: int, c: int, m: int, hidden_bits: int) -> list[int] | None:
    """Recover the full internal states from truncated outputs.

    Writes ``x_i = y_i 2^k + e_i`` with ``0 <= e_i < 2^k``.  Then
    ``e - p`` lies in the lattice ``{v : v_{i+1} = a v_i mod m}``; the
    centred error vector is a bounded-distance decoding target solved with
    CVP.  Returns None when the recovered states are inconsistent.
    """
    n = len(outputs)
    k = hidden_bits
    ys = [y << k for y in outputs]
    p = [0] * n
    for i in range(n - 1):
        t = (a * ys[i] + c - ys[i + 1]) % m
        p[i + 1] = (a * p[i] + t) % m
    basis = [[pow(a, j, m) for j in range(n)]]
    for j in range(1, n):
        row = [0] * n
        row[j] = m
        basis.append(row)
    h = (1 << k) // 2
    target = [pi - h for pi in p]
    near = cvp(basis, target)
    e = [h + ti - li for ti, li in zip(target, near)]
    if not all(0 <= ei < (1 << k) for ei in e):
        return None
    states = [y + ei for y, ei in zip(ys, e)]
    for i in range(n - 1):
        if (a * states[i] + c) % m != states[i + 1]:
            return None
    return states


def predict_next(states: Sequence[int], a: int, c: int, m: int, count: int) -> list[int]:
    """Continue the generator for ``count`` steps from the last known state."""
    s = states[-1]
    out = []
    for _ in range(count):
        s = (a * s + c) % m
        out.append(s)
    return out
