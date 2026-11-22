"""Algebraic (linearisation) attack on filter generators.

Every register bit is a known linear form in the unknown initial state ``x``.
A filter of algebraic degree ``d`` therefore turns each keystream bit into a
polynomial equation of degree ``d`` in ``x``.  Treating each monomial of degree
at most ``d`` as an independent unknown gives a linear system over GF(2) which is
solved by Gaussian elimination; the degree-one unknowns are the key.
"""
from __future__ import annotations

from itertools import combinations
from typing import Sequence

import numpy as np

from .generators import FilterGenerator
from .gf2 import solve

Mono = frozenset  # a monomial = set of variable indices (x_i^2 = x_i)
Poly = set  # a polynomial over GF(2) = set of monomials


def linear_form(mask: int) -> Poly:
    """Polynomial ``sum_{i in mask} x_i``."""
    out: Poly = set()
    i = 0
    while mask >> i:
        if mask >> i & 1:
            out.add(Mono([i]))
        i += 1
    return out


def poly_mul(a: Poly, b: Poly) -> Poly:
    """Multiply two GF(2) polynomials using ``x_i^2 = x_i``."""
    out: Poly = set()
    for m1 in a:
        for m2 in b:
            out ^= {m1 | m2}
    return out


def monomial_index(nvars: int, degree: int) -> dict[Mono, int]:
    """Column index for every monomial of degree 1..``degree``."""
    idx: dict[Mono, int] = {}
    for d in range(1, degree + 1):
        for c in combinations(range(nvars), d):
            idx[Mono(c)] = len(idx)
    return idx


def keystream_equations(gen: FilterGenerator, nbits: int) -> list[Poly]:
    """Symbolic polynomial (in the initial state) for each keystream bit."""
    n = gen.lfsr.length
    ms = gen.lfsr.masks(nbits + n)
    lin = [linear_form(m) for m in ms]
    eqs: list[Poly] = []
    for t in range(nbits):
        total: Poly = set()
        for mono in gen.anf:
            if not mono:
                total ^= {Mono()}
                continue
            prod = lin[t + gen.positions[mono[0]]]
            for i in mono[1:]:
                prod = poly_mul(prod, lin[t + gen.positions[i]])
            total ^= prod
        eqs.append(total)
    return eqs


def algebraic_attack(gen: FilterGenerator, z: Sequence[int], nbits: int | None = None) -> int | None:
    """Recover the initial state from noise-free keystream ``z``.

    Returns ``None`` if the linearised system is inconsistent or fails to
    determine the state uniquely.
    """
    n = gen.lfsr.length
    d = gen.degree
    idx = monomial_index(n, d)
    ncols = len(idx)
    nbits = nbits if nbits is not None else min(len(z), ncols + 20)
    eqs = keystream_equations(gen, nbits)
    rows, rhs = [], []
    for poly, bit in zip(eqs, z[:nbits]):
        row, const = 0, 0
        for mono in poly:
            if len(mono) == 0:
                const ^= 1
            else:
                row |= 1 << idx[mono]
        rows.append(row)
        rhs.append(int(bit) ^ const)
    res = solve(rows, rhs, ncols)
    if res is None:
        return None
    sol, null = res
    lin_cols = [idx[Mono([i])] for i in range(n)]
    if any(v >> c & 1 for v in null for c in lin_cols):
        return None  # key bits not uniquely determined
    state = 0
    for i, c in enumerate(lin_cols):
        if sol >> c & 1:
            state |= 1 << i
    return state


def unknown_count(n: int, degree: int) -> int:
    """Number of linearised unknowns ``sum_{k=1..d} C(n, k)``."""
    from math import comb

    return sum(comb(n, k) for k in range(1, degree + 1))
