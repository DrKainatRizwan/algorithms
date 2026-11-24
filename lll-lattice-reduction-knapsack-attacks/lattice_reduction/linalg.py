"""Exact and floating-point linear algebra helpers for integer lattices.

Bases are stored as lists of integer row vectors.  Gram-Schmidt data is
computed over the rationals so that reduction criteria can be checked
without rounding error.
"""
from __future__ import annotations

from fractions import Fraction
from math import isqrt, log2
from typing import Sequence

Vector = list[int]
Matrix = list[Vector]


def dot(u: Sequence[int], v: Sequence[int]) -> int:
    """Integer inner product."""
    return sum(a * b for a, b in zip(u, v))


def norm_sq(v: Sequence[int]) -> int:
    """Squared Euclidean norm of an integer vector."""
    return dot(v, v)


def copy_basis(basis: Sequence[Sequence[int]]) -> Matrix:
    """Deep copy of a basis as lists of Python ints."""
    return [[int(x) for x in row] for row in basis]


def gram_schmidt(basis: Sequence[Sequence[int]]) -> tuple[list[list[Fraction]], list[Fraction]]:
    """Exact Gram-Schmidt orthogonalisation.

    Returns ``(mu, B)`` where ``mu[i][j]`` (j < i) are the projection
    coefficients and ``B[i] = |b_i*|^2``.  Raises ``ValueError`` if the
    rows are linearly dependent.
    """
    n = len(basis)
    mu = [[Fraction(0)] * n for _ in range(n)]
    B: list[Fraction] = []
    for i in range(n):
        for j in range(i):
            mu[i][j] = Fraction(dot(basis[i], basis[j]))
            for k in range(j):
                mu[i][j] -= mu[j][k] * mu[i][k] * B[k]
            mu[i][j] /= B[j]
        bi = Fraction(norm_sq(basis[i]))
        for j in range(i):
            bi -= mu[i][j] ** 2 * B[j]
        if bi <= 0:
            raise ValueError("basis vectors are linearly dependent")
        B.append(bi)
    return mu, B


def gram_schmidt_float(basis: Sequence[Sequence[int]]) -> tuple[list[list[float]], list[float]]:
    """Gram-Schmidt data converted to floats (computed exactly first)."""
    mu, B = gram_schmidt(basis)
    return [[float(x) for x in row] for row in mu], [float(x) for x in B]


def lattice_det_sq(basis: Sequence[Sequence[int]]) -> Fraction:
    """Squared lattice determinant, i.e. the product of the ``|b_i*|^2``."""
    _, B = gram_schmidt(basis)
    out = Fraction(1)
    for x in B:
        out *= x
    return out


def is_size_reduced(basis: Sequence[Sequence[int]], eta: Fraction = Fraction(1, 2)) -> bool:
    """Check ``|mu_ij| <= eta`` for all ``j < i``."""
    mu, _ = gram_schmidt(basis)
    return all(abs(mu[i][j]) <= eta for i in range(len(basis)) for j in range(i))


def is_lll_reduced(basis: Sequence[Sequence[int]], delta: Fraction = Fraction(3, 4)) -> bool:
    """Check the size-reduction and Lovasz conditions exactly."""
    mu, B = gram_schmidt(basis)
    n = len(basis)
    for i in range(n):
        for j in range(i):
            if abs(mu[i][j]) > Fraction(1, 2):
                return False
    for k in range(1, n):
        if B[k] < (delta - mu[k][k - 1] ** 2) * B[k - 1]:
            return False
    return True


def orthogonality_defect(basis: Sequence[Sequence[int]]) -> float:
    """``prod |b_i| / det(L)``; equals 1 exactly for orthogonal bases."""
    _, B = gram_schmidt(basis)
    num = 0.0
    den = 0.0
    for v, b in zip(basis, B):
        num += log2(norm_sq(v)) / 2
        den += log2(float(b)) / 2
    return 2 ** (num - den)


def hermite_factor(basis: Sequence[Sequence[int]]) -> float:
    """Root Hermite factor ``(|b_1| / det^(1/n))^(1/n)``."""
    n = len(basis)
    _, B = gram_schmidt(basis)
    log_det = sum(log2(float(b)) / 2 for b in B)
    log_b1 = log2(norm_sq(basis[0])) / 2
    return 2 ** ((log_b1 - log_det / n) / n)


def matrix_inverse(m: Sequence[Sequence[int]]) -> list[list[Fraction]]:
    """Exact inverse of a square integer matrix via Gauss-Jordan."""
    n = len(m)
    a = [[Fraction(x) for x in row] + [Fraction(int(i == j)) for j in range(n)]
         for i, row in enumerate(m)]
    for col in range(n):
        piv = next((r for r in range(col, n) if a[r][col] != 0), None)
        if piv is None:
            raise ValueError("matrix is singular")
        a[col], a[piv] = a[piv], a[col]
        p = a[col][col]
        a[col] = [x / p for x in a[col]]
        for r in range(n):
            if r != col and a[r][col] != 0:
                f = a[r][col]
                a[r] = [x - f * y for x, y in zip(a[r], a[col])]
    return [row[n:] for row in a]


def same_lattice(b1: Sequence[Sequence[int]], b2: Sequence[Sequence[int]]) -> bool:
    """True iff two square full-rank bases generate the same lattice."""
    if len(b1) != len(b2):
        return False
    inv = matrix_inverse(b2)
    for row in b1:
        for j in range(len(b2)):
            if sum(row[k] * inv[k][j] for k in range(len(b2))) .denominator != 1:
                return False
    inv = matrix_inverse(b1)
    for row in b2:
        for j in range(len(b1)):
            if sum(row[k] * inv[k][j] for k in range(len(b1))).denominator != 1:
                return False
    return True


def round_div(a: int, b: int) -> int:
    """Nearest integer to ``a / b`` (ties round up), exact for big ints."""
    if b < 0:
        a, b = -a, -b
    return (2 * a + b) // (2 * b)


def int_sqrt_ceil(n: int) -> int:
    """Smallest integer ``r`` with ``r*r >= n``."""
    r = isqrt(n)
    return r if r * r == n else r + 1
