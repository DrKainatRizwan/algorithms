"""Schnorr-Euchner enumeration: exact SVP and CVP, plus Babai's algorithm."""
from __future__ import annotations

from fractions import Fraction
from typing import Sequence

from .linalg import Matrix, dot, gram_schmidt, gram_schmidt_float, norm_sq
from .lll import lll_reduce


def enumerate_coefficients(mu: Sequence[Sequence[float]], B: Sequence[float],
                           tau: Sequence[float], radius_sq: float,
                           exclude_zero: bool = False) -> tuple[list[int], float] | None:
    """Find integer ``x`` minimising ``sum_i B_i (x_i + sum_{j>i} x_j mu_ji - tau_i)^2``.

    Only solutions with value strictly below ``radius_sq`` are returned;
    ``None`` means there is no such point.  With ``exclude_zero`` the
    all-zero coefficient vector is skipped (needed for SVP).
    """
    n = len(B)
    x = [0] * n
    best: list = [None, radius_sq]

    def rec(i: int, dist: float) -> None:
        c = tau[i] - sum(x[j] * mu[j][i] for j in range(i + 1, n))
        r = round(c)
        sgn = 1 if c >= r else -1
        k = 0
        while True:
            if k == 0:
                cand = r
            else:
                cand = r + sgn * ((k + 1) // 2) * (1 if k % 2 == 1 else -1)
            nd = dist + (cand - c) ** 2 * B[i]
            if nd >= best[1]:
                break
            x[i] = cand
            if i == 0:
                if not (exclude_zero and not any(x)):
                    best[0] = x.copy()
                    best[1] = nd
            else:
                rec(i - 1, nd)
            k += 1
        x[i] = 0

    rec(n - 1, 0.0)
    if best[0] is None:
        return None
    return best[0], best[1]


def combine(basis: Sequence[Sequence[int]], coeffs: Sequence[int]) -> list[int]:
    """Integer combination ``sum_i coeffs[i] * basis[i]``."""
    dim = len(basis[0])
    return [sum(c * row[t] for c, row in zip(coeffs, basis)) for t in range(dim)]


def svp(basis: Sequence[Sequence[int]], delta: Fraction = Fraction(99, 100)) -> list[int]:
    """Shortest non-zero lattice vector, found by LLL + exhaustive enumeration."""
    red = lll_reduce(basis, delta)
    mu, B = gram_schmidt_float(red)
    bound = norm_sq(red[0]) * (1 + 1e-9) + 1e-9
    res = enumerate_coefficients(mu, B, [0.0] * len(B), bound, exclude_zero=True)
    if res is None:
        return list(red[0])
    v = combine(red, res[0])
    return v if norm_sq(v) < norm_sq(red[0]) else list(red[0])


def _target_coords(basis: Sequence[Sequence[int]], target: Sequence[int],
                   mu: Sequence[Sequence[Fraction]], B: Sequence[Fraction]) -> list[Fraction]:
    tau: list[Fraction] = []
    for i, row in enumerate(basis):
        t = Fraction(dot(target, row))
        for j in range(i):
            t -= mu[i][j] * tau[j] * B[j]
        tau.append(t / B[i])
    return tau


def babai_nearest_plane(basis: Sequence[Sequence[int]], target: Sequence[int]) -> list[int]:
    """Babai's nearest-plane algorithm; ``basis`` should be reduced.

    Returns the lattice vector (not the coefficients).  The error is at
    most ``sqrt(sum B_i)/2`` for any target.
    """
    mu, B = gram_schmidt(basis)
    tau = _target_coords(basis, target, mu, B)
    n = len(basis)
    x = [0] * n
    for i in range(n - 1, -1, -1):
        c = tau[i] - sum(x[j] * mu[j][i] for j in range(i + 1, n))
        x[i] = int((2 * c.numerator + c.denominator) // (2 * c.denominator))
    return combine(basis, x)


def cvp(basis: Sequence[Sequence[int]], target: Sequence[int],
        delta: Fraction = Fraction(99, 100)) -> list[int]:
    """Closest lattice vector to ``target`` (Babai start + enumeration)."""
    red = lll_reduce(basis, delta)
    v0 = babai_nearest_plane(red, target)
    mu_f, B_f = gram_schmidt(red)
    tau = [float(t) for t in _target_coords(red, target, mu_f, B_f)]
    mu = [[float(a) for a in row] for row in mu_f]
    B = [float(a) for a in B_f]
    d0 = sum((a - b) ** 2 for a, b in zip(v0, target))
    res = enumerate_coefficients(mu, B, tau, d0 * (1 - 1e-9) - 1e-9)
    if res is None:
        return v0
    v = combine(red, res[0])
    d1 = sum((a - b) ** 2 for a, b in zip(v, target))
    return v if d1 < d0 else v0


def dist_sq(u: Sequence[int], v: Sequence[int]) -> int:
    """Squared Euclidean distance between two integer vectors."""
    return sum((a - b) ** 2 for a, b in zip(u, v))
