"""Integral LLL reduction (Cohen, Algorithm 2.6.7).

All arithmetic is over Python integers: the Gram-Schmidt data is kept as
``d_i`` (leading Gram determinants) and ``lambda_ij = d_j * mu_ij``, which
are integers for an integer basis.  No floating-point error can occur.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Sequence

from .linalg import Matrix, copy_basis, dot, round_div


def lll_reduce(basis: Sequence[Sequence[int]], delta: Fraction = Fraction(3, 4),
               stats: dict | None = None) -> Matrix:
    """Return an LLL-reduced basis of the lattice spanned by ``basis``.

    ``delta`` must lie in ``(1/4, 1]``; values close to 1 give shorter
    vectors at the cost of more swaps.  ``stats`` (optional) receives the
    number of swaps and size-reduction steps performed.
    """
    delta = Fraction(delta)
    if not (Fraction(1, 4) < delta <= 1):
        raise ValueError("delta must be in (1/4, 1]")
    n = len(basis)
    if n == 0:
        return []
    p, q = delta.numerator, delta.denominator
    b: list[list[int]] = [[]] + copy_basis(basis)  # 1-indexed
    lam = [[0] * (n + 1) for _ in range(n + 1)]
    d = [0] * (n + 1)
    d[0] = 1
    d[1] = dot(b[1], b[1])
    if d[1] == 0:
        raise ValueError("basis vectors are linearly dependent")
    swaps = 0
    reds = 0

    def red(k: int, l: int) -> None:
        nonlocal reds
        if 2 * abs(lam[k][l]) > d[l]:
            qq = round_div(lam[k][l], d[l])
            b[k] = [x - qq * y for x, y in zip(b[k], b[l])]
            lam[k][l] -= qq * d[l]
            for i in range(1, l):
                lam[k][i] -= qq * lam[l][i]
            reds += 1

    def swap(k: int, kmax: int) -> None:
        b[k], b[k - 1] = b[k - 1], b[k]
        for j in range(1, k - 1):
            lam[k][j], lam[k - 1][j] = lam[k - 1][j], lam[k][j]
        l = lam[k][k - 1]
        big_b = (d[k - 2] * d[k] + l * l) // d[k - 1]
        for i in range(k + 1, kmax + 1):
            t = lam[i][k]
            lam[i][k] = (d[k] * lam[i][k - 1] - l * t) // d[k - 1]
            lam[i][k - 1] = (big_b * t + l * lam[i][k]) // d[k]
        d[k - 1] = big_b

    k, kmax = 2, 1
    while k <= n:
        if k > kmax:
            kmax = k
            for j in range(1, k + 1):
                u = dot(b[k], b[j])
                for i in range(1, j):
                    u = (d[i] * u - lam[k][i] * lam[j][i]) // d[i - 1]
                if j < k:
                    lam[k][j] = u
                else:
                    if u <= 0:
                        raise ValueError("basis vectors are linearly dependent")
                    d[k] = u
        red(k, k - 1)
        # Lovasz condition:  d_k d_{k-2} + lambda^2 >= delta d_{k-1}^2
        if q * (d[k] * d[k - 2] + lam[k][k - 1] ** 2) < p * d[k - 1] ** 2:
            swap(k, kmax)
            swaps += 1
            k = max(2, k - 1)
        else:
            for l in range(k - 2, 0, -1):
                red(k, l)
            k += 1
    if stats is not None:
        stats["swaps"] = swaps
        stats["reductions"] = reds
    return b[1:]
