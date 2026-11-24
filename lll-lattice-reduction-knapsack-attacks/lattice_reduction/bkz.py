"""Block Korkine-Zolotarev (BKZ) reduction with exact-enumeration SVP oracle."""
from __future__ import annotations

from fractions import Fraction
from typing import Sequence

from .enumeration import enumerate_coefficients
from .linalg import Matrix, gram_schmidt_float
from .lll import lll_reduce


def _insert_vector(rows: list[list[int]], coeffs: list[int]) -> bool:
    """Unimodularly change ``rows`` so that its first row is ``sum c_i rows_i``.

    Uses a Euclid-style elimination on the coefficient vector; each step is
    an elementary row operation, so the lattice is unchanged.  Returns False
    if the coefficients were not primitive.
    """
    c = list(coeffs)
    while True:
        nz = [i for i, v in enumerate(c) if v != 0]
        if len(nz) == 1:
            break
        j = min(nz, key=lambda i: abs(c[i]))
        for i in nz:
            if i == j:
                continue
            q = c[i] // c[j]
            rows[j] = [a + q * b for a, b in zip(rows[j], rows[i])]
            c[i] -= q * c[j]
    m = nz[0]
    if abs(c[m]) != 1:
        return False
    if c[m] == -1:
        rows[m] = [-a for a in rows[m]]
    rows.insert(0, rows.pop(m))
    return True


def bkz_reduce(basis: Sequence[Sequence[int]], block_size: int,
               delta: Fraction = Fraction(99, 100), max_tours: int = 20,
               stats: dict | None = None) -> Matrix:
    """BKZ-``block_size`` reduction.

    Each tour slides a window of ``block_size`` vectors along the basis;
    in each window the shortest vector of the projected sublattice is found
    by enumeration and inserted if it improves ``|b_k*|``.  Stops when a full
    tour makes no change (or after ``max_tours``).  ``block_size = 2``
    coincides with LLL; ``block_size = n`` gives a HKZ-like first vector.
    """
    n = len(basis)
    if block_size < 2:
        raise ValueError("block_size must be at least 2")
    b = lll_reduce(basis, delta)
    tours = 0
    inserts = 0
    for _ in range(max_tours):
        tours += 1
        changed = False
        for k in range(n - 1):
            h = min(k + block_size, n)
            mu, B = gram_schmidt_float(b)
            sub_mu = [row[k:h] for row in mu[k:h]]
            sub_B = B[k:h]
            res = enumerate_coefficients(sub_mu, sub_B, [0.0] * (h - k),
                                         sub_B[0] * 0.999, exclude_zero=True)
            if res is None:
                continue
            rows = [r[:] for r in b[k:h]]
            if not _insert_vector(rows, res[0]):
                continue
            b = b[:k] + rows + b[h:]
            b = lll_reduce(b, delta)
            changed = True
            inserts += 1
        if not changed:
            break
    if stats is not None:
        stats["tours"] = tours
        stats["insertions"] = inserts
    return b
