"""Linear algebra over GF(2) using Python integers as bit-vectors.

Row ``i`` of a matrix is an ``int`` whose bit ``j`` is the coefficient of
unknown ``j``.  XOR of two rows is addition of the equations, which makes
Gaussian elimination very compact and fast for the system sizes used here.
"""
from __future__ import annotations

from typing import Sequence


def parity(x: int) -> int:
    """Return the parity (XOR of all bits) of a non-negative integer."""
    return x.bit_count() & 1


def solve(rows: Sequence[int], rhs: Sequence[int], ncols: int):
    """Solve ``A x = b`` over GF(2).

    Parameters
    ----------
    rows: equation coefficient masks (bit ``j`` = coefficient of ``x_j``).
    rhs: right-hand side bits, one per row.
    ncols: number of unknowns.

    Returns
    -------
    ``None`` if the system is inconsistent, otherwise ``(x, null_basis)`` where
    ``x`` is a particular solution (free variables set to 0) packed into an
    int and ``null_basis`` is a list of ints spanning the homogeneous solutions.
    """
    if len(rows) != len(rhs):
        raise ValueError("rows and rhs must have equal length")
    # augmented rows: bit ncols holds the right-hand side
    aug = [r | ((b & 1) << ncols) for r, b in zip(rows, rhs)]
    pivots: list[int] = []  # pivot column per reduced row
    rank = 0
    for col in range(ncols):
        bit = 1 << col
        sel = None
        for i in range(rank, len(aug)):
            if aug[i] & bit:
                sel = i
                break
        if sel is None:
            continue
        aug[rank], aug[sel] = aug[sel], aug[rank]
        prow = aug[rank]
        for i in range(len(aug)):
            if i != rank and aug[i] & bit:
                aug[i] ^= prow
        pivots.append(col)
        rank += 1
    for i in range(rank, len(aug)):
        if aug[i] >> ncols & 1:  # 0 = 1 : inconsistent
            return None
    sol = 0
    for i, col in enumerate(pivots):
        if aug[i] >> ncols & 1:
            sol |= 1 << col
    pivset = set(pivots)
    free = [c for c in range(ncols) if c not in pivset]
    basis: list[int] = []
    for f in free:
        v = 1 << f
        for i, col in enumerate(pivots):
            if aug[i] >> f & 1:
                v |= 1 << col
        basis.append(v)
    return sol, basis


def rank(rows: Sequence[int], ncols: int) -> int:
    """Rank of a GF(2) matrix given as row bitmasks."""
    res = solve(rows, [0] * len(rows), ncols)
    assert res is not None
    return ncols - len(res[1])
