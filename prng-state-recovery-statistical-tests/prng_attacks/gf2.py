"""Gaussian elimination over GF(2) with rows stored as Python integers."""

from __future__ import annotations


def solve_gf2(rows: list[int], rhs: list[int], nvars: int) -> list[int] | None:
    """Solve ``rows[i] . x = rhs[i]`` over GF(2).

    Each row is a bitmask over ``nvars`` variables (bit j = coefficient of x_j).
    Returns one solution (free variables set to 0) or ``None`` if inconsistent.
    """
    pivots: dict[int, tuple[int, int]] = {}   # pivot bit -> (row, rhs)
    for r, b in zip(rows, rhs):
        b &= 1
        while r:
            p = r.bit_length() - 1
            if p in pivots:
                pr, pb = pivots[p]
                r ^= pr
                b ^= pb
            else:
                pivots[p] = (r, b)
                break
        else:
            if b:
                return None
    x = [0] * nvars
    for p in sorted(pivots):                  # ascending: lower pivots first
        r, b = pivots[p]
        rest = r & ~(1 << p)
        val = b
        while rest:
            low = rest & -rest
            val ^= x[low.bit_length() - 1]
            rest ^= low
        x[p] = val
    return x


def gf2_rank(rows: list[int]) -> int:
    """Rank of a GF(2) matrix given as row bitmasks."""
    basis: dict[int, int] = {}
    for r in rows:
        while r:
            p = r.bit_length() - 1
            if p in basis:
                r ^= basis[p]
            else:
                basis[p] = r
                break
    return len(basis)
