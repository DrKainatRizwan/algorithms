"""Error-correcting reconstruction (Berlekamp-Welch) for corrupted shares."""
from __future__ import annotations

from typing import Sequence

from .field import inv_mod, poly_divmod
from .shamir import Share


def _solve_linear(matrix: list[list[int]], rhs: list[int], p: int) -> list[int] | None:
    """Gaussian elimination mod p; free variables are set to zero."""
    rows, cols = len(matrix), len(matrix[0])
    aug = [row[:] + [r] for row, r in zip(matrix, rhs)]
    pivots: list[int] = []
    r = 0
    for c in range(cols):
        piv = next((i for i in range(r, rows) if aug[i][c] % p), None)
        if piv is None:
            continue
        aug[r], aug[piv] = aug[piv], aug[r]
        inv = inv_mod(aug[r][c], p)
        aug[r] = [v * inv % p for v in aug[r]]
        for i in range(rows):
            if i != r and aug[i][c]:
                f = aug[i][c]
                aug[i] = [(a - f * b) % p for a, b in zip(aug[i], aug[r])]
        pivots.append(c)
        r += 1
        if r == rows:
            break
    for i in range(r, rows):
        if aug[i][cols] % p:
            return None
    sol = [0] * cols
    for i, c in enumerate(pivots):
        sol[c] = aug[i][cols]
    return sol


def berlekamp_welch(shares: Sequence[Share], threshold: int,
                    max_errors: int | None = None) -> tuple[list[int], list[int]] | None:
    """Recover the degree < threshold polynomial despite corrupted shares.

    Requires ``len(shares) >= threshold + 2e`` for ``e`` errors. Returns
    ``(coefficients, error_xs)`` or ``None`` if decoding fails.
    """
    p = shares[0].prime
    n, k = len(shares), threshold
    emax = (n - k) // 2 if max_errors is None else min(max_errors, (n - k) // 2)
    for e in range(emax, -1, -1):
        # Unknowns: Q (e + k coefficients) and E's low e coefficients (E monic).
        matrix, rhs = [], []
        for s in shares:
            row = [pow(s.x, j, p) for j in range(e + k)]
            row += [(-s.y * pow(s.x, j, p)) % p for j in range(e)]
            matrix.append(row)
            rhs.append(s.y * pow(s.x, e, p) % p)
        if not matrix[0]:
            continue
        sol = _solve_linear(matrix, rhs, p)
        if sol is None:
            continue
        q_poly = sol[: e + k]
        e_poly = sol[e + k:] + [1]
        quot, rem = poly_divmod(q_poly, e_poly, p)
        if any(rem):
            continue
        quot = (quot + [0] * k)[:k]
        bad = [s.x for s in shares
               if sum(c * pow(s.x, j, p) for j, c in enumerate(quot)) % p != s.y]
        if len(bad) <= e:
            return quot, bad
    return None


def robust_reconstruct(shares: Sequence[Share], threshold: int) -> tuple[int, list[int]]:
    """Return (secret, corrupted_xs) or raise ``ValueError`` if undecodable."""
    res = berlekamp_welch(shares, threshold)
    if res is None:
        raise ValueError("too many corrupted shares to decode")
    coeffs, bad = res
    return coeffs[0], bad
