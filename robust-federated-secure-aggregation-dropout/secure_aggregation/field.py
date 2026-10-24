"""Prime-field arithmetic and Shamir secret sharing.

All secrets handled by the protocol (Diffie-Hellman exponents and self-mask
seeds) are integers below the prime ``P = 2**255 - 19``.  A degree ``t - 1``
polynomial with the secret as constant term is evaluated at each recipient's
(non-zero) integer id; any ``t`` evaluations recover the secret by Lagrange
interpolation at zero.
"""

from __future__ import annotations

import random
from typing import Iterable, Sequence

P: int = 2**255 - 19

Share = tuple[int, int]


def inv_mod(a: int, p: int = P) -> int:
    """Multiplicative inverse of ``a`` modulo the prime ``p``."""
    a %= p
    if a == 0:
        raise ZeroDivisionError("0 has no inverse")
    return pow(a, -1, p)


def shamir_share(secret: int, ids: Sequence[int], threshold: int,
                 rng: random.Random, p: int = P) -> dict[int, int]:
    """Split ``secret`` into one share per id; any ``threshold`` of them suffice.

    Returns a mapping ``id -> f(id)`` where ``f`` is a random polynomial of
    degree ``threshold - 1`` with ``f(0) = secret``.
    """
    if not 1 <= threshold <= len(ids):
        raise ValueError("need 1 <= threshold <= number of recipients")
    if len(set(ids)) != len(ids) or any(i % p == 0 for i in ids):
        raise ValueError("ids must be distinct and non-zero")
    if not 0 <= secret < p:
        raise ValueError("secret out of field range")
    coeffs = [secret] + [rng.randrange(p) for _ in range(threshold - 1)]
    out: dict[int, int] = {}
    for x in ids:
        acc = 0
        for c in reversed(coeffs):  # Horner
            acc = (acc * x + c) % p
        out[x] = acc
    return out


def lagrange_at_zero(xs: Sequence[int], p: int = P) -> list[int]:
    """Lagrange basis coefficients ``l_i(0)`` for the nodes ``xs``."""
    coeffs = []
    for i, xi in enumerate(xs):
        num, den = 1, 1
        for j, xj in enumerate(xs):
            if i != j:
                num = num * (-xj) % p
                den = den * (xi - xj) % p
        coeffs.append(num * inv_mod(den, p) % p)
    return coeffs


def shamir_reconstruct(shares: Iterable[Share], p: int = P) -> int:
    """Recover the secret from ``(id, value)`` pairs (uses all pairs given)."""
    pts = list(shares)
    if not pts:
        raise ValueError("no shares supplied")
    xs = [x for x, _ in pts]
    if len(set(xs)) != len(xs):
        raise ValueError("duplicate share ids")
    lam = lagrange_at_zero(xs, p)
    return sum(l * y for l, (_, y) in zip(lam, pts)) % p
