"""Feldman verifiable secret sharing in a prime-order subgroup."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from .field import find_safe_prime, poly_eval, random_poly
from .shamir import Share


@dataclass(frozen=True)
class GroupParams:
    """Schnorr group: g generates the order-q subgroup of Z_p^*, p = 2q + 1."""

    p: int
    q: int
    g: int

    @staticmethod
    def generate(bits: int = 64, seed: int = 0) -> "GroupParams":
        rng = random.Random(seed)
        p, q = find_safe_prime(bits, rng)
        while True:
            h = rng.randrange(2, p - 1)
            g = pow(h, 2, p)  # squares generate the order-q subgroup
            if g != 1:
                return GroupParams(p, q, g)


@dataclass(frozen=True)
class FeldmanDealing:
    """Public commitments plus the private shares."""

    commitments: tuple[int, ...]
    shares: tuple[Share, ...]


def deal(secret: int, threshold: int, n: int, group: GroupParams,
         rng: random.Random) -> FeldmanDealing:
    """Share ``secret`` mod q and publish g^{a_j} commitments to coefficients."""
    coeffs = random_poly(secret, threshold - 1, group.q, rng)
    commitments = tuple(pow(group.g, c, group.p) for c in coeffs)
    shares = tuple(Share(x, poly_eval(coeffs, x, group.q), group.q)
                   for x in range(1, n + 1))
    return FeldmanDealing(commitments, shares)


def verify_share(share: Share, commitments: Sequence[int], group: GroupParams) -> bool:
    """Check g^y == prod_j C_j^{x^j} (mod p)."""
    lhs = pow(group.g, share.y, group.p)
    rhs = 1
    xp = 1
    for c in commitments:
        rhs = rhs * pow(c, xp, group.p) % group.p
        xp = xp * share.x % group.q
    return lhs == rhs


def combine_commitments(a: Sequence[int], b: Sequence[int], group: GroupParams) -> tuple[int, ...]:
    """Commitments to the sum of two sharing polynomials (homomorphic)."""
    return tuple(x * y % group.p for x, y in zip(a, b))


def find_cheaters(shares: Sequence[Share], commitments: Sequence[int],
                  group: GroupParams) -> list[int]:
    """Indices (x values) of shares that fail verification."""
    return [s.x for s in shares if not verify_share(s, commitments, group)]
