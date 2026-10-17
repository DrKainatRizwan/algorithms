"""Shamir's (t, n) threshold secret sharing over a prime field."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from .field import (PRIME_127, interpolate_at, interpolate_poly, poly_eval,
                    random_poly, inv_mod)


@dataclass(frozen=True)
class Share:
    """A point (x, y) on the sharing polynomial."""

    x: int
    y: int
    prime: int = PRIME_127

    def __add__(self, other: "Share") -> "Share":
        self._check(other)
        return Share(self.x, (self.y + other.y) % self.prime, self.prime)

    def __sub__(self, other: "Share") -> "Share":
        self._check(other)
        return Share(self.x, (self.y - other.y) % self.prime, self.prime)

    def scale(self, k: int) -> "Share":
        """Multiply the shared secret by a public constant."""
        return Share(self.x, self.y * k % self.prime, self.prime)

    def add_const(self, k: int) -> "Share":
        """Add a public constant to the shared secret."""
        return Share(self.x, (self.y + k) % self.prime, self.prime)

    def _check(self, other: "Share") -> None:
        if self.x != other.x or self.prime != other.prime:
            raise ValueError("shares must have the same x and field to be combined")


def make_shares(secret: int, threshold: int, n: int, prime: int = PRIME_127,
                rng: random.Random | None = None) -> list[Share]:
    """Split ``secret`` into ``n`` shares; any ``threshold`` of them recover it."""
    if not 1 <= threshold <= n:
        raise ValueError("need 1 <= threshold <= n")
    if n >= prime:
        raise ValueError("field too small for n shares")
    if not 0 <= secret < prime:
        raise ValueError("secret must lie in [0, prime)")
    rng = rng or random.SystemRandom()
    coeffs = random_poly(secret, threshold - 1, prime, rng)
    return [Share(x, poly_eval(coeffs, x, prime), prime) for x in range(1, n + 1)]


def reconstruct(shares: Sequence[Share]) -> int:
    """Recover the secret from at least ``threshold`` shares (Lagrange at 0)."""
    if not shares:
        raise ValueError("no shares supplied")
    p = shares[0].prime
    return interpolate_at([s.x for s in shares], [s.y for s in shares], p, 0)


def recover_polynomial(shares: Sequence[Share]) -> list[int]:
    """Coefficients of the sharing polynomial (needs >= threshold shares)."""
    p = shares[0].prime
    return interpolate_poly([s.x for s in shares], [s.y for s in shares], p)


def is_consistent(shares: Sequence[Share], threshold: int) -> bool:
    """True if all shares lie on one polynomial of degree < ``threshold``."""
    if len(shares) <= threshold:
        return True
    p = shares[0].prime
    base = shares[:threshold]
    xs = [s.x for s in base]
    ys = [s.y for s in base]
    return all(interpolate_at(xs, ys, p, s.x) == s.y for s in shares[threshold:])


def consistent_with_secret(shares: Sequence[Share], candidate: int) -> bool:
    """Perfect-secrecy witness: do ``shares`` (fewer than t) admit ``candidate``?

    With fewer than ``threshold`` shares plus the point (0, candidate), the
    interpolation always exists, so every candidate secret is equally likely.
    """
    p = shares[0].prime
    xs = [0] + [s.x for s in shares]
    ys = [candidate % p] + [s.y for s in shares]
    return len(set(xs)) == len(xs) and len(interpolate_poly(xs, ys, p)) == len(xs)


def add_shares(a: Sequence[Share], b: Sequence[Share]) -> list[Share]:
    """Share-wise sum: shares of (secret_a + secret_b)."""
    return [x + y for x, y in zip(a, b)]


def refresh_shares(shares: Sequence[Share], threshold: int,
                   rng: random.Random | None = None) -> list[Share]:
    """Proactive refresh: add a random zero-secret sharing to every share.

    Old shares become useless in combination with new ones, but the secret
    is unchanged.
    """
    p = shares[0].prime
    rng = rng or random.SystemRandom()
    coeffs = random_poly(0, threshold - 1, p, rng)
    return [Share(s.x, (s.y + poly_eval(coeffs, s.x, p)) % p, p) for s in shares]


def share_bytes(data: bytes, threshold: int, n: int, prime: int = PRIME_127,
                rng: random.Random | None = None) -> list[list[Share]]:
    """Share arbitrary bytes by chunking into field elements (length prefixed).

    Returns one list of shares per participant-index: result[i] holds the
    chunk-shares of participant i+1.
    """
    chunk = (prime.bit_length() - 1) // 8
    framed = len(data).to_bytes(4, "big") + data
    framed += b"\x00" * (-len(framed) % chunk)
    per_party: list[list[Share]] = [[] for _ in range(n)]
    for off in range(0, len(framed), chunk):
        value = int.from_bytes(framed[off:off + chunk], "big")
        for i, sh in enumerate(make_shares(value, threshold, n, prime, rng)):
            per_party[i].append(sh)
    return per_party


def reconstruct_bytes(parties: Sequence[Sequence[Share]]) -> bytes:
    """Inverse of :func:`share_bytes` given the share-lists of >= t parties."""
    p = parties[0][0].prime
    chunk = (p.bit_length() - 1) // 8
    out = b""
    for c in range(len(parties[0])):
        value = reconstruct([party[c] for party in parties])
        out += value.to_bytes(chunk, "big")
    length = int.from_bytes(out[:4], "big")
    return out[4:4 + length]


def reconstruct_with_weights(shares: Sequence[Share], weights: Sequence[int]) -> int:
    """Reconstruct using precomputed Lagrange weights (used by MPC)."""
    p = shares[0].prime
    return sum(w * s.y for w, s in zip(weights, shares)) % p


__all__ = ["Share", "make_shares", "reconstruct", "recover_polynomial",
           "is_consistent", "consistent_with_secret", "add_shares",
           "refresh_shares", "share_bytes", "reconstruct_bytes",
           "reconstruct_with_weights", "inv_mod"]
