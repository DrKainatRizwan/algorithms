"""Secure multi-party computation: sum protocols and BGW arithmetic."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Sequence

from .field import PRIME_127, lagrange_coefficients
from .shamir import Share, make_shares, reconstruct


def additive_shares(secret: int, n: int, prime: int, rng: random.Random) -> list[int]:
    """n-of-n additive sharing: shares sum to ``secret`` mod prime."""
    parts = [rng.randrange(prime) for _ in range(n - 1)]
    parts.append((secret - sum(parts)) % prime)
    return parts


def additive_secure_sum(inputs: Sequence[int], prime: int = PRIME_127,
                        seed: int = 0) -> int:
    """Secure sum where every party sends one additive share to every other."""
    rng = random.Random(seed)
    n = len(inputs)
    dealt = [additive_shares(v, n, prime, rng) for v in inputs]
    partial = [sum(dealt[i][j] for i in range(n)) % prime for j in range(n)]
    return sum(partial) % prime


@dataclass
class SumResult:
    """Outcome of a Shamir-based secure sum."""

    total: int
    contributors: list[int]
    messages: int


def shamir_secure_sum(inputs: Sequence[int], threshold: int, prime: int = PRIME_127,
                      dropped_before: Sequence[int] = (),
                      dropped_after: Sequence[int] = (),
                      seed: int = 0) -> SumResult:
    """Dropout-tolerant secure sum with a (threshold, n) Shamir scheme.

    ``dropped_before`` parties never distribute shares (their input is
    excluded). ``dropped_after`` parties distribute shares but do not
    publish their partial sums, so the total only needs ``threshold``
    surviving evaluators.
    """
    rng = random.Random(seed)
    n = len(inputs)
    contributors = [i for i in range(n) if i not in set(dropped_before)]
    dealt = {i: make_shares(inputs[i], threshold, n, prime, rng) for i in contributors}
    messages = len(contributors) * (n - 1)
    evaluators = [j for j in range(n)
                  if j not in set(dropped_before) | set(dropped_after)]
    if len(evaluators) < threshold:
        raise ValueError("too few surviving parties to reconstruct the sum")
    partial = []
    for j in evaluators:
        y = sum(dealt[i][j].y for i in contributors) % prime
        partial.append(Share(j + 1, y, prime))
    messages += len(evaluators) * (len(evaluators) - 1)
    total = reconstruct(partial[:threshold])
    return SumResult(total, contributors, messages)


@dataclass
class Network:
    """Counts messages and field elements exchanged during a protocol."""

    messages: int = 0
    rounds: int = 0

    def send(self, count: int = 1) -> None:
        self.messages += count


@dataclass
class BGW:
    """Semi-honest BGW engine over GF(prime) with n parties, threshold t < n/2.

    Values are held as lists of shares (one per party). Addition and public
    scaling are local; multiplication needs one round of degree reduction.
    """

    n: int
    t: int
    prime: int = PRIME_127
    seed: int = 0
    net: Network = field(default_factory=Network)

    def __post_init__(self) -> None:
        if 2 * self.t + 1 > self.n:
            raise ValueError("BGW multiplication needs n >= 2t + 1")
        self.rng = random.Random(self.seed)
        self._recomb = lagrange_coefficients(list(range(1, self.n + 1)), self.prime)

    def input(self, value: int) -> list[Share]:
        """A party secret-shares its input among all parties."""
        self.net.send(self.n - 1)
        return make_shares(value, self.t + 1, self.n, self.prime, self.rng)

    def add(self, a: Sequence[Share], b: Sequence[Share]) -> list[Share]:
        return [x + y for x, y in zip(a, b)]

    def sub(self, a: Sequence[Share], b: Sequence[Share]) -> list[Share]:
        return [x - y for x, y in zip(a, b)]

    def scale(self, a: Sequence[Share], k: int) -> list[Share]:
        return [s.scale(k) for s in a]

    def add_const(self, a: Sequence[Share], k: int) -> list[Share]:
        return [s.add_const(k) for s in a]

    def mul(self, a: Sequence[Share], b: Sequence[Share]) -> list[Share]:
        """Multiply shared values (degree reduction via resharing)."""
        p = self.prime
        local = [(x.y * y.y) % p for x, y in zip(a, b)]  # degree 2t shares
        # Each party reshares its product with a fresh degree-t polynomial.
        resh = [make_shares(v, self.t + 1, self.n, p, self.rng) for v in local]
        self.net.send(self.n * (self.n - 1))
        self.net.rounds += 1
        out = []
        for j in range(self.n):
            y = sum(self._recomb[i] * resh[i][j].y for i in range(self.n)) % p
            out.append(Share(j + 1, y, p))
        return out

    def open(self, a: Sequence[Share]) -> int:
        """Publicly reconstruct a shared value."""
        self.net.send(self.n * (self.n - 1))
        return reconstruct(list(a)[: self.t + 1])

    def dot(self, xs: Sequence[Sequence[Share]], ys: Sequence[Sequence[Share]]) -> list[Share]:
        """Inner product of two vectors of shared values."""
        acc = self.mul(xs[0], ys[0])
        for x, y in zip(xs[1:], ys[1:]):
            acc = self.add(acc, self.mul(x, y))
        return acc


def secure_mean_variance(values: Sequence[int], t: int, prime: int = PRIME_127,
                         seed: int = 0) -> tuple[float, float, Network]:
    """Compute mean and variance of private inputs, one value per party.

    Opens only sum and sum of squares, never any individual value.
    """
    n = len(values)
    eng = BGW(n=n, t=t, prime=prime, seed=seed)
    shared = [eng.input(v) for v in values]
    s1 = shared[0]
    for sh in shared[1:]:
        s1 = eng.add(s1, sh)
    s2 = eng.dot(shared, shared)
    total, total_sq = eng.open(s1), eng.open(s2)
    mean = total / n
    return mean, total_sq / n - mean * mean, eng.net


def secure_equality_test(a: int, b: int, t: int = 1, n: int = 3,
                              prime: int = PRIME_127, seed: int = 0) -> bool:
    """Private equality test via Fermat: (a-b)^(p-1) is 0 iff a == b.

    The opened value is 0 or 1 only, revealing nothing else about a - b.
    """
    eng = BGW(n=n, t=t, prime=prime, seed=seed)
    d = eng.sub(eng.input(a), eng.input(b))
    result = d
    exp = prime - 1
    base = d
    result = None
    while exp:
        if exp & 1:
            result = base if result is None else eng.mul(result, base)
        exp >>= 1
        if exp:
            base = eng.mul(base, base)
    return eng.open(result) == 0
