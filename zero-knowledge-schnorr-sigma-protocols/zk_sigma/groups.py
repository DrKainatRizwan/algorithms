"""Prime-order subgroups of Z_p^* (Schnorr groups) used by all protocols."""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

_MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def is_probable_prime(n: int) -> bool:
    """Miller-Rabin primality test (deterministic below ~3.3e24)."""
    if n < 2:
        return False
    for sp in _MR_BASES:
        if n % sp == 0:
            return n == sp
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in _MR_BASES:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


@dataclass(frozen=True)
class SchnorrGroup:
    """Subgroup of order ``q`` in Z_p^*, with ``p = k*q + 1``.

    ``g`` generates the subgroup; ``h`` is a second generator whose discrete
    log with respect to ``g`` is unknown (derived by hashing), as required by
    Pedersen commitments.
    """

    p: int
    q: int
    g: int
    h: int

    @staticmethod
    def build(p: int, q: int, g: int | None = None) -> "SchnorrGroup":
        if (p - 1) % q:
            raise ValueError("q must divide p-1")
        cof = (p - 1) // q
        if g is None:
            g = SchnorrGroup._hash_to(p, cof, b"generator-g")
        h = SchnorrGroup._hash_to(p, cof, b"generator-h", g)
        grp = SchnorrGroup(p, q, g, h)
        if not (grp.is_member(g) and g != 1 and grp.is_member(h) and h != 1):
            raise ValueError("invalid generators")
        return grp

    @staticmethod
    def _hash_to(p: int, cof: int, label: bytes, avoid: int = 0) -> int:
        ctr = 0
        while True:
            dig = hashlib.sha256(label + ctr.to_bytes(4, "big")).digest()
            x = pow(int.from_bytes(dig, "big") % p, cof, p)
            if x not in (0, 1, avoid):
                return x
            ctr += 1

    def is_member(self, x: int) -> bool:
        return 0 < x < self.p and pow(x, self.q, self.p) == 1

    def mul(self, a: int, b: int) -> int:
        return a * b % self.p

    def inv(self, a: int) -> int:
        return pow(a, -1, self.p)

    def exp(self, base: int, e: int) -> int:
        return pow(base, e % self.q, self.p)

    def multiexp(self, bases: list[int], exps: list[int]) -> int:
        acc = 1
        for b, e in zip(bases, exps):
            acc = acc * pow(b, e % self.q, self.p) % self.p
        return acc

    def random_scalar(self, rng: random.Random) -> int:
        return rng.randrange(1, self.q)

    def random_nonce(self, rng: random.Random) -> int:
        """Uniform in Z_q (zero allowed) so simulated and real runs match exactly."""
        return rng.randrange(self.q)

    def random_element(self, rng: random.Random) -> int:
        return self.exp(self.g, self.random_scalar(rng))


# Safe primes p = 2q+1 found by seeded search.
_Q64 = 4995170651525545391
_Q128 = 126516394030417222297910077996473308003


def standard_group(bits: int = 128) -> SchnorrGroup:
    """Return a built-in safe-prime group with 64- or 128-bit modulus."""
    table = {64: _Q64, 128: _Q128}
    if bits not in table:
        raise ValueError("bits must be 64 or 128")
    q = table[bits]
    return SchnorrGroup.build(2 * q + 1, q)


def generate_group(bits: int, seed: int = 0) -> SchnorrGroup:
    """Search for a fresh safe-prime group of about ``bits`` bits."""
    rng = random.Random(seed)
    while True:
        q = rng.getrandbits(bits - 1) | (1 << (bits - 2)) | 1
        if is_probable_prime(q) and is_probable_prime(2 * q + 1):
            return SchnorrGroup.build(2 * q + 1, q)


def toy_group() -> SchnorrGroup:
    """Tiny group (p=23, q=11) for exhaustive distribution checks."""
    return SchnorrGroup.build(23, 11, g=4)
