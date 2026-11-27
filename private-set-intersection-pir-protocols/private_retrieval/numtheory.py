"""Number-theoretic helpers: primality testing, safe primes and hashing into groups."""
from __future__ import annotations

import hashlib
import random

_SMALL_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71)


def is_probable_prime(n: int, rounds: int = 24, rng: random.Random | None = None) -> bool:
    """Miller-Rabin primality test (deterministic for ``n < 3.3e24`` via fixed bases)."""
    if n < 2:
        return False
    for p in _SMALL_PRIMES:
        if n == p:
            return True
        if n % p == 0:
            return False
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    bases = list(_SMALL_PRIMES[:13])
    if n.bit_length() > 80:
        r = rng or random.Random(n & 0xFFFFFFFF)
        bases += [r.randrange(2, n - 1) for _ in range(rounds)]
    for a in bases:
        a %= n
        if a < 2:
            continue
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


def random_prime(bits: int, rng: random.Random) -> int:
    """Return a random prime with exactly ``bits`` bits."""
    if bits < 3:
        raise ValueError("bits must be >= 3")
    while True:
        c = rng.getrandbits(bits) | (1 << (bits - 1)) | 1
        if is_probable_prime(c, rng=rng):
            return c


def random_safe_prime(bits: int, rng: random.Random) -> int:
    """Return a safe prime ``p = 2q + 1`` (``q`` prime) with ``bits`` bits."""
    while True:
        q = random_prime(bits - 1, rng)
        p = 2 * q + 1
        if p.bit_length() == bits and is_probable_prime(p, rng=rng):
            return p


def modinv(a: int, m: int) -> int:
    """Modular inverse of ``a`` modulo ``m``."""
    return pow(a, -1, m)


def hash_to_int(data: bytes, bits: int, domain: bytes = b"") -> int:
    """Expand SHA-256 in counter mode to an integer with at least ``bits`` bits."""
    out = b""
    ctr = 0
    while len(out) * 8 < bits + 64:
        out += hashlib.sha256(domain + ctr.to_bytes(4, "big") + data).digest()
        ctr += 1
    return int.from_bytes(out, "big")


class SchnorrGroup:
    """Quadratic-residue subgroup of ``Z_p^*`` for a safe prime ``p = 2q + 1``.

    The subgroup has prime order ``q``; DDH is believed hard in it, which is
    what Diffie-Hellman based PSI relies on.
    """

    def __init__(self, p: int) -> None:
        q = (p - 1) // 2
        if p % 2 == 0 or 2 * q + 1 != p or not is_probable_prime(p) or not is_probable_prime(q):
            raise ValueError("p must be a safe prime")
        self.p, self.q = p, q

    @classmethod
    def generate(cls, bits: int, seed: int = 0) -> "SchnorrGroup":
        """Generate a group with a ``bits``-bit safe prime from a seeded RNG."""
        return cls(random_safe_prime(bits, random.Random(seed)))

    def hash_to_group(self, item: bytes) -> int:
        """Hash ``item`` to a subgroup element by squaring a hashed residue."""
        while True:
            h = hash_to_int(item, self.p.bit_length(), b"h2g") % self.p
            e = h * h % self.p
            if e not in (0, 1):
                return e
            item += b"\x00"

    def random_exponent(self, rng: random.Random) -> int:
        """Uniform exponent in ``[1, q-1]``."""
        return rng.randrange(1, self.q)

    def exp(self, base: int, e: int) -> int:
        """``base ** e mod p``."""
        return pow(base, e, self.p)

    def contains(self, x: int) -> bool:
        """Membership test: ``x`` is a quadratic residue."""
        return 0 < x < self.p and pow(x, self.q, self.p) == 1
