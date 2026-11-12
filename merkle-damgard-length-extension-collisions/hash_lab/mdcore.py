"""Generic Merkle-Damgard construction.

A Merkle-Damgard hash keeps a fixed-size chaining state, absorbs the message
block by block through a compression function and finishes with
length-strengthening padding.  Because the digest *is* the final chaining state
(for the full-width variants), anyone holding a digest can resume hashing from
it.  This module exposes that capability explicitly via the ``state`` and
``count`` constructor arguments so the length-extension attack can be written
against a clean API.
"""
from __future__ import annotations

from typing import ClassVar, Sequence

__all__ = ["MDHash", "md_padding", "iroot", "first_primes", "rotl", "rotr"]


def rotl(x: int, n: int, bits: int = 32) -> int:
    """Rotate ``x`` left by ``n`` bits within a ``bits``-wide word."""
    mask = (1 << bits) - 1
    n %= bits
    return ((x << n) | (x >> (bits - n))) & mask


def rotr(x: int, n: int, bits: int = 32) -> int:
    """Rotate ``x`` right by ``n`` bits within a ``bits``-wide word."""
    return rotl(x, bits - (n % bits), bits)


def iroot(n: int, k: int) -> int:
    """Return floor(n ** (1/k)) using exact integer arithmetic (binary search)."""
    if n < 2:
        return n
    lo, hi = 1, 1 << ((n.bit_length() + k - 1) // k + 1)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if mid**k <= n:
            lo = mid
        else:
            hi = mid - 1
    return lo


def first_primes(count: int) -> list[int]:
    """Return the first ``count`` primes by trial division."""
    primes: list[int] = []
    cand = 2
    while len(primes) < count:
        if all(cand % p for p in primes if p * p <= cand):
            primes.append(cand)
        cand += 1
    return primes


def md_padding(length: int, block_size: int, length_size: int, byteorder: str) -> bytes:
    """Return the padding appended to a message of ``length`` bytes.

    The padding is a ``0x80`` byte, enough zeros to reach ``block_size -
    length_size`` modulo ``block_size`` and then the message length in bits.
    """
    zeros = (block_size - length_size - (length + 1)) % block_size
    bit_len = (length * 8) % (1 << (8 * length_size))
    return b"\x80" + b"\x00" * zeros + bit_len.to_bytes(length_size, byteorder)  # type: ignore[arg-type]


class MDHash:
    """Base class for Merkle-Damgard hash functions.

    Subclasses define the class attributes and implement :meth:`initial_state`,
    :meth:`compress`, :meth:`pack_state` and :meth:`unpack_state`.

    Parameters
    ----------
    data:
        Initial message bytes.
    state:
        Optional chaining state to resume from (used by length extension).
    count:
        Number of bytes already absorbed to reach ``state``.  Must be a
        multiple of the block size.
    """

    name: ClassVar[str] = "md"
    block_size: ClassVar[int] = 64
    digest_size: ClassVar[int] = 32
    length_size: ClassVar[int] = 8
    byteorder: ClassVar[str] = "big"

    def __init__(self, data: bytes = b"", *, state: Sequence[int] | None = None,
                 count: int = 0) -> None:
        if state is None:
            if count:
                raise ValueError("count requires an explicit state")
            state = self.initial_state()
        elif count % self.block_size:
            raise ValueError("count must be a multiple of the block size")
        self._state: tuple[int, ...] = tuple(state)
        self._count = count
        self._buf = b""
        if data:
            self.update(data)

    # -- interface for subclasses -------------------------------------------------
    @classmethod
    def initial_state(cls) -> tuple[int, ...]:
        raise NotImplementedError

    @classmethod
    def compress(cls, state: tuple[int, ...], block: bytes) -> tuple[int, ...]:
        raise NotImplementedError

    @classmethod
    def pack_state(cls, state: Sequence[int]) -> bytes:
        raise NotImplementedError

    @classmethod
    def unpack_state(cls, digest: bytes) -> tuple[int, ...]:
        raise NotImplementedError

    # -- public API ---------------------------------------------------------------
    def update(self, data: bytes) -> "MDHash":
        """Absorb ``data`` and return ``self`` for chaining."""
        self._count += len(data)
        buf = self._buf + data
        bs = self.block_size
        full = len(buf) - len(buf) % bs
        state = self._state
        for i in range(0, full, bs):
            state = self.compress(state, buf[i:i + bs])
        self._state = state
        self._buf = buf[full:]
        return self

    def copy(self) -> "MDHash":
        """Return an independent copy of the running hash."""
        other = type(self).__new__(type(self))
        other._state, other._count, other._buf = self._state, self._count, self._buf
        return other

    def digest(self) -> bytes:
        """Return the digest of everything absorbed so far."""
        pad = md_padding(self._count, self.block_size, self.length_size, self.byteorder)
        tail = self._buf + pad
        state = self._state
        for i in range(0, len(tail), self.block_size):
            state = self.compress(state, tail[i:i + self.block_size])
        return self.pack_state(state)

    def hexdigest(self) -> str:
        """Return the digest as lowercase hexadecimal."""
        return self.digest().hex()

    @classmethod
    def hash(cls, data: bytes) -> bytes:
        """One-shot convenience: digest of ``data``."""
        return cls(data).digest()
