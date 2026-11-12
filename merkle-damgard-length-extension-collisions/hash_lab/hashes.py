"""MD5, SHA-1, SHA-256 and SHA-512 implemented from first principles.

Round constants for the SHA-2 family are *derived* (fractional parts of square
and cube roots of primes, computed with exact integer arithmetic) rather than
pasted, which doubles as a check on the specification.
"""
from __future__ import annotations

import math
import struct
from typing import Sequence

from .mdcore import MDHash, first_primes, iroot, rotl, rotr

__all__ = ["MD5", "SHA1", "SHA256", "SHA512", "ALGORITHMS"]


# ---------------------------------------------------------------------------- MD5
_MD5_SHIFTS = ([7, 12, 17, 22] * 4 + [5, 9, 14, 20] * 4 +
               [4, 11, 16, 23] * 4 + [6, 10, 15, 21] * 4)
_MD5_K = [int(abs(math.sin(i + 1)) * 2**32) & 0xFFFFFFFF for i in range(64)]


class MD5(MDHash):
    """MD5 (RFC 1321). Little-endian words and length field."""

    name = "md5"
    block_size = 64
    digest_size = 16
    length_size = 8
    byteorder = "little"

    @classmethod
    def initial_state(cls) -> tuple[int, ...]:
        return (0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476)

    @classmethod
    def compress(cls, state: tuple[int, ...], block: bytes) -> tuple[int, ...]:
        m = struct.unpack("<16I", block)
        a, b, c, d = state
        mask = 0xFFFFFFFF
        for i in range(64):
            if i < 16:
                f, g = (b & c) | (~b & d), i
            elif i < 32:
                f, g = (d & b) | (~d & c), (5 * i + 1) % 16
            elif i < 48:
                f, g = b ^ c ^ d, (3 * i + 5) % 16
            else:
                f, g = c ^ (b | (~d & mask)), (7 * i) % 16
            f = (f + a + _MD5_K[i] + m[g]) & mask
            a, d, c = d, c, b
            b = (b + rotl(f, _MD5_SHIFTS[i])) & mask
        return tuple((x + y) & mask for x, y in zip(state, (a, b, c, d)))

    @classmethod
    def pack_state(cls, state: Sequence[int]) -> bytes:
        return struct.pack("<4I", *state)

    @classmethod
    def unpack_state(cls, digest: bytes) -> tuple[int, ...]:
        return struct.unpack("<4I", digest)


# --------------------------------------------------------------------------- SHA-1
class SHA1(MDHash):
    """SHA-1 (FIPS 180-4). Big-endian words and length field."""

    name = "sha1"
    block_size = 64
    digest_size = 20
    length_size = 8
    byteorder = "big"

    @classmethod
    def initial_state(cls) -> tuple[int, ...]:
        return (0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0)

    @classmethod
    def compress(cls, state: tuple[int, ...], block: bytes) -> tuple[int, ...]:
        mask = 0xFFFFFFFF
        w = list(struct.unpack(">16I", block))
        for t in range(16, 80):
            w.append(rotl(w[t - 3] ^ w[t - 8] ^ w[t - 14] ^ w[t - 16], 1))
        a, b, c, d, e = state
        for t in range(80):
            if t < 20:
                f, k = (b & c) | (~b & d), 0x5A827999
            elif t < 40:
                f, k = b ^ c ^ d, 0x6ED9EBA1
            elif t < 60:
                f, k = (b & c) | (b & d) | (c & d), 0x8F1BBCDC
            else:
                f, k = b ^ c ^ d, 0xCA62C1D6
            tmp = (rotl(a, 5) + (f & mask) + e + k + w[t]) & mask
            a, b, c, d, e = tmp, a, rotl(b, 30), c, d
        return tuple((x + y) & mask for x, y in zip(state, (a, b, c, d, e)))

    @classmethod
    def pack_state(cls, state: Sequence[int]) -> bytes:
        return struct.pack(">5I", *state)

    @classmethod
    def unpack_state(cls, digest: bytes) -> tuple[int, ...]:
        return struct.unpack(">5I", digest)


# -------------------------------------------------------------------------- SHA-2
def _frac_root(p: int, k: int, bits: int) -> int:
    """First ``bits`` bits of the fractional part of the k-th root of ``p``."""
    return iroot(p << (k * bits), k) & ((1 << bits) - 1)


_PRIMES = first_primes(80)
_K256 = [_frac_root(p, 3, 32) for p in _PRIMES[:64]]
_H256 = tuple(_frac_root(p, 2, 32) for p in _PRIMES[:8])
_K512 = [_frac_root(p, 3, 64) for p in _PRIMES]
_H512 = tuple(_frac_root(p, 2, 64) for p in _PRIMES[:8])


class SHA256(MDHash):
    """SHA-256 (FIPS 180-4)."""

    name = "sha256"
    block_size = 64
    digest_size = 32
    length_size = 8
    byteorder = "big"

    @classmethod
    def initial_state(cls) -> tuple[int, ...]:
        return _H256

    @classmethod
    def compress(cls, state: tuple[int, ...], block: bytes) -> tuple[int, ...]:
        mask = 0xFFFFFFFF
        w = list(struct.unpack(">16I", block))
        for t in range(16, 64):
            x, y = w[t - 15], w[t - 2]
            s0 = rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3)
            s1 = rotr(y, 17) ^ rotr(y, 19) ^ (y >> 10)
            w.append((w[t - 16] + s0 + w[t - 7] + s1) & mask)
        a, b, c, d, e, f, g, h = state
        for t in range(64):
            s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)
            ch = (e & f) ^ (~e & mask & g)
            t1 = (h + s1 + ch + _K256[t] + w[t]) & mask
            s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)
            maj = (a & b) ^ (a & c) ^ (b & c)
            t2 = (s0 + maj) & mask
            h, g, f, e, d, c, b, a = g, f, e, (d + t1) & mask, c, b, a, (t1 + t2) & mask
        return tuple((x + y) & mask for x, y in zip(state, (a, b, c, d, e, f, g, h)))

    @classmethod
    def pack_state(cls, state: Sequence[int]) -> bytes:
        return struct.pack(">8I", *state)

    @classmethod
    def unpack_state(cls, digest: bytes) -> tuple[int, ...]:
        return struct.unpack(">8I", digest)


class SHA512(MDHash):
    """SHA-512 (FIPS 180-4): 64-bit words, 128-byte blocks, 128-bit length."""

    name = "sha512"
    block_size = 128
    digest_size = 64
    length_size = 16
    byteorder = "big"

    @classmethod
    def initial_state(cls) -> tuple[int, ...]:
        return _H512

    @classmethod
    def compress(cls, state: tuple[int, ...], block: bytes) -> tuple[int, ...]:
        mask = (1 << 64) - 1
        w = list(struct.unpack(">16Q", block))
        for t in range(16, 80):
            x, y = w[t - 15], w[t - 2]
            s0 = rotr(x, 1, 64) ^ rotr(x, 8, 64) ^ (x >> 7)
            s1 = rotr(y, 19, 64) ^ rotr(y, 61, 64) ^ (y >> 6)
            w.append((w[t - 16] + s0 + w[t - 7] + s1) & mask)
        a, b, c, d, e, f, g, h = state
        for t in range(80):
            s1 = rotr(e, 14, 64) ^ rotr(e, 18, 64) ^ rotr(e, 41, 64)
            ch = (e & f) ^ (~e & mask & g)
            t1 = (h + s1 + ch + _K512[t] + w[t]) & mask
            s0 = rotr(a, 28, 64) ^ rotr(a, 34, 64) ^ rotr(a, 39, 64)
            maj = (a & b) ^ (a & c) ^ (b & c)
            t2 = (s0 + maj) & mask
            h, g, f, e, d, c, b, a = g, f, e, (d + t1) & mask, c, b, a, (t1 + t2) & mask
        return tuple((x + y) & mask for x, y in zip(state, (a, b, c, d, e, f, g, h)))

    @classmethod
    def pack_state(cls, state: Sequence[int]) -> bytes:
        return struct.pack(">8Q", *state)

    @classmethod
    def unpack_state(cls, digest: bytes) -> tuple[int, ...]:
        return struct.unpack(">8Q", digest)


ALGORITHMS: dict[str, type[MDHash]] = {c.name: c for c in (MD5, SHA1, SHA256, SHA512)}
