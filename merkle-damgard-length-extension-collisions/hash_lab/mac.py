"""Message authentication and key derivation built on :class:`MDHash`.

Includes the *vulnerable* prefix MAC ``H(key || msg)`` that the length-extension
attack targets, and the HMAC construction (RFC 2104) that fixes it, together
with HKDF (RFC 5869) and PBKDF2 (RFC 8018).
"""
from __future__ import annotations

from typing import Type

from .mdcore import MDHash

__all__ = ["prefix_mac", "hmac", "hkdf", "pbkdf2", "constant_time_equal"]


def prefix_mac(hash_cls: Type[MDHash], key: bytes, msg: bytes) -> bytes:
    """Naive secret-prefix MAC ``H(key || msg)`` -- insecure for MD hashes."""
    return hash_cls(key + msg).digest()


def hmac(hash_cls: Type[MDHash], key: bytes, msg: bytes) -> bytes:
    """HMAC (RFC 2104)."""
    bs = hash_cls.block_size
    if len(key) > bs:
        key = hash_cls.hash(key)
    key = key.ljust(bs, b"\x00")
    ipad = bytes(b ^ 0x36 for b in key)
    opad = bytes(b ^ 0x5C for b in key)
    return hash_cls(opad + hash_cls(ipad + msg).digest()).digest()


def hkdf(hash_cls: Type[MDHash], ikm: bytes, length: int, salt: bytes = b"",
         info: bytes = b"") -> bytes:
    """HKDF extract-and-expand (RFC 5869)."""
    n = hash_cls.digest_size
    if length > 255 * n:
        raise ValueError("requested length too large")
    prk = hmac(hash_cls, salt or b"\x00" * n, ikm)
    out, block = b"", b""
    counter = 1
    while len(out) < length:
        block = hmac(hash_cls, prk, block + info + bytes([counter]))
        out += block
        counter += 1
    return out[:length]


def pbkdf2(hash_cls: Type[MDHash], password: bytes, salt: bytes, iterations: int,
           dklen: int) -> bytes:
    """PBKDF2-HMAC (RFC 8018)."""
    if iterations < 1:
        raise ValueError("iterations must be positive")
    out = b""
    index = 1
    while len(out) < dklen:
        u = hmac(hash_cls, password, salt + index.to_bytes(4, "big"))
        acc = int.from_bytes(u, "big")
        for _ in range(iterations - 1):
            u = hmac(hash_cls, password, u)
            acc ^= int.from_bytes(u, "big")
        out += acc.to_bytes(hash_cls.digest_size, "big")
        index += 1
    return out[:dklen]


def constant_time_equal(a: bytes, b: bytes) -> bool:
    """Compare two byte strings without early exit."""
    if len(a) != len(b):
        return False
    diff = 0
    for x, y in zip(a, b):
        diff |= x ^ y
    return diff == 0
