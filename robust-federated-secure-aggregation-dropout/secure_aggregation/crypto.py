"""Minimal cryptographic building blocks for the aggregation protocol.

* Diffie-Hellman key agreement in the multiplicative group modulo ``2**255-19``
  (educational parameters: this is *not* a vetted prime-order group).
* SHA-256 based key derivation.
* A hash-counter stream cipher with an HMAC-SHA256 tag (encrypt-then-MAC) used
  to forward Shamir shares through the untrusted server.
* A seeded pseudo-random generator expanding a seed into a mask vector.
"""

from __future__ import annotations

import hashlib
import hmac
import random

import numpy as np

from .field import P

G: int = 2
TAG_LEN = 32


class AuthenticationError(Exception):
    """Raised when a ciphertext fails its integrity check."""


def dh_keygen(rng: random.Random) -> tuple[int, int]:
    """Return ``(secret, public)`` with ``public = G**secret mod P``."""
    sk = rng.randrange(2, P - 1)
    return sk, pow(G, sk, P)


def dh_agree(sk: int, peer_pk: int) -> int:
    """Shared group element ``peer_pk**sk mod P``."""
    if not 1 < peer_pk < P - 1:
        raise ValueError("invalid public key")
    return pow(peer_pk, sk, P)


def kdf(secret: int, label: bytes) -> bytes:
    """Derive a 32-byte key from an integer secret and a domain label."""
    return hashlib.sha256(label + b"|" + secret.to_bytes(32, "big")).digest()


def _keystream(key: bytes, nonce: bytes, n: int) -> bytes:
    out = bytearray()
    counter = 0
    while len(out) < n:
        out += hashlib.sha256(key + nonce + counter.to_bytes(4, "big")).digest()
        counter += 1
    return bytes(out[:n])


def encrypt(key: bytes, nonce: bytes, plaintext: bytes) -> bytes:
    """Encrypt-then-MAC; the returned blob is ``ciphertext || tag``."""
    ks = _keystream(key, nonce, len(plaintext))
    ct = bytes(a ^ b for a, b in zip(plaintext, ks))
    tag = hmac.new(key, nonce + ct, hashlib.sha256).digest()
    return ct + tag


def decrypt(key: bytes, nonce: bytes, blob: bytes) -> bytes:
    """Verify the tag and decrypt; raises :class:`AuthenticationError`."""
    if len(blob) < TAG_LEN:
        raise AuthenticationError("ciphertext too short")
    ct, tag = blob[:-TAG_LEN], blob[-TAG_LEN:]
    expect = hmac.new(key, nonce + ct, hashlib.sha256).digest()
    if not hmac.compare_digest(tag, expect):
        raise AuthenticationError("bad tag")
    ks = _keystream(key, nonce, len(ct))
    return bytes(a ^ b for a, b in zip(ct, ks))


def prg_vector(seed: bytes, length: int, bits: int = 32) -> np.ndarray:
    """Expand ``seed`` into ``length`` uniform integers in ``[0, 2**bits)``."""
    gen = np.random.Generator(np.random.PCG64(int.from_bytes(seed, "big")))
    return gen.integers(0, 1 << bits, size=length, dtype=np.uint64)
