"""A Fujisaki-Okamoto style KEM on top of Ring-LWE (toy parameters, not for production)."""
from __future__ import annotations

import hashlib

import numpy as np

from .rlwe import RLWEParams, rlwe_decrypt, rlwe_encrypt, rlwe_keygen

MSG_BYTES = 32


def _hash(*parts: bytes, out: int = 32) -> bytes:
    h = hashlib.shake_256()
    for p in parts:
        h.update(len(p).to_bytes(4, "big") + p)
    return h.digest(out)


def _serialize_pub(pub) -> bytes:
    return b"".join(np.asarray(x, dtype=">i4").tobytes() for x in pub)


def _serialize_ct(ct) -> bytes:
    return b"".join(np.asarray(x, dtype=">i4").tobytes() for x in ct)


def _bytes_to_bits(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8)).astype(np.int64)


def _bits_to_bytes(bits: np.ndarray) -> bytes:
    return np.packbits(np.asarray(bits, dtype=np.uint8)).tobytes()


def _deterministic_encrypt(pub, params: RLWEParams, msg: bytes, coins: bytes):
    rng = np.random.default_rng(int.from_bytes(coins, "big"))
    return rlwe_encrypt(pub, params, _bytes_to_bits(msg), rng)


def kem_keygen(params: RLWEParams, rng: np.random.Generator):
    """Return (public_key, secret_key); the secret key carries a rejection seed."""
    pub, s = rlwe_keygen(params, rng)
    z = rng.bytes(32)
    return pub, (s, pub, z)


def kem_encaps(pub, params: RLWEParams, rng: np.random.Generator):
    """Return (ciphertext, shared_key). Coins are derived from the message (FO transform)."""
    m = rng.bytes(MSG_BYTES)
    h_pk = _hash(_serialize_pub(pub))
    kbar_coins = _hash(m, h_pk, out=64)
    kbar, coins = kbar_coins[:32], kbar_coins[32:]
    ct = _deterministic_encrypt(pub, params, m, coins)
    return ct, _hash(kbar, _hash(_serialize_ct(ct)))


def kem_decaps(sk, params: RLWEParams, ct) -> bytes:
    """Decapsulate with re-encryption check and implicit rejection."""
    s, pub, z = sk
    m = _bits_to_bytes(rlwe_decrypt(s, params, ct, MSG_BYTES * 8))
    h_pk = _hash(_serialize_pub(pub))
    kbar_coins = _hash(m, h_pk, out=64)
    kbar, coins = kbar_coins[:32], kbar_coins[32:]
    ct2 = _deterministic_encrypt(pub, params, m, coins)
    same = all(np.array_equal(x, y) for x, y in zip(ct, ct2))
    key_material = kbar if same else z
    return _hash(key_material, _hash(_serialize_ct(ct)))
