"""Key generation, ECDH key agreement, HKDF and a small ECIES-style hybrid scheme."""

from __future__ import annotations

import hashlib
import hmac
import random
from dataclasses import dataclass

from .curves import Curve, Point


@dataclass(frozen=True)
class KeyPair:
    curve: Curve
    private: int
    public: tuple[int, int]


def generate_keypair(curve: Curve, rng: random.Random | None = None) -> KeyPair:
    """Generate a key pair. Pass a seeded ``random.Random`` for reproducible output;
    otherwise the OS entropy source is used."""
    r = rng if rng is not None else random.SystemRandom()
    d = r.randrange(1, curve.n)
    Q = curve.mul(d)
    assert Q is not None
    return KeyPair(curve, d, Q)


def validate_public_key(curve: Curve, Q: Point) -> bool:
    """Full public-key validation: finite, on curve and in the prime-order subgroup."""
    if Q is None or not curve.is_on_curve(Q):
        return False
    return curve.h == 1 or curve.mul(curve.n, Q) is None


def ecdh_shared_secret(curve: Curve, private: int, peer_public: Point) -> bytes:
    """Return the x-coordinate of ``private * peer_public`` as a fixed-width byte string."""
    if not validate_public_key(curve, peer_public):
        raise ValueError("invalid peer public key")
    S = curve.mul(private % curve.n, peer_public)
    if S is None:
        raise ValueError("shared point is the point at infinity")
    return S[0].to_bytes(curve.byte_length, "big")


def hkdf_sha256(ikm: bytes, length: int, salt: bytes = b"", info: bytes = b"") -> bytes:
    """HKDF (RFC 5869) with HMAC-SHA256."""
    if length > 255 * 32:
        raise ValueError("requested length too large")
    prk = hmac.new(salt or b"\x00" * 32, ikm, hashlib.sha256).digest()
    out, t, i = b"", b"", 1
    while len(out) < length:
        t = hmac.new(prk, t + info + bytes([i]), hashlib.sha256).digest()
        out += t
        i += 1
    return out[:length]


def _keystream(key: bytes, n: int) -> bytes:
    out = b""
    ctr = 0
    while len(out) < n:
        out += hashlib.sha256(key + ctr.to_bytes(8, "big")).digest()
        ctr += 1
    return out[:n]


def ecies_encrypt(curve: Curve, recipient: Point, message: bytes,
                  rng: random.Random | None = None) -> bytes:
    """Encrypt-then-MAC with an ephemeral ECDH key (educational construction, not for production)."""
    eph = generate_keypair(curve, rng)
    secret = ecdh_shared_secret(curve, eph.private, recipient)
    keys = hkdf_sha256(secret, 64, info=b"ecies")
    ct = bytes(a ^ b for a, b in zip(message, _keystream(keys[:32], len(message))))
    tag = hmac.new(keys[32:], ct, hashlib.sha256).digest()
    return curve.encode_point(eph.public) + ct + tag


def ecies_decrypt(curve: Curve, private: int, blob: bytes) -> bytes:
    """Inverse of :func:`ecies_encrypt`; raises ``ValueError`` if authentication fails."""
    hdr = curve.byte_length + 1
    eph = curve.decode_point(blob[:hdr])
    ct, tag = blob[hdr:-32], blob[-32:]
    keys = hkdf_sha256(ecdh_shared_secret(curve, private, eph), 64, info=b"ecies")
    if not hmac.compare_digest(tag, hmac.new(keys[32:], ct, hashlib.sha256).digest()):
        raise ValueError("authentication failed")
    return bytes(a ^ b for a, b in zip(ct, _keystream(keys[:32], len(ct))))
