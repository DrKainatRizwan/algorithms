"""ECDSA signing (RFC 6979 deterministic nonces), verification and public-key recovery."""

from __future__ import annotations

import hashlib
import hmac

from .curves import Curve, Point
from .field import inv_mod


def bits2int(curve: Curve, data: bytes) -> int:
    """Convert a hash to an integer, truncating to the bit length of the group order."""
    v = int.from_bytes(data, "big")
    excess = len(data) * 8 - curve.n.bit_length()
    return v >> excess if excess > 0 else v


def rfc6979_nonce(curve: Curve, d: int, digest: bytes, hashfunc=hashlib.sha256) -> int:
    """Deterministic nonce generation following RFC 6979 section 3.2."""
    qlen = (curve.n.bit_length() + 7) // 8
    hlen = hashfunc().digest_size
    x = d.to_bytes(qlen, "big")
    h1 = (bits2int(curve, digest) % curve.n).to_bytes(qlen, "big")
    V, K = b"\x01" * hlen, b"\x00" * hlen
    K = hmac.new(K, V + b"\x00" + x + h1, hashfunc).digest()
    V = hmac.new(K, V, hashfunc).digest()
    K = hmac.new(K, V + b"\x01" + x + h1, hashfunc).digest()
    V = hmac.new(K, V, hashfunc).digest()
    while True:
        T = b""
        while len(T) < qlen:
            V = hmac.new(K, V, hashfunc).digest()
            T += V
        k = bits2int(curve, T[:qlen] if qlen * 8 >= curve.n.bit_length() else T)
        if 1 <= k < curve.n:
            return k
        K = hmac.new(K, V + b"\x00", hashfunc).digest()
        V = hmac.new(K, V, hashfunc).digest()


def sign(curve: Curve, d: int, digest: bytes, k: int | None = None,
         low_s: bool = False) -> tuple[int, int]:
    """Sign a message digest. ``k`` may be supplied explicitly (insecure; used to demonstrate attacks)."""
    z = bits2int(curve, digest)
    n = curve.n
    attempt = 0
    while True:
        nonce = k if k is not None else rfc6979_nonce(curve, d, digest + bytes([attempt] * (attempt > 0)))
        R = curve.mul(nonce)
        assert R is not None
        r = R[0] % n
        s = inv_mod(nonce, n) * (z + r * d) % n if r else 0
        if r and s:
            if low_s and s > n // 2:
                s = n - s
            return r, s
        if k is not None:
            raise ValueError("supplied nonce produces a degenerate signature")
        attempt += 1


def verify(curve: Curve, Q: Point, digest: bytes, sig: tuple[int, int]) -> bool:
    """Verify an ECDSA signature using Shamir's trick for ``u1*G + u2*Q``."""
    r, s = sig
    n = curve.n
    if not (1 <= r < n and 1 <= s < n) or Q is None or not curve.is_on_curve(Q):
        return False
    w = inv_mod(s, n)
    z = bits2int(curve, digest)
    X = curve.mul_add(z * w % n, curve.G, r * w % n, Q)
    return X is not None and X[0] % n == r


def recover_public_keys(curve: Curve, digest: bytes, sig: tuple[int, int]) -> list[tuple[int, int]]:
    """Return every public key consistent with ``(digest, sig)``: Q = r^-1 (sR - zG)."""
    r, s = sig
    n, p = curve.n, curve.p
    z = bits2int(curve, digest)
    rinv = inv_mod(r, n)
    keys: list[tuple[int, int]] = []
    for j in range(curve.h + 1 if p > n else 1):
        x = r + j * n
        if x >= p:
            break
        for odd in (False, True):
            R = curve.lift_x(x, odd)
            if R is None or curve.mul(n, R) is not None:
                continue
            Q = curve.mul_add(s * rinv % n, R, (-z * rinv) % n, curve.G)
            if Q is not None and verify(curve, Q, digest, sig) and Q not in keys:
                keys.append(Q)
    return keys
