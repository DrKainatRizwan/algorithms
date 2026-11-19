"""AES-128 implemented from scratch (FIPS-197), used as the attack target."""
from __future__ import annotations

import numpy as np

_POLY = 0x11B


def gf_mul(a: int, b: int) -> int:
    """Multiply two bytes in GF(2^8) modulo x^8+x^4+x^3+x+1."""
    r = 0
    while b:
        if b & 1:
            r ^= a
        a <<= 1
        if a & 0x100:
            a ^= _POLY
        b >>= 1
    return r


def _build_sbox() -> list[int]:
    inv = [0] * 256
    for x in range(1, 256):
        for y in range(1, 256):
            if gf_mul(x, y) == 1:
                inv[x] = y
                break
    box = []
    for x in range(256):
        b = inv[x]
        s = b
        for i in range(1, 5):
            s ^= ((b << i) | (b >> (8 - i))) & 0xFF
        box.append(s ^ 0x63)
    return box


SBOX: list[int] = _build_sbox()
SBOX_NP = np.array(SBOX, dtype=np.uint8)
HW = np.array([bin(i).count("1") for i in range(256)], dtype=np.float64)
_RCON = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36]


def expand_key(key: bytes) -> list[list[int]]:
    """Return the 11 round keys (each 16 bytes) of AES-128."""
    if len(key) != 16:
        raise ValueError("AES-128 key must be 16 bytes")
    w = [list(key[4 * i:4 * i + 4]) for i in range(4)]
    for i in range(4, 44):
        t = list(w[i - 1])
        if i % 4 == 0:
            t = t[1:] + t[:1]
            t = [SBOX[x] for x in t]
            t[0] ^= _RCON[i // 4 - 1]
        w.append([a ^ b for a, b in zip(w[i - 4], t)])
    return [sum(w[4 * r:4 * r + 4], []) for r in range(11)]


def invert_key_schedule(last_round_key: bytes) -> bytes:
    """Recover the master key from the round-10 key."""
    w = [None] * 44
    for i in range(4):
        w[40 + i] = list(last_round_key[4 * i:4 * i + 4])
    for i in range(43, 3, -1):
        if i % 4 != 0:
            w[i - 4] = [a ^ b for a, b in zip(w[i], w[i - 1])]
        else:
            # w[i] = w[i-4] ^ f(w[i-1]); w[i-1] is known from the next group
            t = list(w[i - 1])
            t = t[1:] + t[:1]
            t = [SBOX[x] for x in t]
            t[0] ^= _RCON[i // 4 - 1]
            w[i - 4] = [a ^ b for a, b in zip(w[i], t)]
    return bytes(sum(w[:4], []))


def _sub_shift(state: list[int]) -> list[int]:
    s = [SBOX[x] for x in state]
    out = [0] * 16
    for c in range(4):
        for r in range(4):
            out[4 * c + r] = s[4 * ((c + r) % 4) + r]
    return out


def _mix(state: list[int]) -> list[int]:
    out = [0] * 16
    for c in range(4):
        a = state[4 * c:4 * c + 4]
        out[4 * c + 0] = gf_mul(a[0], 2) ^ gf_mul(a[1], 3) ^ a[2] ^ a[3]
        out[4 * c + 1] = a[0] ^ gf_mul(a[1], 2) ^ gf_mul(a[2], 3) ^ a[3]
        out[4 * c + 2] = a[0] ^ a[1] ^ gf_mul(a[2], 2) ^ gf_mul(a[3], 3)
        out[4 * c + 3] = gf_mul(a[0], 3) ^ a[1] ^ a[2] ^ gf_mul(a[3], 2)
    return out


def encrypt_block(key: bytes, plaintext: bytes) -> bytes:
    """Encrypt one 16-byte block with AES-128."""
    rk = expand_key(key)
    s = [p ^ k for p, k in zip(plaintext, rk[0])]
    for r in range(1, 10):
        s = _mix(_sub_shift(s))
        s = [a ^ b for a, b in zip(s, rk[r])]
    s = _sub_shift(s)
    return bytes(a ^ b for a, b in zip(s, rk[10]))
