"""A 16-bit substitution-permutation network with four 4-bit S-boxes.

Layout (``R`` rounds, ``R + 1`` round keys ``K[0..R]``)::

    for r in 1..R-1:  x ^= K[r-1];  x = S(x);  x = P(x)
    last round:       x ^= K[R-1];  x = S(x);  x ^= K[R]

Nibble 0 / bit 0 is the most significant one, matching the usual cipher
notation.  All operations also work on ``numpy`` arrays for fast batch use.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .sbox import SBOX_HEYS, invert

BLOCK_BITS = 16
NIBBLES = 4


def default_permutation() -> tuple[int, ...]:
    """Bit ``i`` (MSB first) moves to position ``4*(i % 4) + i // 4``."""
    return tuple(4 * (i % 4) + i // 4 for i in range(BLOCK_BITS))


def nibble(x: int, i: int) -> int:
    """Return nibble ``i`` (0 = most significant) of a 16-bit word."""
    return (x >> (12 - 4 * i)) & 0xF


def set_nibble(x: int, i: int, v: int) -> int:
    """Return ``x`` with nibble ``i`` replaced by ``v``."""
    shift = 12 - 4 * i
    return (x & ~(0xF << shift) & 0xFFFF) | ((v & 0xF) << shift)


def nibbles(x: int) -> list[int]:
    """Split a 16-bit word into its four nibbles."""
    return [nibble(x, i) for i in range(NIBBLES)]


def from_nibbles(ns: list[int] | tuple[int, ...]) -> int:
    """Join four nibbles into a 16-bit word."""
    out = 0
    for v in ns:
        out = (out << 4) | (v & 0xF)
    return out


def active_nibbles(x: int) -> list[int]:
    """Indices of non-zero nibbles."""
    return [i for i in range(NIBBLES) if nibble(x, i)]


def _perm_table(perm: tuple[int, ...]) -> np.ndarray:
    tab = np.zeros(1 << BLOCK_BITS, dtype=np.uint16)
    xs = np.arange(1 << BLOCK_BITS, dtype=np.uint32)
    for i, p in enumerate(perm):
        tab |= (((xs >> (15 - i)) & 1) << (15 - p)).astype(np.uint16)
    return tab


@dataclass
class SPN:
    """Round-reduced SPN cipher with independent round keys.

    Parameters
    ----------
    keys:
        ``rounds + 1`` 16-bit round keys.
    sbox:
        Bijective 4-bit substitution table.
    perm:
        Bit permutation applied between rounds (MSB-first indices).
    """

    keys: list[int]
    sbox: tuple[int, ...] = SBOX_HEYS
    perm: tuple[int, ...] = field(default_factory=default_permutation)

    def __post_init__(self) -> None:
        if len(self.keys) < 2:
            raise ValueError("need at least two round keys")
        self.keys = [k & 0xFFFF for k in self.keys]
        self.rounds = len(self.keys) - 1
        self._inv_sbox = invert(self.sbox)
        self._inv_perm = tuple(np.argsort(self.perm))
        self._s_tab = _sub_table(self.sbox)
        self._si_tab = _sub_table(self._inv_sbox)
        self._p_tab = _perm_table(self.perm)
        self._pi_tab = _perm_table(self._inv_perm)

    # ------------------------------------------------------------------
    def encrypt(self, x):
        """Encrypt an ``int`` or a ``numpy`` array of 16-bit blocks."""
        scalar = not isinstance(x, np.ndarray)
        v = np.atleast_1d(np.asarray(x, dtype=np.uint16))
        for r in range(self.rounds - 1):
            v = self._p_tab[self._s_tab[v ^ np.uint16(self.keys[r])]]
        v = self._s_tab[v ^ np.uint16(self.keys[self.rounds - 1])]
        v = v ^ np.uint16(self.keys[self.rounds])
        return int(v[0]) if scalar else v

    def decrypt(self, c):
        """Inverse of :meth:`encrypt`."""
        scalar = not isinstance(c, np.ndarray)
        v = np.atleast_1d(np.asarray(c, dtype=np.uint16))
        v = self._si_tab[v ^ np.uint16(self.keys[self.rounds])]
        v = v ^ np.uint16(self.keys[self.rounds - 1])
        for r in range(self.rounds - 2, -1, -1):
            v = self._si_tab[self._pi_tab[v]] ^ np.uint16(self.keys[r])
        return int(v[0]) if scalar else v

    def last_round_input(self, x):
        """State entering the final S-box layer *before* the key ``K[R-1]``.

        This is the state a distinguisher over the first ``R-1`` rounds
        predicts (up to the constant key mask).
        """
        v = np.atleast_1d(np.asarray(x, dtype=np.uint16))
        for r in range(self.rounds - 1):
            v = self._p_tab[self._s_tab[v ^ np.uint16(self.keys[r])]]
        return v

    @classmethod
    def random(cls, rounds: int, rng: np.random.Generator,
               sbox: tuple[int, ...] = SBOX_HEYS) -> "SPN":
        """Build a cipher with uniformly random round keys."""
        keys = [int(k) for k in rng.integers(0, 1 << 16, size=rounds + 1)]
        return cls(keys, sbox=sbox)


def _sub_table(sbox: tuple[int, ...]) -> np.ndarray:
    """16-bit lookup table applying ``sbox`` to every nibble."""
    xs = np.arange(1 << BLOCK_BITS, dtype=np.uint32)
    out = np.zeros(1 << BLOCK_BITS, dtype=np.uint16)
    for i in range(NIBBLES):
        sh = 12 - 4 * i
        nib = (xs >> sh) & 0xF
        out |= (np.array(sbox, dtype=np.uint16)[nib] << sh).astype(np.uint16)
    return out


def permute(x: int, perm: tuple[int, ...] | None = None) -> int:
    """Apply the bit permutation to a single 16-bit word (MSB-first)."""
    perm = perm or default_permutation()
    out = 0
    for i, p in enumerate(perm):
        out |= ((x >> (15 - i)) & 1) << (15 - p)
    return out


def sub_nibbles(x: int, sbox: tuple[int, ...]) -> int:
    """Apply an S-box to each nibble of ``x``."""
    return from_nibbles([sbox[n] for n in nibbles(x)])
