"""Keystream generators built from LFSRs: Geffe, threshold and filter generators."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .lfsr import LFSR


@dataclass
class GeffeGenerator:
    """Geffe generator: ``z = x2`` if ``x1 = 1`` else ``x3``.

    Output agrees with ``x2`` and with ``x3`` with probability 3/4 each, which
    is the classic target of Siegenthaler's correlation attack.
    """

    r1: LFSR  # selector
    r2: LFSR
    r3: LFSR

    def keystream(self, states: Sequence[int], nbits: int) -> np.ndarray:
        a = self.r1.stream(states[0], nbits)
        b = self.r2.stream(states[1], nbits)
        c = self.r3.stream(states[2], nbits)
        return np.where(a == 1, b, c).astype(np.uint8)

    @property
    def linear_complexity_bound(self) -> int:
        """Upper bound ``n1 n2 + n1 n3 + n3`` on the linear complexity."""
        n1, n2, n3 = self.r1.length, self.r2.length, self.r3.length
        return n1 * n2 + n1 * n3 + n3


@dataclass
class ThresholdGenerator:
    """Majority-of-three generator; each input correlates with output at 3/4."""

    regs: tuple[LFSR, LFSR, LFSR]

    def keystream(self, states: Sequence[int], nbits: int) -> np.ndarray:
        s = sum(r.stream(st, nbits).astype(np.int8) for r, st in zip(self.regs, states))
        return (s >= 2).astype(np.uint8)


@dataclass
class FilterGenerator:
    """Non-linear filter generator.

    The output is ``f`` evaluated on the register bits at ``positions``.  ``f``
    is given in algebraic normal form: a list of monomials, each a tuple of
    indices into ``positions`` (the empty tuple is the constant 1).
    """

    lfsr: LFSR
    positions: tuple[int, ...]
    anf: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if any(not 0 <= p < self.lfsr.length for p in self.positions):
            raise ValueError("filter position outside register")
        for mono in self.anf:
            if any(not 0 <= i < len(self.positions) for i in mono):
                raise ValueError("monomial index outside filter inputs")

    @property
    def degree(self) -> int:
        return max((len(m) for m in self.anf), default=0)

    def keystream(self, state: int, nbits: int) -> np.ndarray:
        seq = self.lfsr.stream(state, nbits + self.lfsr.length)
        inputs = [seq[p : p + nbits].astype(np.uint8) for p in self.positions]
        out = np.zeros(nbits, dtype=np.uint8)
        for mono in self.anf:
            term = np.ones(nbits, dtype=np.uint8)
            for i in mono:
                term &= inputs[i]
            out ^= term
        return out


def add_noise(bits: np.ndarray, p: float, rng: np.random.Generator) -> np.ndarray:
    """Pass ``bits`` through a binary symmetric channel with flip probability ``p``."""
    flips = (rng.random(len(bits)) < p).astype(np.uint8)
    return bits ^ flips
