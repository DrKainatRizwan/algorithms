"""Hellman's original time-memory trade-off (1980).

A table holds ``m`` chains of length ``t`` built with the step function
``F_i(x) = f(x) XOR i``.  Only (start, end) pairs are stored.  Using ``l``
tables with different masks ``i`` makes the union of tables cover the space
despite chain merges.  With ``m * t**2 = N`` (the matrix stopping rule) the
trade-off is ``T * M**2 = N**2`` for ``T = l*t`` online work.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .oneway import OneWayFunction


@dataclass
class HellmanTable:
    """One Hellman table with its own step-function variant."""

    f: OneWayFunction
    m: int
    t: int
    variant: int
    seed: int = 0
    ends: dict[int, list[int]] = field(default_factory=dict, init=False)

    def step(self, x: int) -> int:
        """Variant step function ``F_i(x) = f(x) xor i`` (masked to the space)."""
        return self.f(x) ^ (self.variant & self.f.mask)

    def build(self) -> "HellmanTable":
        """Generate ``m`` chains from random starts (offline phase)."""
        rng = random.Random(self.seed * 7919 + self.variant)
        self.ends = {}
        for _ in range(self.m):
            start = rng.randrange(self.f.size)
            x = start
            for _ in range(self.t):
                x = self.step(x)
            self.ends.setdefault(x, []).append(start)
        return self

    @property
    def stored_pairs(self) -> int:
        return sum(len(v) for v in self.ends.values())

    def lookup(self, y: int) -> tuple[int | None, int]:
        """Try to find ``x`` with ``f(x) == y``.

        Returns ``(preimage or None, false_alarms)``.  A false alarm is an
        endpoint match whose regenerated chain does not contain a preimage.
        """
        target = y ^ (self.variant & self.f.mask)  # equals F_i(x) if f(x)=y
        cur = target
        false_alarms = 0
        for _ in range(self.t):
            starts = self.ends.get(cur)
            if starts is not None:
                found = False
                for s in starts:
                    cand = self._find_in_chain(s, target, y)
                    if cand is not None:
                        return cand, false_alarms
                    found = True
                if found:
                    false_alarms += 1
            cur = self.step(cur)
        return None, false_alarms

    def _find_in_chain(self, start: int, target: int, y: int) -> int | None:
        x = start
        for _ in range(self.t):
            nxt = self.step(x)
            if nxt == target and self.f(x) == y:
                return x
            x = nxt
        return None


class HellmanAttack:
    """A set of ``l`` Hellman tables with distinct step-function variants."""

    def __init__(self, f: OneWayFunction, m: int, t: int, tables: int, seed: int = 0):
        self.f = f
        self.tables = [HellmanTable(f, m, t, variant=i + 1, seed=seed) for i in range(tables)]
        self.false_alarms = 0

    def precompute(self) -> int:
        """Build all tables; returns the number of ``f`` evaluations used."""
        self.f.reset_counter()
        for tb in self.tables:
            tb.build()
        return self.f.reset_counter()

    @property
    def memory(self) -> int:
        """Stored (start, end) pairs."""
        return sum(tb.stored_pairs for tb in self.tables)

    def invert(self, y: int) -> int | None:
        """Return a preimage of ``y`` under ``f`` or ``None``."""
        for tb in self.tables:
            x, fa = tb.lookup(y)
            self.false_alarms += fa
            if x is not None:
                return x
        return None
