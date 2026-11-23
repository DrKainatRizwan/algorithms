"""Rainbow tables (Oechslin, 2003).

Each column ``k`` of a chain uses its own reduction ``R_k``, so two chains
that collide in different columns do not merge.  One table of ``m`` chains
of length ``t`` is equivalent in coverage to roughly ``t`` classic tables
while a lookup costs ``t(t-1)/2`` evaluations in the worst case.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .oneway import OneWayFunction


@dataclass
class RainbowTable:
    """A single rainbow table; ``ends`` maps final point -> list of starts."""

    f: OneWayFunction
    m: int
    t: int
    seed: int = 0
    dedup: bool = True
    ends: dict[int, list[int]] = field(default_factory=dict, init=False)

    def reduce(self, y: int, column: int) -> int:
        """Column-dependent reduction back into the search space."""
        return (y + column * 0x9E3779B1 + column) & self.f.mask

    def chain_end(self, x: int, first: int, last: int) -> int:
        """Apply columns ``first..last-1``: x -> f -> R_col, repeatedly."""
        for col in range(first, last):
            x = self.reduce(self.f(x), col)
        return x

    def build(self) -> "RainbowTable":
        rng = random.Random(self.seed)
        self.ends = {}
        for _ in range(self.m):
            start = rng.randrange(self.f.size)
            end = self.chain_end(start, 0, self.t)
            bucket = self.ends.setdefault(end, [])
            if self.dedup and bucket:
                continue  # merged chain: drop duplicates of the same endpoint
            bucket.append(start)
        return self

    @property
    def stored_pairs(self) -> int:
        return sum(len(v) for v in self.ends.values())

    def lookup(self, y: int) -> tuple[int | None, int]:
        """Search every possible column position of ``y``; returns (x, false_alarms)."""
        false_alarms = 0
        for k in range(self.t - 1, -1, -1):
            z = self.reduce(y, k)
            z = self.chain_end(z, k + 1, self.t)
            starts = self.ends.get(z)
            if not starts:
                continue
            hit = False
            for s in starts:
                x = self.chain_end(s, 0, k)
                if self.f(x) == y:
                    return x, false_alarms
                hit = True
            if hit:
                false_alarms += 1
        return None, false_alarms


class RainbowAttack:
    """One or more rainbow tables with independent start points."""

    def __init__(self, f: OneWayFunction, m: int, t: int, tables: int = 1, seed: int = 0):
        self.f = f
        self.tables = [RainbowTable(f, m, t, seed=seed * 31 + i + 1) for i in range(tables)]
        self.false_alarms = 0

    def precompute(self) -> int:
        self.f.reset_counter()
        for tb in self.tables:
            tb.build()
        return self.f.reset_counter()

    @property
    def memory(self) -> int:
        return sum(tb.stored_pairs for tb in self.tables)

    def invert(self, y: int) -> int | None:
        for tb in self.tables:
            x, fa = tb.lookup(y)
            self.false_alarms += fa
            if x is not None:
                return x
        return None
