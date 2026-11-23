"""Distinguished-point (DP) variant of Hellman tables (Rivest's refinement).

A chain stops at the first point whose low ``d`` bits are zero.  Online, each
table needs a single dictionary lookup per chain walk rather than one per
step, which removes most random disk accesses in practice.  Chains that
loop or exceed ``max_len`` are discarded, which also cleans loops from the table.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .oneway import OneWayFunction


@dataclass
class DistinguishedPointTable:
    """One DP table; ``ends`` maps endpoint -> list of (start, length)."""

    f: OneWayFunction
    m: int
    d: int
    variant: int
    max_len: int = 0
    seed: int = 0
    ends: dict[int, list[tuple[int, int]]] = field(default_factory=dict, init=False)
    discarded: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.max_len <= 0:
            self.max_len = 8 * (1 << self.d)
        self.dp_mask = (1 << self.d) - 1

    def step(self, x: int) -> int:
        return self.f(x) ^ (self.variant & self.f.mask)

    def is_dp(self, x: int) -> bool:
        return (x & self.dp_mask) == 0

    def walk(self, x: int) -> tuple[int, int] | None:
        """Walk to the next DP; return ``(dp, steps)`` or ``None`` if too long."""
        for n in range(1, self.max_len + 1):
            x = self.step(x)
            if self.is_dp(x):
                return x, n
        return None

    def build(self) -> "DistinguishedPointTable":
        rng = random.Random(self.seed * 104729 + self.variant)
        self.ends, self.discarded = {}, 0
        seen_pairs: set[tuple[int, int]] = set()
        for _ in range(self.m):
            start = rng.randrange(self.f.size)
            res = self.walk(start)
            if res is None:
                self.discarded += 1
                continue
            end, n = res
            if (end, n) in seen_pairs:  # identical merged chain, redundant
                self.discarded += 1
                continue
            seen_pairs.add((end, n))
            self.ends.setdefault(end, []).append((start, n))
        return self

    @property
    def stored_pairs(self) -> int:
        return sum(len(v) for v in self.ends.values())

    def lookup(self, y: int) -> tuple[int | None, int]:
        """Return ``(preimage or None, false_alarms)`` for ``f(x) == y``."""
        target = y ^ (self.variant & self.f.mask)
        res = self._walk_from_target(target)
        if res is None:
            return None, 0
        end, steps = res
        false_alarms = 0
        for start, n in self.ends.get(end, []):
            x = start
            for _ in range(n):
                nxt = self.step(x)
                if nxt == target and self.f(x) == y:
                    return x, false_alarms
                x = nxt
            false_alarms += 1
        return None, false_alarms

    def _walk_from_target(self, x: int) -> tuple[int, int] | None:
        if self.is_dp(x):
            return x, 0
        return self.walk(x)


class DistinguishedPointAttack:
    """Several DP tables with distinct variants."""

    def __init__(self, f: OneWayFunction, m: int, d: int, tables: int, seed: int = 0):
        self.f = f
        self.tables = [
            DistinguishedPointTable(f, m, d, variant=i + 1, seed=seed) for i in range(tables)
        ]
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
