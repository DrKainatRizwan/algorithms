"""Logical clocks: Lamport scalar clocks and immutable vector clocks."""
from __future__ import annotations

from typing import Iterable, Iterator, Mapping

BEFORE = "before"
AFTER = "after"
EQUAL = "equal"
CONCURRENT = "concurrent"


class LamportClock:
    """Scalar logical clock (Lamport, 1978)."""

    def __init__(self, start: int = 0) -> None:
        self.time = start

    def tick(self) -> int:
        """Advance for a local event and return the new time."""
        self.time += 1
        return self.time

    def observe(self, remote: int) -> int:
        """Merge a remote timestamp (receive event) and advance."""
        self.time = max(self.time, remote) + 1
        return self.time


class VectorClock:
    """Immutable vector clock mapping node ids to event counts.

    Missing entries are treated as zero, so clocks over different node sets
    are directly comparable.
    """

    __slots__ = ("_c",)

    def __init__(self, counts: Mapping[str, int] | None = None) -> None:
        self._c: dict[str, int] = {k: v for k, v in (counts or {}).items() if v > 0}

    def get(self, node: str) -> int:
        return self._c.get(node, 0)

    def nodes(self) -> Iterator[str]:
        return iter(self._c)

    def items(self) -> Iterable[tuple[str, int]]:
        return self._c.items()

    def incremented(self, node: str) -> "VectorClock":
        """Return a new clock with ``node``'s entry increased by one."""
        c = dict(self._c)
        c[node] = c.get(node, 0) + 1
        return VectorClock(c)

    def merged(self, other: "VectorClock") -> "VectorClock":
        """Pointwise maximum (the join of the vector-clock lattice)."""
        c = dict(self._c)
        for k, v in other._c.items():
            if v > c.get(k, 0):
                c[k] = v
        return VectorClock(c)

    def dominates(self, other: "VectorClock") -> bool:
        """True iff every entry of ``self`` is >= the entry of ``other``."""
        return all(self.get(k) >= v for k, v in other._c.items())

    def compare(self, other: "VectorClock") -> str:
        """Return BEFORE, AFTER, EQUAL or CONCURRENT relative to ``other``."""
        ge = self.dominates(other)
        le = other.dominates(self)
        if ge and le:
            return EQUAL
        if le:
            return BEFORE
        if ge:
            return AFTER
        return CONCURRENT

    def key(self) -> tuple[tuple[str, int], ...]:
        return tuple(sorted(self._c.items()))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, VectorClock) and self._c == other._c

    def __hash__(self) -> int:
        return hash(self.key())

    def __repr__(self) -> str:
        return "VC(" + ", ".join(f"{k}:{v}" for k, v in self.key()) + ")"
