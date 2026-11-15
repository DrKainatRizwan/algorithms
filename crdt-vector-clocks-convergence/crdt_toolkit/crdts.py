"""State-based CRDTs (convergent replicated data types).

Every type implements ``merge`` as a join in a semilattice, so merge is
commutative, associative and idempotent.  ``canonical`` returns a plain,
comparable snapshot of the full state used for equality checks.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import copy as _copy
from typing import Any, Hashable

from .clocks import VectorClock


class CRDT(ABC):
    """Base class for state-based CRDTs."""

    def __init__(self, node_id: str) -> None:
        self.node_id = node_id

    @abstractmethod
    def merge(self, other: "CRDT") -> None:
        """Join ``other``'s state into this replica (in place)."""

    @abstractmethod
    def canonical(self) -> Any:
        """Comparable snapshot of the whole internal state."""

    @abstractmethod
    def value(self) -> Any:
        """The user-visible value."""

    def copy(self) -> "CRDT":
        return _copy.deepcopy(self)

    def joined(self, other: "CRDT") -> "CRDT":
        """Return a new replica equal to ``self`` joined with ``other``."""
        r = self.copy()
        r.merge(other)
        return r

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other) and self.canonical() == other.canonical()  # type: ignore[attr-defined]

    __hash__ = None  # type: ignore[assignment]


class GCounter(CRDT):
    """Grow-only counter: one slot per node, merged by pointwise max."""

    def __init__(self, node_id: str) -> None:
        super().__init__(node_id)
        self.counts: dict[str, int] = {}

    def increment(self, n: int = 1) -> None:
        if n < 0:
            raise ValueError("GCounter only grows")
        self.counts[self.node_id] = self.counts.get(self.node_id, 0) + n

    def merge(self, other: "GCounter") -> None:  # type: ignore[override]
        for k, v in other.counts.items():
            if v > self.counts.get(k, 0):
                self.counts[k] = v

    def value(self) -> int:
        return sum(self.counts.values())

    def canonical(self) -> Any:
        return tuple(sorted((k, v) for k, v in self.counts.items() if v))


class PNCounter(CRDT):
    """Counter supporting increments and decrements (two GCounters)."""

    def __init__(self, node_id: str) -> None:
        super().__init__(node_id)
        self.p = GCounter(node_id)
        self.n = GCounter(node_id)

    def increment(self, k: int = 1) -> None:
        self.p.increment(k)

    def decrement(self, k: int = 1) -> None:
        self.n.increment(k)

    def merge(self, other: "PNCounter") -> None:  # type: ignore[override]
        self.p.merge(other.p)
        self.n.merge(other.n)

    def value(self) -> int:
        return self.p.value() - self.n.value()

    def canonical(self) -> Any:
        return (self.p.canonical(), self.n.canonical())


class GSet(CRDT):
    """Grow-only set."""

    def __init__(self, node_id: str) -> None:
        super().__init__(node_id)
        self.items: set[Hashable] = set()

    def add(self, x: Hashable) -> None:
        self.items.add(x)

    def merge(self, other: "GSet") -> None:  # type: ignore[override]
        self.items |= other.items

    def value(self) -> frozenset:
        return frozenset(self.items)

    def canonical(self) -> Any:
        return tuple(sorted(self.items, key=repr))


class TwoPSet(CRDT):
    """Two-phase set: an element removed once can never be re-added."""

    def __init__(self, node_id: str) -> None:
        super().__init__(node_id)
        self.added: set[Hashable] = set()
        self.removed: set[Hashable] = set()

    def add(self, x: Hashable) -> None:
        self.added.add(x)

    def remove(self, x: Hashable) -> None:
        if x in self.added:
            self.removed.add(x)

    def merge(self, other: "TwoPSet") -> None:  # type: ignore[override]
        self.added |= other.added
        self.removed |= other.removed

    def value(self) -> frozenset:
        return frozenset(self.added - self.removed)

    def canonical(self) -> Any:
        return (tuple(sorted(self.added, key=repr)), tuple(sorted(self.removed, key=repr)))


class ORSet(CRDT):
    """Observed-remove set (add-wins).

    Each add creates a unique tag ``(node, seq)``.  A remove tombstones only
    the tags it has *observed*, so a concurrent add survives the remove.
    """

    def __init__(self, node_id: str) -> None:
        super().__init__(node_id)
        self.tags: dict[Hashable, set[tuple[str, int]]] = {}
        self.tombstones: set[tuple[str, int]] = set()
        self._seq = 0

    def add(self, x: Hashable) -> tuple[str, int]:
        self._seq += 1
        tag = (self.node_id, self._seq)
        self.tags.setdefault(x, set()).add(tag)
        return tag

    def remove(self, x: Hashable) -> None:
        self.tombstones |= self.tags.get(x, set())

    def contains(self, x: Hashable) -> bool:
        return bool(self.tags.get(x, set()) - self.tombstones)

    def merge(self, other: "ORSet") -> None:  # type: ignore[override]
        for x, ts in other.tags.items():
            self.tags.setdefault(x, set()).update(ts)
        self.tombstones |= other.tombstones

    def value(self) -> frozenset:
        return frozenset(x for x, ts in self.tags.items() if ts - self.tombstones)

    def canonical(self) -> Any:
        live = {x: tuple(sorted(ts)) for x, ts in self.tags.items() if ts}
        return (tuple(sorted(live.items(), key=lambda kv: repr(kv[0]))),
                tuple(sorted(self.tombstones)))


class LWWRegister(CRDT):
    """Last-writer-wins register; ties on timestamp are broken by node id."""

    def __init__(self, node_id: str) -> None:
        super().__init__(node_id)
        self.val: Any = None
        self.stamp: tuple[int, str] = (0, "")

    def set(self, value: Any, timestamp: int) -> None:
        stamp = (timestamp, self.node_id)
        if stamp > self.stamp:
            self.val, self.stamp = value, stamp

    def merge(self, other: "LWWRegister") -> None:  # type: ignore[override]
        if other.stamp > self.stamp:
            self.val, self.stamp = other.val, other.stamp

    def value(self) -> Any:
        return self.val

    def canonical(self) -> Any:
        return (repr(self.val), self.stamp)


class MVRegister(CRDT):
    """Multi-value register: concurrent writes are all preserved as siblings."""

    def __init__(self, node_id: str) -> None:
        super().__init__(node_id)
        self.entries: list[tuple[Any, VectorClock]] = []

    def _context(self) -> VectorClock:
        ctx = VectorClock()
        for _, vc in self.entries:
            ctx = ctx.merged(vc)
        return ctx

    def assign(self, value: Any) -> None:
        """Overwrite everything this replica has seen with ``value``."""
        self.entries = [(value, self._context().incremented(self.node_id))]

    def merge(self, other: "MVRegister") -> None:  # type: ignore[override]
        pool = self.entries + other.entries
        keep: list[tuple[Any, VectorClock]] = []
        for v, vc in pool:
            dominated = any(vc2 != vc and vc2.dominates(vc) for _, vc2 in pool)
            if not dominated and not any(vc == k and repr(v) == repr(kv) for kv, k in keep):
                keep.append((v, vc))
        self.entries = keep

    def value(self) -> frozenset:
        return frozenset(v for v, _ in self.entries)

    def canonical(self) -> Any:
        return tuple(sorted(((repr(v), vc.key()) for v, vc in self.entries)))


class LWWMap(CRDT):
    """Map from keys to LWW registers with tombstoned (timestamped) deletes."""

    def __init__(self, node_id: str) -> None:
        super().__init__(node_id)
        self.slots: dict[Hashable, tuple[tuple[int, str], Any, bool]] = {}

    def put(self, key: Hashable, value: Any, timestamp: int) -> None:
        self._write(key, (timestamp, self.node_id), value, False)

    def delete(self, key: Hashable, timestamp: int) -> None:
        self._write(key, (timestamp, self.node_id), None, True)

    def _write(self, key: Hashable, stamp: tuple[int, str], value: Any, dead: bool) -> None:
        cur = self.slots.get(key)
        if cur is None or stamp > cur[0]:
            self.slots[key] = (stamp, value, dead)

    def merge(self, other: "LWWMap") -> None:  # type: ignore[override]
        for k, (stamp, v, dead) in other.slots.items():
            self._write(k, stamp, v, dead)

    def value(self) -> dict:
        return {k: v for k, (_, v, dead) in self.slots.items() if not dead}

    def canonical(self) -> Any:
        return tuple(sorted(((repr(k), s, repr(v), d) for k, (s, v, d) in self.slots.items())))
