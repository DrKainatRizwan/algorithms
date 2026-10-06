"""Timestamped array: query the state of the data at any past moment."""
from __future__ import annotations

from bisect import bisect_right
from typing import Any, Sequence

from .monoid import Monoid, SUM
from .tree import PersistentSegmentTree


class TimeTravelArray:
    """Array whose every write is stamped with a non-decreasing timestamp.

    ``query(t, l, r)`` answers over the array as it stood after all writes
    with timestamp <= t. ``rollback`` forks a new head from an old state
    without discarding history.
    """

    def __init__(self, values: Sequence[Any], monoid: Monoid = SUM,
                 start_time: float = 0) -> None:
        self.tree = PersistentSegmentTree(values, monoid)
        self.times: list[float] = [start_time]
        self.versions: list[int] = [0]

    @property
    def head(self) -> int:
        return self.versions[-1]

    def set(self, timestamp: float, index: int, value: Any) -> int:
        """Write at ``timestamp`` (must not precede the previous write)."""
        if timestamp < self.times[-1]:
            raise ValueError("timestamps must be non-decreasing")
        v = self.tree.update(self.head, index, value)
        self.times.append(timestamp)
        self.versions.append(v)
        return v

    def version_at(self, timestamp: float) -> int:
        """Version id visible at ``timestamp``."""
        i = bisect_right(self.times, timestamp) - 1
        if i < 0:
            raise ValueError("timestamp precedes the beginning of history")
        return self.versions[i]

    def query(self, timestamp: float, left: int, right: int) -> Any:
        return self.tree.query(self.version_at(timestamp), left, right)

    def get(self, timestamp: float, index: int) -> Any:
        return self.tree.get(self.version_at(timestamp), index)

    def rollback(self, timestamp: float, new_time: float) -> int:
        """Make the state at ``timestamp`` the current head, stamped ``new_time``."""
        if new_time < self.times[-1]:
            raise ValueError("timestamps must be non-decreasing")
        v = self.version_at(timestamp)
        self.times.append(new_time)
        self.versions.append(v)
        return v

    def history(self, index: int) -> list[tuple[float, Any]]:
        """(timestamp, value) each time the value at ``index`` changed."""
        out: list[tuple[float, Any]] = []
        last = object()
        for t, v in zip(self.times, self.versions):
            val = self.tree.get(v, index)
            if val != last:
                out.append((t, val))
                last = val
        return out
