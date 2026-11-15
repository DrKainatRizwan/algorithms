"""A deterministic lossy network: delay, drop, duplication, reordering, partitions."""
from __future__ import annotations

import heapq
import random
from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass
class NetStats:
    sent: int = 0
    dropped: int = 0
    duplicated: int = 0
    delivered: int = 0


@dataclass
class Network:
    """Discrete-time message transport.

    Messages get a random delay in ``[1, max_delay]`` (which reorders them),
    may be dropped with ``drop_prob`` and duplicated with ``dup_prob``.
    Messages crossing a partition boundary at send time are lost.
    """

    seed: int = 0
    drop_prob: float = 0.0
    dup_prob: float = 0.0
    max_delay: int = 1
    now: int = 0
    stats: NetStats = field(default_factory=NetStats)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self._queue: list[tuple[int, int, str, str, Any]] = []
        self._seq = 0
        self._group: dict[str, int] = {}

    def partition(self, groups: Iterable[Iterable[str]]) -> None:
        """Split nodes into mutually unreachable groups."""
        self._group = {n: i for i, g in enumerate(groups) for n in g}

    def heal(self) -> None:
        self._group = {}

    def reachable(self, a: str, b: str) -> bool:
        return self._group.get(a, -1) == self._group.get(b, -1)

    def send(self, src: str, dst: str, payload: Any) -> None:
        self.stats.sent += 1
        if not self.reachable(src, dst) or self._rng.random() < self.drop_prob:
            self.stats.dropped += 1
            return
        copies = 1
        if self._rng.random() < self.dup_prob:
            copies = 2
            self.stats.duplicated += 1
        for _ in range(copies):
            delay = self._rng.randint(1, max(1, self.max_delay))
            self._seq += 1
            heapq.heappush(self._queue, (self.now + delay, self._seq, src, dst, payload))

    def advance(self) -> list[tuple[str, str, Any]]:
        """Move time forward one tick and return messages now due."""
        self.now += 1
        due = []
        while self._queue and self._queue[0][0] <= self.now:
            _, _, src, dst, payload = heapq.heappop(self._queue)
            self.stats.delivered += 1
            due.append((src, dst, payload))
        return due

    def in_flight(self) -> int:
        return len(self._queue)
