"""Deterministic simulated network with delay, loss and partitions."""
from __future__ import annotations

import heapq
import random
from typing import Dict, Iterable, List, Optional, Tuple

from .messages import Message


class Network:
    """Message transport driven by a seeded RNG.

    Messages get a random delay in ``[min_delay, max_delay]`` and may be
    dropped with probability ``drop_rate``. A partition is a list of node
    groups; only nodes in the same group can communicate.
    """

    def __init__(self, seed: int = 0, min_delay: int = 2, max_delay: int = 15,
                 drop_rate: float = 0.0):
        self.rng = random.Random(seed)
        self.min_delay, self.max_delay = min_delay, max_delay
        self.drop_rate = drop_rate
        self._queue: List[Tuple[int, int, int, int, Message]] = []
        self._seq = 0
        self._group: Optional[Dict[int, int]] = None
        self.sent = 0
        self.dropped = 0
        self.delivered = 0

    def partition(self, groups: Iterable[Iterable[int]]) -> None:
        self._group = {n: i for i, g in enumerate(groups) for n in g}

    def heal(self) -> None:
        self._group = None

    def connected(self, a: int, b: int) -> bool:
        if self._group is None:
            return True
        return a in self._group and b in self._group and self._group[a] == self._group[b]

    def send(self, src: int, dst: int, msg: Message, now: int) -> None:
        self.sent += 1
        if not self.connected(src, dst) or self.rng.random() < self.drop_rate:
            self.dropped += 1
            return
        t = now + self.rng.randint(self.min_delay, self.max_delay)
        self._seq += 1
        heapq.heappush(self._queue, (t, self._seq, src, dst, msg))

    def deliver_due(self, now: int) -> List[Tuple[int, Message]]:
        """Pop messages due at ``now``; those crossing a partition are lost."""
        out = []
        while self._queue and self._queue[0][0] <= now:
            _, _, src, dst, msg = heapq.heappop(self._queue)
            if self.connected(src, dst):
                out.append((dst, msg))
                self.delivered += 1
            else:
                self.dropped += 1
        return out

    def in_flight(self) -> int:
        return len(self._queue)
