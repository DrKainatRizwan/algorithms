"""Signal controllers: fixed-time, longest-queue-first, max-pressure and learned policies."""
from __future__ import annotations

from typing import Protocol

import numpy as np

from .agents import StateEncoder
from .network import TrafficGrid


class Controller(Protocol):
    def reset(self) -> None: ...
    def act(self, env: TrafficGrid) -> np.ndarray: ...


class FixedTime:
    """Alternate phases every ``green`` seconds regardless of traffic."""

    def __init__(self, green: int = 20) -> None:
        self.green = green
        self.name = f"fixed({green}s)"

    def reset(self) -> None:
        pass

    def act(self, env: TrafficGrid) -> np.ndarray:
        return np.array([int(env.green_time[i] >= self.green and env.green_ok(i))
                         for i in range(env.n)])


class LongestQueueFirst:
    """Switch whenever the waiting group has strictly more vehicles than the served one."""

    name = "longest-queue"

    def __init__(self, margin: int = 2) -> None:
        self.margin = margin

    def reset(self) -> None:
        pass

    def act(self, env: TrafficGrid) -> np.ndarray:
        out = np.zeros(env.n, dtype=np.int64)
        for i in range(env.n):
            ns, ew = env.group_queues(i)
            served, waiting = (ns, ew) if env.phase[i] == 0 else (ew, ns)
            out[i] = int(env.green_ok(i) and waiting >= served + self.margin)
        return out


class MaxPressure:
    """Max-pressure control (Varaiya, 2013): pressure = upstream queue minus downstream queue."""

    name = "max-pressure"

    def reset(self) -> None:
        pass

    def _pressure(self, env: TrafficGrid, node: int, dirs: tuple[int, int]) -> float:
        total = 0.0
        for d in dirs:
            nxt = env._neighbour(node, d)
            down = 0.0 if nxt is None else float(env.queue[nxt, d])
            total += env.queue[node, d] - down
        return total

    def act(self, env: TrafficGrid) -> np.ndarray:
        out = np.zeros(env.n, dtype=np.int64)
        for i in range(env.n):
            p_ns = self._pressure(env, i, (0, 1))
            p_ew = self._pressure(env, i, (2, 3))
            better = 0 if p_ns > p_ew else 1 if p_ew > p_ns else int(env.phase[i])
            out[i] = int(env.green_ok(i) and better != env.phase[i])
        return out


class RandomPolicy:
    """Switch with a fixed probability (sanity baseline)."""

    name = "random"

    def __init__(self, p: float = 0.2, seed: int = 0) -> None:
        self.p, self.seed = p, seed
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def act(self, env: TrafficGrid) -> np.ndarray:
        return (self.rng.random(env.n) < self.p).astype(np.int64)


class TablePolicy:
    """Greedy policy backed by a learned Q table (shared across all intersections)."""

    def __init__(self, q: np.ndarray, encoder: StateEncoder, name: str = "q-learning") -> None:
        self.q, self.encoder, self.name = q, encoder, name

    def reset(self) -> None:
        pass

    def act(self, env: TrafficGrid) -> np.ndarray:
        return np.array([int(np.argmax(self.q[self.encoder.encode(env.observe(i))]))
                         for i in range(env.n)])
