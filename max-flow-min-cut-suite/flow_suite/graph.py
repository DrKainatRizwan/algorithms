"""Residual flow network representation shared by all max-flow solvers."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FlowNetwork:
    """Directed capacitated graph stored as paired forward/backward arcs.

    Arc ``i`` and arc ``i ^ 1`` are mutual reverses, so pushing flow along
    ``i`` is just ``cap[i] -= d; cap[i ^ 1] += d``.
    """

    n: int
    to: list[int] = field(default_factory=list)
    cap: list[int] = field(default_factory=list)
    orig: list[int] = field(default_factory=list)
    adj: list[list[int]] = field(init=False)

    def __post_init__(self) -> None:
        if self.n <= 0:
            raise ValueError("network needs at least one vertex")
        self.adj = [[] for _ in range(self.n)]

    def add_edge(self, u: int, v: int, capacity: int) -> int:
        """Add arc ``u -> v``; returns the index of the forward arc."""
        if not (0 <= u < self.n and 0 <= v < self.n):
            raise ValueError("vertex out of range")
        if capacity < 0:
            raise ValueError("capacity must be non-negative")
        idx = len(self.to)
        self.to += [v, u]
        self.cap += [capacity, 0]
        self.orig += [capacity, 0]
        self.adj[u].append(idx)
        self.adj[v].append(idx + 1)
        return idx

    @property
    def num_edges(self) -> int:
        return len(self.to) // 2

    def source_of(self, arc: int) -> int:
        return self.to[arc ^ 1]

    def reset(self) -> None:
        """Restore all residual capacities to the original ones."""
        self.cap = list(self.orig)

    def copy(self) -> "FlowNetwork":
        g = FlowNetwork(self.n)
        g.to, g.cap, g.orig = list(self.to), list(self.cap), list(self.orig)
        g.adj = [list(a) for a in self.adj]
        return g

    def edges(self) -> list[tuple[int, int, int]]:
        """Forward arcs as ``(u, v, capacity)``."""
        return [(self.to[i ^ 1], self.to[i], self.orig[i]) for i in range(0, len(self.to), 2)]

    def flow_on(self, arc: int) -> int:
        """Flow currently routed over forward arc ``arc``."""
        return self.orig[arc] - self.cap[arc]

    def net_outflow(self, v: int) -> int:
        """Total flow leaving ``v`` minus total flow entering it."""
        total = 0
        for a in self.adj[v]:
            if a % 2 == 0:
                total += self.flow_on(a)
            else:
                total -= self.flow_on(a ^ 1)
        return total

    def check_flow(self, s: int, t: int) -> int:
        """Validate capacity and conservation constraints; return flow value."""
        for a in range(0, len(self.to), 2):
            f = self.flow_on(a)
            if f < 0 or f > self.orig[a]:
                raise AssertionError(f"capacity violated on arc {a}")
        for v in range(self.n):
            if v not in (s, t) and self.net_outflow(v) != 0:
                raise AssertionError(f"conservation violated at {v}")
        return self.net_outflow(s)
