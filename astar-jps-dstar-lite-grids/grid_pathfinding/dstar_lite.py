"""D* Lite (Koenig & Likhachev, 2002) incremental replanning on 8-connected grids.

The search runs backwards from the goal, so the agent's start can move and the
map can change while most of the previous search effort is reused.
"""
from __future__ import annotations

import heapq
from typing import Iterable

from .grid import INF, Grid, Node, octile

Key = tuple[float, float]


class DStarLite:
    """Incremental shortest-path planner over a mutable *known* grid."""

    def __init__(self, grid: Grid, start: Node, goal: Node) -> None:
        if not (grid.is_free(*start) and grid.is_free(*goal)):
            raise ValueError("start and goal must be free cells")
        self.grid = grid
        self.start = start
        self.goal = goal
        self.last = start
        self.km = 0.0
        self.g: dict[Node, float] = {}
        self.rhs: dict[Node, float] = {goal: 0.0}
        self._keys: dict[Node, Key] = {}
        self._heap: list[tuple[Key, Node]] = []
        self.total_expanded = 0
        self._insert(goal, self._calc_key(goal))

    # -- helpers ------------------------------------------------------------
    def _g(self, u: Node) -> float:
        return self.g.get(u, INF)

    def _rhs(self, u: Node) -> float:
        return self.rhs.get(u, INF)

    def _calc_key(self, u: Node) -> Key:
        m = min(self._g(u), self._rhs(u))
        return (m + octile(self.start, u) + self.km, m)

    def _insert(self, u: Node, key: Key) -> None:
        self._keys[u] = key
        heapq.heappush(self._heap, (key, u))

    def _peek(self) -> tuple[Key, Node] | None:
        while self._heap:
            key, u = self._heap[0]
            if self._keys.get(u) == key:
                return key, u
            heapq.heappop(self._heap)
        return None

    # -- core ---------------------------------------------------------------
    def update_vertex(self, u: Node) -> None:
        """Recompute ``rhs(u)`` and fix its queue membership."""
        if u != self.goal:
            if not self.grid.is_free(*u):
                self.rhs[u] = INF
            else:
                self.rhs[u] = min(
                    (c + self._g((nx, ny)) for nx, ny, c in self.grid.neighbors(*u)),
                    default=INF)
        self._keys.pop(u, None)
        if self._g(u) != self._rhs(u):
            self._insert(u, self._calc_key(u))

    def compute_shortest_path(self) -> int:
        """Expand inconsistent vertices until the start is locally consistent.

        Returns the number of vertex expansions performed in this call."""
        expanded = 0
        while True:
            top = self._peek()
            if top is None:
                break
            k_old, u = top
            if not (k_old < self._calc_key(self.start) or self._rhs(self.start) != self._g(self.start)):
                break
            heapq.heappop(self._heap)
            del self._keys[u]
            expanded += 1
            k_new = self._calc_key(u)
            if k_old < k_new:
                self._insert(u, k_new)
            elif self._g(u) > self._rhs(u):
                self.g[u] = self._rhs(u)
                for nb in self.grid.in_bounds_neighbors(*u):
                    self.update_vertex(nb)
            else:
                self.g[u] = INF
                self.update_vertex(u)
                for nb in self.grid.in_bounds_neighbors(*u):
                    self.update_vertex(nb)
        self.total_expanded += expanded
        return expanded

    def cost_to_goal(self) -> float:
        """Current estimate of the start-to-goal cost (``inf`` if unreachable)."""
        return self._g(self.start)

    def update_cells(self, changes: Iterable[tuple[int, int, bool]]) -> None:
        """Apply ``(x, y, blocked)`` edits to the known grid and queue affected vertices."""
        touched: set[Node] = set()
        for x, y, blocked in changes:
            self.grid.set_blocked(x, y, blocked)
            touched.add((x, y))
            touched.update(self.grid.in_bounds_neighbors(x, y))
        if touched:
            self.km += octile(self.last, self.start)
            self.last = self.start
            for u in touched:
                self.update_vertex(u)

    def move_start(self, new_start: Node) -> None:
        """Relocate the agent (the key offset ``km`` is adjusted on the next edit)."""
        if not self.grid.is_free(*new_start):
            raise ValueError("cannot move into an obstacle")
        self.start = new_start

    def next_step(self) -> Node | None:
        """Greedy successor of the start on the current shortest path."""
        if self._g(self.start) == INF:
            return None
        best, best_val = None, INF
        for nx, ny, c in self.grid.neighbors(*self.start):
            val = c + self._g((nx, ny))
            if val < best_val - 1e-12:
                best, best_val = (nx, ny), val
        return best

    def extract_path(self, limit: int | None = None) -> list[Node] | None:
        """Follow greedy descent on ``g`` from the start to the goal."""
        limit = limit if limit is not None else self.grid.width * self.grid.height
        saved = self.start
        path = [saved]
        try:
            while self.start != self.goal:
                nxt = self.next_step()
                if nxt is None or len(path) > limit:
                    return None
                self.start = nxt
                path.append(nxt)
        finally:
            self.start = saved
        return path
