"""Jump Point Search (Harabor & Grastien) for 8-connected grids without corner cutting.

JPS prunes symmetric paths by *jumping* along rows, columns and diagonals until it
finds the goal or a cell with a forced neighbour; only those jump points enter the
open list.  Optimality is preserved on uniform-cost grids.
"""
from __future__ import annotations

import heapq

from .astar import SearchResult, reconstruct
from .grid import INF, Grid, Node, octile


def _sign(v: int) -> int:
    return (v > 0) - (v < 0)


class JumpPointSearch:
    """Reusable JPS planner bound to a grid."""

    def __init__(self, grid: Grid) -> None:
        self.grid = grid
        self.scanned = 0  # cells touched by jump scans in the last search

    # -- jump primitives ----------------------------------------------------
    def _straight(self, x: int, y: int, dx: int, dy: int, goal: Node) -> Node | None:
        free = self.grid.is_free
        while True:
            x += dx
            y += dy
            if not free(x, y):
                return None
            self.scanned += 1
            if (x, y) == goal:
                return (x, y)
            if dx:
                if (free(x, y + 1) and not free(x - dx, y + 1)) or \
                   (free(x, y - 1) and not free(x - dx, y - 1)):
                    return (x, y)
            else:
                if (free(x + 1, y) and not free(x + 1, y - dy)) or \
                   (free(x - 1, y) and not free(x - 1, y - dy)):
                    return (x, y)

    def _jump(self, x: int, y: int, dx: int, dy: int, goal: Node) -> Node | None:
        """Return the next jump point from ``(x, y)`` in direction ``(dx, dy)``."""
        if not (dx and dy):
            return self._straight(x, y, dx, dy, goal)
        free = self.grid.is_free
        while True:
            if not (free(x + dx, y) and free(x, y + dy) and free(x + dx, y + dy)):
                return None
            x += dx
            y += dy
            self.scanned += 1
            if (x, y) == goal:
                return (x, y)
            if self._straight(x, y, dx, 0, goal) is not None or \
               self._straight(x, y, 0, dy, goal) is not None:
                return (x, y)

    def _directions(self, node: Node, parent: Node | None) -> list[tuple[int, int]]:
        """Pruned set of search directions given the direction of arrival."""
        x, y = node
        if parent is None:
            return [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        dx, dy = _sign(x - parent[0]), _sign(y - parent[1])
        free = self.grid.is_free
        if dx and dy:
            return [(dx, dy), (dx, 0), (0, dy)]
        dirs = [(dx, dy)]
        if dx:
            for s in (1, -1):
                if free(x, y + s) and not free(x - dx, y + s):
                    dirs += [(0, s), (dx, s)]
        else:
            for s in (1, -1):
                if free(x + s, y) and not free(x + s, y - dy):
                    dirs += [(s, 0), (s, dy)]
        return dirs

    # -- search -------------------------------------------------------------
    def search(self, start: Node, goal: Node) -> SearchResult:
        """Find an optimal path; the returned path is fully expanded to unit steps."""
        self.scanned = 0
        grid = self.grid
        if not (grid.is_free(*start) and grid.is_free(*goal)):
            return SearchResult(None, INF, 0)
        if start == goal:
            return SearchResult([start], 0.0, 0)
        g = {start: 0.0}
        parent: dict[Node, Node | None] = {start: None}
        h0 = octile(start, goal)
        heap = [(h0, h0, start)]
        closed: set[Node] = set()
        expanded = 0
        while heap:
            _, _, node = heapq.heappop(heap)
            if node in closed:
                continue
            if node == goal:
                return SearchResult(self.expand(reconstruct(parent, node)), g[node],
                                    expanded, len(g))
            closed.add(node)
            expanded += 1
            for dx, dy in self._directions(node, parent[node]):
                jp = self._jump(node[0], node[1], dx, dy, goal)
                if jp is None or jp in closed:
                    continue
                ng = g[node] + octile(node, jp)
                if ng < g.get(jp, INF):
                    g[jp] = ng
                    parent[jp] = node
                    h = octile(jp, goal)
                    heapq.heappush(heap, (ng + h, h, jp))
        return SearchResult(None, INF, expanded, len(g))

    @staticmethod
    def expand(jump_points: list[Node]) -> list[Node]:
        """Interpolate unit steps between consecutive jump points."""
        if not jump_points:
            return []
        out = [jump_points[0]]
        for (ax, ay), (bx, by) in zip(jump_points, jump_points[1:]):
            dx, dy = _sign(bx - ax), _sign(by - ay)
            x, y = ax, ay
            while (x, y) != (bx, by):
                x += dx
                y += dy
                out.append((x, y))
        return out


def jps(grid: Grid, start: Node, goal: Node) -> SearchResult:
    """Convenience wrapper around :class:`JumpPointSearch`."""
    return JumpPointSearch(grid).search(start, goal)
