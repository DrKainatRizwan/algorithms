"""A* / weighted A* / Dijkstra on 8-connected grids."""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Callable

from .grid import INF, Grid, Node, octile, zero


@dataclass
class SearchResult:
    """Outcome of a single-shot planner."""

    path: list[Node] | None
    cost: float
    expanded: int
    generated: int = 0

    @property
    def found(self) -> bool:
        return self.path is not None


def reconstruct(parent: dict[Node, Node | None], node: Node) -> list[Node]:
    """Follow parent pointers back to the root and return the forward path."""
    out = [node]
    while parent[node] is not None:
        node = parent[node]  # type: ignore[assignment]
        out.append(node)
    out.reverse()
    return out


def astar(
    grid: Grid,
    start: Node,
    goal: Node,
    weight: float = 1.0,
    heuristic: Callable[[Node, Node], float] = octile,
) -> SearchResult:
    """Weighted A*.  ``weight=1`` with an admissible heuristic is optimal;
    larger weights give solutions at most ``weight`` times optimal."""
    if weight < 1.0:
        raise ValueError("weight must be >= 1")
    if not (grid.is_free(*start) and grid.is_free(*goal)):
        return SearchResult(None, INF, 0)
    g = {start: 0.0}
    parent: dict[Node, Node | None] = {start: None}
    h0 = heuristic(start, goal)
    heap: list[tuple[float, float, Node]] = [(weight * h0, h0, start)]
    closed: set[Node] = set()
    expanded = 0
    while heap:
        _, _, node = heapq.heappop(heap)
        if node in closed:
            continue
        if node == goal:
            return SearchResult(reconstruct(parent, node), g[node], expanded, len(g))
        closed.add(node)
        expanded += 1
        gn = g[node]
        for nx, ny, c in grid.neighbors(*node):
            nb = (nx, ny)
            if nb in closed:
                continue
            ng = gn + c
            if ng < g.get(nb, INF):
                g[nb] = ng
                parent[nb] = node
                h = heuristic(nb, goal)
                heapq.heappush(heap, (ng + weight * h, h, nb))
    return SearchResult(None, INF, expanded, len(g))


def dijkstra(grid: Grid, start: Node, goal: Node) -> SearchResult:
    """Uninformed uniform-cost search (A* with a zero heuristic)."""
    return astar(grid, start, goal, 1.0, zero)
