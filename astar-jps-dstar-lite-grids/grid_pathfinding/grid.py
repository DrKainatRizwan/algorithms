"""Occupancy grid, neighbourhood rules, heuristics and map generators.

Coordinates are ``(x, y)`` with ``x`` the column and ``y`` the row.  Movement is
8-connected with unit orthogonal cost and ``sqrt(2)`` diagonal cost.  Diagonal
moves may not cut corners: both orthogonally adjacent cells must be free.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Iterable, Iterator, Sequence

import numpy as np

Node = tuple[int, int]

SQRT2 = math.sqrt(2.0)
INF = math.inf
DIRS8: tuple[tuple[int, int], ...] = (
    (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1),
)


def octile(a: Node, b: Node) -> float:
    """Exact distance on an obstacle-free 8-connected grid (admissible, consistent)."""
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return (dx + dy) + (SQRT2 - 2.0) * min(dx, dy)


def euclidean(a: Node, b: Node) -> float:
    """Straight-line distance; admissible but weaker than octile."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def manhattan(a: Node, b: Node) -> float:
    """Manhattan distance; NOT admissible for 8-connected movement."""
    return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))


def zero(a: Node, b: Node) -> float:
    """Null heuristic (turns A* into Dijkstra)."""
    return 0.0


class Grid:
    """A rectangular occupancy grid with mutable obstacles."""

    def __init__(self, width: int, height: int, blocked: np.ndarray | None = None) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("grid dimensions must be positive")
        self.width = width
        self.height = height
        if blocked is None:
            data = np.zeros((height, width), dtype=bool)
        else:
            data = np.asarray(blocked, dtype=bool)
            if data.shape != (height, width):
                raise ValueError("blocked array shape mismatch")
        self._rows: list[bytearray] = [bytearray(row.astype(np.uint8).tobytes()) for row in data]

    # -- construction -------------------------------------------------------
    @classmethod
    def from_strings(cls, lines: Sequence[str]) -> "Grid":
        """Build a grid from text rows where ``#`` is an obstacle."""
        width = len(lines[0])
        if any(len(line) != width for line in lines):
            raise ValueError("ragged rows")
        arr = np.array([[c == "#" for c in line] for line in lines], dtype=bool)
        return cls(width, len(lines), arr)

    def copy(self) -> "Grid":
        g = Grid(self.width, self.height)
        g._rows = [bytearray(r) for r in self._rows]
        return g

    # -- queries ------------------------------------------------------------
    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def is_free(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height and not self._rows[y][x]

    def set_blocked(self, x: int, y: int, value: bool = True) -> None:
        if not self.in_bounds(x, y):
            raise IndexError((x, y))
        self._rows[y][x] = 1 if value else 0

    def to_array(self) -> np.ndarray:
        """Return a ``(height, width)`` boolean array, True where blocked."""
        return np.array([list(r) for r in self._rows], dtype=bool)

    def free_count(self) -> int:
        return sum(self.width - sum(r) for r in self._rows)

    def free_cells(self) -> list[Node]:
        return [(x, y) for y in range(self.height) for x in range(self.width) if not self._rows[y][x]]

    def in_bounds_neighbors(self, x: int, y: int) -> Iterator[Node]:
        """All in-bounds 8-neighbours regardless of occupancy."""
        for dx, dy in DIRS8:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                yield nx, ny

    def neighbors(self, x: int, y: int) -> list[tuple[int, int, float]]:
        """Traversable successors of ``(x, y)`` with edge costs (no corner cutting)."""
        if not self.is_free(x, y):
            return []
        free = self.is_free
        out: list[tuple[int, int, float]] = []
        for dx, dy in DIRS8:
            nx, ny = x + dx, y + dy
            if not free(nx, ny):
                continue
            if dx and dy:
                if not (free(x + dx, y) and free(x, y + dy)):
                    continue
                out.append((nx, ny, SQRT2))
            else:
                out.append((nx, ny, 1.0))
        return out

    def edge_cost(self, a: Node, b: Node) -> float:
        """Cost of the single move ``a -> b`` (``inf`` if illegal)."""
        dx, dy = b[0] - a[0], b[1] - a[1]
        if max(abs(dx), abs(dy)) != 1 or not (self.is_free(*a) and self.is_free(*b)):
            return INF
        if dx and dy:
            if not (self.is_free(a[0] + dx, a[1]) and self.is_free(a[0], a[1] + dy)):
                return INF
            return SQRT2
        return 1.0

    # -- paths --------------------------------------------------------------
    def path_cost(self, path: Sequence[Node]) -> float:
        """Total cost of a unit-step path; ``inf`` if any step is illegal."""
        return sum(self.edge_cost(a, b) for a, b in zip(path, path[1:]))

    def is_valid_path(self, path: Sequence[Node]) -> bool:
        return len(path) > 0 and self.is_free(*path[0]) and self.path_cost(path) < INF

    def reachable(self, start: Node) -> dict[Node, int]:
        """Breadth-first step counts from ``start`` over legal moves."""
        dist = {start: 0}
        queue = deque([start])
        while queue:
            x, y = queue.popleft()
            for nx, ny, _ in self.neighbors(x, y):
                if (nx, ny) not in dist:
                    dist[(nx, ny)] = dist[(x, y)] + 1
                    queue.append((nx, ny))
        return dist

    def render(self, path: Iterable[Node] = ()) -> str:
        marks = set(path)
        lines = []
        for y in range(self.height):
            lines.append("".join(
                "#" if self._rows[y][x] else ("*" if (x, y) in marks else ".")
                for x in range(self.width)))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Map generators (all deterministic given the seed)
# ---------------------------------------------------------------------------
def random_obstacles(width: int, height: int, density: float, seed: int) -> Grid:
    """Independent Bernoulli obstacles with the given density."""
    rng = np.random.default_rng(seed)
    return Grid(width, height, rng.random((height, width)) < density)


def cave_map(width: int, height: int, seed: int, fill: float = 0.45, steps: int = 4) -> Grid:
    """Cellular-automaton caves: a cell becomes wall if >= 5 of its 3x3 block are walls."""
    rng = np.random.default_rng(seed)
    wall = rng.random((height, width)) < fill
    for _ in range(steps):
        padded = np.pad(wall, 1, constant_values=True).astype(np.int32)
        count = sum(padded[1 + dy: 1 + dy + height, 1 + dx: 1 + dx + width]
                    for dy in (-1, 0, 1) for dx in (-1, 0, 1))
        wall = count >= 5
    return Grid(width, height, wall)


def room_map(width: int, height: int, seed: int, rooms: int = 18) -> Grid:
    """Rectangular rooms joined by one-cell corridors (always connected)."""
    rng = np.random.default_rng(seed)
    wall = np.ones((height, width), dtype=bool)
    centres: list[Node] = []
    for _ in range(rooms):
        w = int(rng.integers(5, max(6, width // 6)))
        h = int(rng.integers(5, max(6, height // 6)))
        x0 = int(rng.integers(1, max(2, width - w - 1)))
        y0 = int(rng.integers(1, max(2, height - h - 1)))
        wall[y0:y0 + h, x0:x0 + w] = False
        centres.append((min(width - 2, x0 + w // 2), min(height - 2, y0 + h // 2)))
    for (ax, ay), (bx, by) in zip(centres, centres[1:]):
        wall[ay, min(ax, bx):max(ax, bx) + 1] = False
        wall[min(ay, by):max(ay, by) + 1, bx] = False
    return Grid(width, height, wall)


def maze_map(width: int, height: int, seed: int, braid: float = 0.05) -> Grid:
    """Perfect maze (recursive backtracker) with a fraction ``braid`` of walls knocked out."""
    width |= 1
    height |= 1
    rng = np.random.default_rng(seed)
    wall = np.ones((height, width), dtype=bool)
    wall[1, 1] = False
    stack = [(1, 1)]
    while stack:
        x, y = stack[-1]
        options = [(x + dx, y + dy, dx, dy) for dx, dy in ((2, 0), (-2, 0), (0, 2), (0, -2))
                   if 0 < x + dx < width - 1 and 0 < y + dy < height - 1 and wall[y + dy, x + dx]]
        if not options:
            stack.pop()
            continue
        nx, ny, dx, dy = options[int(rng.integers(len(options)))]
        wall[y + dy // 2, x + dx // 2] = False
        wall[ny, nx] = False
        stack.append((nx, ny))
    if braid > 0:
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if wall[y, x] and (x + y) % 2 == 1 and rng.random() < braid:
                    wall[y, x] = False
    return Grid(width, height, wall)


def random_query(grid: Grid, rng: np.random.Generator, far_fraction: float = 0.7) -> tuple[Node, Node]:
    """Pick a connected start/goal pair; the goal is at least ``far_fraction`` of the max BFS depth away."""
    free = grid.free_cells()
    if len(free) < 2:
        raise ValueError("grid has fewer than two free cells")
    for _ in range(50):
        start = free[int(rng.integers(len(free)))]
        dist = grid.reachable(start)
        if len(dist) < 2:
            continue
        far = max(dist.values())
        cands = sorted(c for c, d in dist.items() if d >= far_fraction * far and c != start)
        if cands:
            return start, cands[int(rng.integers(len(cands)))]
    raise ValueError("could not find a connected pair")
