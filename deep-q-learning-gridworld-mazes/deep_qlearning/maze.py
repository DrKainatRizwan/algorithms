"""Maze generation and a small gridworld environment."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

# action order: up, right, down, left
ACTIONS: tuple[tuple[int, int], ...] = ((-1, 0), (0, 1), (1, 0), (0, -1))


@dataclass(frozen=True)
class Maze:
    """Immutable grid: ``walls[r, c]`` is True for blocked cells."""

    walls: np.ndarray
    start: tuple[int, int]
    goal: tuple[int, int]

    @property
    def shape(self) -> tuple[int, int]:
        return self.walls.shape  # type: ignore[return-value]

    def is_free(self, r: int, c: int) -> bool:
        h, w = self.shape
        return 0 <= r < h and 0 <= c < w and not self.walls[r, c]

    def render(self, path: list[tuple[int, int]] | None = None) -> str:
        on_path = set(path or [])
        rows = []
        for r in range(self.shape[0]):
            line = []
            for c in range(self.shape[1]):
                if (r, c) == self.start:
                    line.append("S")
                elif (r, c) == self.goal:
                    line.append("G")
                elif self.walls[r, c]:
                    line.append("#")
                elif (r, c) in on_path:
                    line.append("*")
                else:
                    line.append(".")
            rows.append("".join(line))
        return "\n".join(rows)


def generate_maze(cells: int, seed: int = 0, extra_openings: int = 0) -> Maze:
    """Perfect maze by randomized depth-first search on a ``cells x cells`` lattice.

    The resulting grid has side ``2 * cells + 1``. ``extra_openings`` knocks out
    additional interior walls to create loops (multiple routes).
    """
    if cells < 2:
        raise ValueError("cells must be >= 2")
    rng = np.random.default_rng(seed)
    n = 2 * cells + 1
    walls = np.ones((n, n), dtype=bool)
    visited = np.zeros((cells, cells), dtype=bool)
    stack = [(0, 0)]
    visited[0, 0] = True
    walls[1, 1] = False
    while stack:
        r, c = stack[-1]
        nbrs = [(r + dr, c + dc) for dr, dc in ACTIONS
                if 0 <= r + dr < cells and 0 <= c + dc < cells and not visited[r + dr, c + dc]]
        if not nbrs:
            stack.pop()
            continue
        nr, nc = nbrs[int(rng.integers(len(nbrs)))]
        visited[nr, nc] = True
        walls[2 * nr + 1, 2 * nc + 1] = False
        walls[r + nr + 1, c + nc + 1] = False  # wall between the two cells
        stack.append((nr, nc))
    candidates = [(r, c) for r in range(1, n - 1) for c in range(1, n - 1)
                  if walls[r, c] and (r + c) % 2 == 1]
    rng.shuffle(candidates)
    for r, c in candidates[:extra_openings]:
        walls[r, c] = False
    return Maze(walls, (1, 1), (n - 2, n - 2))


def shortest_path_length(maze: Maze) -> int:
    """BFS distance from start to goal, or -1 if unreachable."""
    dist = {maze.start: 0}
    q = deque([maze.start])
    while q:
        r, c = q.popleft()
        if (r, c) == maze.goal:
            return dist[(r, c)]
        for dr, dc in ACTIONS:
            nxt = (r + dr, c + dc)
            if maze.is_free(*nxt) and nxt not in dist:
                dist[nxt] = dist[(r, c)] + 1
                q.append(nxt)
    return -1


class MazeEnv:
    """Gym-like environment. Observation is a one-hot vector over grid cells.

    Rewards: +1 at the goal, -0.01 per step, -0.05 for bumping a wall.
    """

    GOAL_REWARD = 1.0
    STEP_REWARD = -0.01
    WALL_REWARD = -0.05

    def __init__(self, maze: Maze, max_steps: int | None = None) -> None:
        self.maze = maze
        h, w = maze.shape
        self.obs_dim = h * w
        self.n_actions = len(ACTIONS)
        self.max_steps = max_steps or 4 * h * w
        self.pos = maze.start
        self.t = 0

    def encode(self, pos: tuple[int, int]) -> np.ndarray:
        v = np.zeros(self.obs_dim, dtype=np.float64)
        v[pos[0] * self.maze.shape[1] + pos[1]] = 1.0
        return v

    def reset(self) -> np.ndarray:
        self.pos = self.maze.start
        self.t = 0
        return self.encode(self.pos)

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        if not 0 <= action < self.n_actions:
            raise ValueError(f"invalid action {action}")
        dr, dc = ACTIONS[action]
        nxt = (self.pos[0] + dr, self.pos[1] + dc)
        self.t += 1
        if self.maze.is_free(*nxt):
            self.pos = nxt
            reward = self.STEP_REWARD
        else:
            reward = self.WALL_REWARD
        reached = self.pos == self.maze.goal
        if reached:
            reward = self.GOAL_REWARD
        done = reached or self.t >= self.max_steps
        return self.encode(self.pos), reward, done
