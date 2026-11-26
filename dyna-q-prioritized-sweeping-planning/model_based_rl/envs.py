"""Tabular gridworld environments with an exportable exact model."""

from __future__ import annotations

import numpy as np

# action index -> (d_row, d_col): up, right, down, left
MOVES = ((-1, 0), (0, 1), (1, 0), (0, -1))


class GridWorld:
    """Rectangular gridworld with walls, a start cell and an absorbing goal.

    Reward is 1 on entering the goal and 0 otherwise. With probability
    ``slip`` the chosen action is replaced by a uniformly random one.
    Layout characters: ``#`` wall, ``S`` start, ``G`` goal, ``.`` free.
    """

    n_actions = 4

    def __init__(self, layout: list[str], slip: float = 0.0) -> None:
        if not 0.0 <= slip < 1.0:
            raise ValueError("slip must be in [0, 1)")
        self.slip = slip
        self.rows = len(layout)
        self.cols = len(layout[0])
        if any(len(r) != self.cols for r in layout):
            raise ValueError("ragged layout")
        self.start = self.goal = -1
        self.walls = np.zeros((self.rows, self.cols), dtype=bool)
        self._parse(layout)
        self.state = self.start

    def _parse(self, layout: list[str]) -> None:
        start = goal = None
        for r, line in enumerate(layout):
            for c, ch in enumerate(line):
                if ch == "#":
                    self.walls[r, c] = True
                elif ch == "S":
                    start = r * self.cols + c
                elif ch == "G":
                    goal = r * self.cols + c
        if start is None or goal is None:
            raise ValueError("layout needs S and G")
        self.walls[:] = False
        for r, line in enumerate(layout):
            for c, ch in enumerate(line):
                self.walls[r, c] = ch == "#"
        self.start, self.goal = start, goal

    @property
    def n_states(self) -> int:
        return self.rows * self.cols

    def set_layout(self, layout: list[str]) -> None:
        """Replace the maze in place (used for non-stationary experiments)."""
        if len(layout) != self.rows or len(layout[0]) != self.cols:
            raise ValueError("layout shape mismatch")
        self._parse(layout)

    def _move(self, s: int, a: int) -> int:
        r, c = divmod(s, self.cols)
        dr, dc = MOVES[a]
        nr, nc = r + dr, c + dc
        if 0 <= nr < self.rows and 0 <= nc < self.cols and not self.walls[nr, nc]:
            return nr * self.cols + nc
        return s

    def reset(self) -> int:
        self.state = self.start
        return self.state

    def step(self, a: int, rng: np.random.Generator) -> tuple[int, float, bool]:
        """Apply action ``a``; returns ``(next_state, reward, done)``."""
        if self.slip > 0 and rng.random() < self.slip:
            a = int(rng.integers(self.n_actions))
        s2 = self._move(self.state, a)
        done = s2 == self.goal
        self.state = s2
        return s2, 1.0 if done else 0.0, done

    def transition_model(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Exact ``P[s,a,s']``, expected reward ``R[s,a]`` and terminal mask.

        The goal is absorbing with zero further reward.
        """
        S, A = self.n_states, self.n_actions
        P = np.zeros((S, A, S))
        R = np.zeros((S, A))
        terminal = np.zeros(S, dtype=bool)
        terminal[self.goal] = True
        for s in range(S):
            r, c = divmod(s, self.cols)
            if self.walls[r, c]:
                P[s, :, s] = 1.0
                continue
            if s == self.goal:
                P[s, :, s] = 1.0
                continue
            for a in range(A):
                for b in range(A):
                    p = (1 - self.slip) * (a == b) + self.slip / A
                    if p == 0:
                        continue
                    s2 = self._move(s, b)
                    P[s, a, s2] += p
                    if s2 == self.goal:
                        R[s, a] += p
        return P, R, terminal

    def render(self, policy: np.ndarray | None = None) -> str:
        """ASCII picture; optionally overlay a greedy policy as arrows."""
        arrows = "^>v<"
        out = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                s = r * self.cols + c
                if self.walls[r, c]:
                    row.append("#")
                elif s == self.goal:
                    row.append("G")
                elif policy is not None:
                    row.append(arrows[int(policy[s])])
                elif s == self.start:
                    row.append("S")
                else:
                    row.append(".")
            out.append("".join(row))
        return "\n".join(out)


def blocking_maze_layouts() -> tuple[list[str], list[str]]:
    """Sutton & Barto's blocking maze: the gap in the wall moves left."""
    before = ["........G", ".........", ".........", "########.", ".........", "...S....."]
    after = ["........G", ".........", ".........", ".########", ".........", "...S....."]
    return before, after


def shortcut_maze_layouts() -> tuple[list[str], list[str]]:
    """Shortcut maze: a second, shorter gap opens on the right."""
    before = ["........G", ".........", ".........", ".########", ".........", "...S....."]
    after = ["........G", ".........", ".........", ".#######.", ".........", "...S....."]
    return before, after


def dyna_maze() -> list[str]:
    """The 6x9 maze from the Dyna chapter of Sutton & Barto."""
    return [
        "........G",
        "..#......",
        "S.#......",
        "..#...#..",
        "......#..",
        "......#..",
    ]
