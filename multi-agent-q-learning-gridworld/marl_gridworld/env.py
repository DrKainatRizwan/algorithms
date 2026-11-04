"""Cooperative gridworld with simultaneous moves and collision resolution.

Layout characters: ``#`` wall, ``.`` free, ``0``-``9`` start of agent i,
``a``-``j`` goal of agent i (``a`` is agent 0's goal, and so on).
All agents share one team reward. An agent that reaches its goal leaves the
board, so it no longer blocks anyone.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# up, down, left, right, stay
ACTIONS: tuple[tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1), (0, 0))
Pos = tuple[int, int]


def parse_layout(rows: list[str]) -> tuple[np.ndarray, list[Pos], list[Pos]]:
    """Parse ASCII rows into (wall mask, starts, goals)."""
    h, w = len(rows), len(rows[0])
    if any(len(r) != w for r in rows):
        raise ValueError("all layout rows must have equal length")
    walls = np.zeros((h, w), dtype=bool)
    starts: dict[int, Pos] = {}
    goals: dict[int, Pos] = {}
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch == "#":
                walls[r, c] = True
            elif ch.isdigit():
                starts[int(ch)] = (r, c)
            elif "a" <= ch <= "j":
                goals[ord(ch) - ord("a")] = (r, c)
            elif ch != ".":
                raise ValueError(f"bad layout character {ch!r}")
    n = len(starts)
    if n == 0 or sorted(starts) != list(range(n)) or sorted(goals) != list(range(n)):
        raise ValueError("layout needs starts 0..n-1 and matching goals a..")
    return walls, [starts[i] for i in range(n)], [goals[i] for i in range(n)]


@dataclass
class CoopGridWorld:
    """Fully cooperative multi-agent navigation task.

    Reward per step: ``step_cost`` each active agent, ``goal_reward`` for each
    arrival, ``team_bonus`` when the last agent arrives, ``conflict_penalty``
    per blocked agent pair.
    """

    walls: np.ndarray
    starts: list[Pos]
    goals: list[Pos]
    max_steps: int = 60
    step_cost: float = -0.05
    goal_reward: float = 1.0
    team_bonus: float = 3.0
    conflict_penalty: float = -0.3
    pos: list[Pos] = field(default_factory=list, init=False)
    active: list[bool] = field(default_factory=list, init=False)
    t: int = field(default=0, init=False)
    collisions: int = field(default=0, init=False)

    @property
    def n_agents(self) -> int:
        return len(self.starts)

    @property
    def n_actions(self) -> int:
        return len(ACTIONS)

    def reset(self) -> list[Pos]:
        self.pos = list(self.starts)
        self.active = [True] * self.n_agents
        self.t = 0
        self.collisions = 0
        return list(self.pos)

    def _free(self, p: Pos) -> bool:
        h, w = self.walls.shape
        return 0 <= p[0] < h and 0 <= p[1] < w and not self.walls[p]

    def resolve(self, actions: list[int]) -> tuple[list[Pos], int]:
        """Return new positions and the number of agents blocked by others."""
        target: dict[int, Pos] = {}
        for i, a in enumerate(actions):
            if not self.active[i]:
                continue
            dr, dc = ACTIONS[a]
            p = (self.pos[i][0] + dr, self.pos[i][1] + dc)
            target[i] = p if self._free(p) else self.pos[i]
        blocked: set[int] = set()
        changed = True
        while changed:  # iterate to a fixed point: blocked agents act as walls
            changed = False
            final = {i: (self.pos[i] if i in blocked else target[i]) for i in target}
            for i in target:
                if i in blocked:
                    continue
                for j in target:
                    if j == i:
                        continue
                    same_cell = final[i] == final[j]
                    swap = final[i] == self.pos[j] and final[j] == self.pos[i] \
                        and self.pos[i] != self.pos[j]
                    if same_cell or swap:
                        blocked.add(i)
                        changed = True
                        break
        new = list(self.pos)
        for i in target:
            new[i] = self.pos[i] if i in blocked else target[i]
        blocked_moves = {i for i in blocked if target[i] != self.pos[i]}
        return new, len(blocked_moves)

    def step(self, actions: list[int]) -> tuple[list[Pos], float, bool, dict]:
        """Apply joint action; returns (positions, team reward, done, info)."""
        if len(actions) != self.n_agents:
            raise ValueError("one action per agent required")
        n_active = sum(self.active)
        new, nblocked = self.resolve(actions)
        reward = self.step_cost * n_active
        reward += self.conflict_penalty * nblocked
        self.collisions += nblocked
        arrived = []
        for i in range(self.n_agents):
            if self.active[i] and new[i] == self.goals[i]:
                self.active[i] = False
                arrived.append(i)
                reward += self.goal_reward
        self.pos = new
        self.t += 1
        success = not any(self.active)
        if success:
            reward += self.team_bonus
        done = success or self.t >= self.max_steps
        return list(self.pos), reward, done, {
            "arrived": arrived, "success": success, "blocked": nblocked}

    def observe(self, i: int, mode: str = "local") -> tuple:
        """Observation for agent ``i``.

        ``local``: own cell plus a bitmask of which 4 neighbours hold another
        agent. ``own``: own cell only. ``global``: every agent's cell.
        """
        if mode == "global":
            return tuple(p if a else (-1, -1) for p, a in zip(self.pos, self.active))
        if mode == "own":
            return (self.pos[i],)
        if mode != "local":
            raise ValueError(f"unknown observation mode {mode!r}")
        r, c = self.pos[i]
        occupied = {p for j, p in enumerate(self.pos) if j != i and self.active[j]}
        bits = tuple(int((r + dr, c + dc) in occupied) for dr, dc in ACTIONS[:4])
        return (self.pos[i], bits)
