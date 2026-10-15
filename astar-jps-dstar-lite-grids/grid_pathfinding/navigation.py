"""Navigation in partially known terrain: D* Lite vs. replanning A* from scratch.

The agent starts with a belief map (by default empty: the free-space assumption),
senses true obstacles within a square radius, and replans whenever it learns that
the world differs from its belief.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .astar import astar
from .dstar_lite import DStarLite
from .grid import INF, Grid, Node


@dataclass
class NavigationResult:
    """Summary of one navigation episode."""

    success: bool
    trajectory: list[Node]
    travelled: float
    replans: int
    expansions: int
    discovered: int = 0
    log: list[str] = field(default_factory=list)


def sense(true_grid: Grid, known: Grid, pos: Node, radius: int) -> list[tuple[int, int, bool]]:
    """Cells within ``radius`` (Chebyshev) whose true state differs from the belief."""
    changes = []
    for y in range(max(0, pos[1] - radius), min(true_grid.height, pos[1] + radius + 1)):
        for x in range(max(0, pos[0] - radius), min(true_grid.width, pos[0] + radius + 1)):
            truth = not true_grid.is_free(x, y)
            if truth != (not known.is_free(x, y)):
                changes.append((x, y, truth))
    return changes


def navigate_dstar(true_grid: Grid, start: Node, goal: Node, radius: int = 3,
                   known: Grid | None = None, max_steps: int | None = None) -> NavigationResult:
    """Drive an agent to ``goal`` with incremental D* Lite replanning."""
    belief = known.copy() if known is not None else Grid(true_grid.width, true_grid.height)
    max_steps = max_steps or 4 * true_grid.width * true_grid.height
    pos = start
    for x, y, b in sense(true_grid, belief, pos, radius):
        belief.set_blocked(x, y, b)
    planner = DStarLite(belief, pos, goal)
    planner.compute_shortest_path()
    traj, cost, replans, discovered = [pos], 0.0, 0, 0
    while pos != goal and len(traj) <= max_steps:
        if planner.cost_to_goal() == INF:
            return NavigationResult(False, traj, cost, replans, planner.total_expanded, discovered)
        nxt = planner.next_step()
        if nxt is None:
            break
        cost += true_grid.edge_cost(pos, nxt) if true_grid.edge_cost(pos, nxt) < INF else 0.0
        pos = nxt
        planner.move_start(pos)
        traj.append(pos)
        changes = sense(true_grid, planner.grid, pos, radius)
        if changes:
            discovered += len(changes)
            planner.update_cells(changes)
            planner.compute_shortest_path()
            replans += 1
    return NavigationResult(pos == goal, traj, cost, replans, planner.total_expanded, discovered)


def navigate_replan(true_grid: Grid, start: Node, goal: Node, radius: int = 3,
                    known: Grid | None = None, max_steps: int | None = None,
                    planner: Callable = astar) -> NavigationResult:
    """Same episode, but replan from scratch with ``planner`` when the plan is invalidated."""
    belief = known.copy() if known is not None else Grid(true_grid.width, true_grid.height)
    max_steps = max_steps or 4 * true_grid.width * true_grid.height
    pos = start
    for x, y, b in sense(true_grid, belief, pos, radius):
        belief.set_blocked(x, y, b)
    res = planner(belief, pos, goal)
    expansions, replans, discovered = res.expanded, 0, 0
    path = res.path
    traj, cost, i = [pos], 0.0, 0
    while pos != goal and len(traj) <= max_steps:
        if path is None:
            return NavigationResult(False, traj, cost, replans, expansions, discovered)
        nxt = path[i + 1]
        cost += true_grid.edge_cost(pos, nxt)
        pos = nxt
        i += 1
        traj.append(pos)
        changes = sense(true_grid, belief, pos, radius)
        if changes:
            discovered += len(changes)
            for x, y, b in changes:
                belief.set_blocked(x, y, b)
            if not belief.is_valid_path(path[i:]):
                res = planner(belief, pos, goal)
                expansions += res.expanded
                replans += 1
                path, i = res.path, 0
    return NavigationResult(pos == goal, traj, cost, replans, expansions, discovered)
