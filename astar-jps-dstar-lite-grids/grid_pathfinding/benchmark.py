"""Benchmark harness comparing Dijkstra, A*, weighted A* and JPS (and replanning)."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .astar import SearchResult, astar, dijkstra
from .grid import Grid, cave_map, maze_map, random_obstacles, random_query, room_map
from .jps import JumpPointSearch
from .navigation import navigate_dstar, navigate_replan

MAP_FACTORIES: dict[str, Callable[[int, int, int], Grid]] = {
    "random20": lambda w, h, s: random_obstacles(w, h, 0.20, s),
    "caves": lambda w, h, s: cave_map(w, h, s),
    "rooms": lambda w, h, s: room_map(w, h, s),
    "maze": lambda w, h, s: maze_map(w, h, s, braid=0.05),
}


@dataclass
class PlannerStats:
    name: str
    mean_ms: float
    mean_expanded: float
    mean_cost: float


def planners(grid: Grid) -> dict[str, Callable[[tuple[int, int], tuple[int, int]], SearchResult]]:
    jp = JumpPointSearch(grid)
    return {
        "dijkstra": lambda s, t: dijkstra(grid, s, t),
        "astar": lambda s, t: astar(grid, s, t),
        "wastar(2.0)": lambda s, t: astar(grid, s, t, weight=2.0),
        "jps": lambda s, t: jp.search(s, t),
    }


def compare_static(kind: str, size: int, queries: int, seed: int) -> list[PlannerStats]:
    """Run every static planner on the same random queries of one map type."""
    grid = MAP_FACTORIES[kind](size, size, seed)
    rng = np.random.default_rng(seed + 1)
    pairs = [random_query(grid, rng) for _ in range(queries)]
    out = []
    for name, fn in planners(grid).items():
        t_total, exp, cost = 0.0, 0, 0.0
        for s, t in pairs:
            t0 = time.perf_counter()
            r = fn(s, t)
            t_total += time.perf_counter() - t0
            exp += r.expanded
            cost += r.cost
        out.append(PlannerStats(name, 1000 * t_total / queries, exp / queries, cost / queries))
    return out


def compare_navigation(size: int, episodes: int, seed: int, radius: int = 4) -> dict[str, float]:
    """Average work of D* Lite vs. scratch A* replanning in unknown terrain."""
    tot = {"dstar_exp": 0.0, "astar_exp": 0.0, "dstar_cost": 0.0, "astar_cost": 0.0,
           "dstar_replans": 0.0, "astar_replans": 0.0, "done": 0.0}
    for ep in range(episodes):
        grid = random_obstacles(size, size, 0.25, seed + ep)
        rng = np.random.default_rng(seed + 100 + ep)
        s, t = random_query(grid, rng, 0.8)
        a = navigate_dstar(grid, s, t, radius)
        b = navigate_replan(grid, s, t, radius)
        if not (a.success and b.success):
            continue
        tot["done"] += 1
        tot["dstar_exp"] += a.expansions
        tot["astar_exp"] += b.expansions
        tot["dstar_cost"] += a.travelled
        tot["astar_cost"] += b.travelled
        tot["dstar_replans"] += a.replans
        tot["astar_replans"] += b.replans
    n = max(1.0, tot["done"])
    return {k: (v if k == "done" else v / n) for k, v in tot.items()}
