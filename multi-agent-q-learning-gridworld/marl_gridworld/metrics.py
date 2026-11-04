"""Helper statistics and shortest-path lower bounds."""
from __future__ import annotations

from collections import deque

import numpy as np

from .env import ACTIONS, CoopGridWorld


def moving_average(x: list[float] | np.ndarray, window: int) -> np.ndarray:
    """Trailing moving average (shorter windows at the start)."""
    a = np.asarray(x, dtype=float)
    if window < 1:
        raise ValueError("window must be >= 1")
    c = np.cumsum(np.insert(a, 0, 0.0))
    idx = np.arange(1, len(a) + 1)
    lo = np.maximum(0, idx - window)
    return (c[idx] - c[lo]) / (idx - lo)


def bfs_distances(walls: np.ndarray, source: tuple[int, int]) -> np.ndarray:
    """Grid shortest-path distances from ``source`` (-1 if unreachable)."""
    dist = np.full(walls.shape, -1, dtype=int)
    dist[source] = 0
    q = deque([source])
    while q:
        r, c = q.popleft()
        for dr, dc in ACTIONS[:4]:
            n = (r + dr, c + dc)
            if (0 <= n[0] < walls.shape[0] and 0 <= n[1] < walls.shape[1]
                    and not walls[n] and dist[n] < 0):
                dist[n] = dist[r, c] + 1
                q.append(n)
    return dist


def optimal_makespan(env: CoopGridWorld) -> int:
    """Lower bound on steps to finish: the longest individual shortest path."""
    best = 0
    for s, g in zip(env.starts, env.goals):
        d = int(bfs_distances(env.walls, s)[g])
        if d < 0:
            raise ValueError("goal unreachable")
        best = max(best, d)
    return best
