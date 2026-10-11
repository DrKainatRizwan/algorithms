"""Constructive heuristics and deterministic local search (2-opt, Or-opt)."""

from __future__ import annotations

import numpy as np

from .instances import Instance


def nearest_neighbour(inst: Instance, start: int = 0) -> np.ndarray:
    """Greedy nearest-neighbour tour, O(n^2)."""
    n = inst.n
    visited = np.zeros(n, dtype=bool)
    tour = [start]
    visited[start] = True
    for _ in range(n - 1):
        row = np.where(visited, np.inf, inst.dist[tour[-1]])
        nxt = int(np.argmin(row))
        tour.append(nxt)
        visited[nxt] = True
    return np.array(tour)


def two_opt_delta(inst: Instance, tour: np.ndarray, i: int, j: int) -> float:
    """Length change from reversing tour[i..j] (0 <= i < j < n)."""
    n = len(tour)
    d = inst.dist
    a, b = tour[i - 1], tour[i]
    c, e = tour[j], tour[(j + 1) % n]
    if a == e:  # reversing everything but the wrap edge changes nothing
        return 0.0
    return float(d[a, c] + d[b, e] - d[a, b] - d[c, e])


def two_opt(inst: Instance, tour: np.ndarray, max_passes: int = 100) -> np.ndarray:
    """First-improvement 2-opt until a local optimum (or max_passes)."""
    t = np.array(tour)
    n = len(t)
    for _ in range(max_passes):
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                if two_opt_delta(inst, t, i, j) < -1e-12:
                    t[i:j + 1] = t[i:j + 1][::-1]
                    improved = True
        if not improved:
            break
    return t


def or_opt(inst: Instance, tour: np.ndarray, max_seg: int = 3, max_passes: int = 50) -> np.ndarray:
    """Move segments of up to max_seg cities to a better position (either orientation)."""
    d = inst.dist
    t = list(tour)
    n = len(t)
    for _ in range(max_passes):
        improved = False
        for seg in range(1, max_seg + 1):
            i = 0
            while i < n:
                idx = [(i + k) % n for k in range(seg)]
                s = [t[k] for k in idx]
                prev, nxt = t[(i - 1) % n], t[(i + seg) % n]
                if n - seg < 3:
                    break
                remove_gain = d[prev, s[0]] + d[s[-1], nxt] - d[prev, nxt]
                rest = [t[(i + seg + k) % n] for k in range(n - seg)]
                best, best_pos, best_rev = 1e-12, None, False
                for p in range(len(rest) - 1):
                    u, v = rest[p], rest[p + 1]
                    base = d[u, v]
                    fwd = d[u, s[0]] + d[s[-1], v] - base
                    rev = d[u, s[-1]] + d[s[0], v] - base
                    if remove_gain - fwd > best:
                        best, best_pos, best_rev = remove_gain - fwd, p, False
                    if remove_gain - rev > best:
                        best, best_pos, best_rev = remove_gain - rev, p, True
                if best_pos is not None:
                    seg_nodes = s[::-1] if best_rev else s
                    t = rest[:best_pos + 1] + seg_nodes + rest[best_pos + 1:]
                    improved = True
                i += 1
        if not improved:
            break
    return np.array(t)
