"""Incremental Bowyer-Watson Delaunay triangulation with walking point location."""

from __future__ import annotations

import random
from typing import Iterable

from .primitives import Point, in_circle, orient

Edge = tuple[int, int]


class Delaunay:
    """Delaunay triangulation of a planar point set.

    Points are inserted in a seeded random order; each insertion locates its
    containing triangle by a visibility walk, grows the cavity of triangles
    whose circumcircle contains the point, and re-triangulates the cavity.
    Expected O(n log n) for random insertion order.

    Triangle vertex indices refer to the original input order. Duplicate input
    points are ignored (they map to the first occurrence).
    """

    SUPER_SCALE = 1e4

    def __init__(self, points: Iterable[Point], seed: int = 0) -> None:
        self.points: list[Point] = [(float(x), float(y)) for x, y in points]
        self._rng = random.Random(seed)
        n = len(self.points)
        self.duplicates: dict[int, int] = {}
        first: dict[Point, int] = {}
        order: list[int] = []
        for i, p in enumerate(self.points):
            if p in first:
                self.duplicates[i] = first[p]
            else:
                first[p] = i
                order.append(i)
        self._n = n
        self._verts: list[Point] = list(self.points)
        self._tris: dict[int, tuple[int, int, int]] = {}
        self._edge: dict[Edge, int] = {}
        self._next_id = 0
        self._last = -1
        self._adj: dict[int, set[int]] | None = None
        if len(order) >= 3:
            self._build(order)

    # ---- construction -------------------------------------------------
    def _add_tri(self, a: int, b: int, c: int) -> int:
        t = self._next_id
        self._next_id += 1
        self._tris[t] = (a, b, c)
        self._edge[(a, b)] = t
        self._edge[(b, c)] = t
        self._edge[(c, a)] = t
        self._last = t
        return t

    def _remove_tri(self, t: int) -> None:
        a, b, c = self._tris.pop(t)
        for e in ((a, b), (b, c), (c, a)):
            if self._edge.get(e) == t:
                del self._edge[e]

    def _build(self, order: list[int]) -> None:
        xs = [self.points[i][0] for i in order]
        ys = [self.points[i][1] for i in order]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
        r = span * self.SUPER_SCALE
        s0 = len(self._verts)
        self._verts += [(cx - 2 * r, cy - r), (cx + 2 * r, cy - r), (cx, cy + 2 * r)]
        self._add_tri(s0, s0 + 1, s0 + 2)
        shuffled = order[:]
        self._rng.shuffle(shuffled)
        for i in shuffled:
            self._insert(i)

    def _locate(self, p: Point) -> int:
        """Visibility walk from the most recent triangle (linear fallback)."""
        t = self._last if self._last in self._tris else next(iter(self._tris))
        for _ in range(4 * len(self._tris) + 10):
            a, b, c = self._tris[t]
            moved = False
            edges = [(a, b), (b, c), (c, a)]
            self._rng.shuffle(edges)
            for u, v in edges:
                if orient(self._verts[u], self._verts[v], p) < 0:
                    nb = self._edge.get((v, u))
                    if nb is None:
                        return t
                    t = nb
                    moved = True
                    break
            if not moved:
                return t
        for t, (a, b, c) in self._tris.items():  # pragma: no cover
            va, vb, vc = self._verts[a], self._verts[b], self._verts[c]
            if min(orient(va, vb, p), orient(vb, vc, p), orient(vc, va, p)) >= 0:
                return t
        raise RuntimeError("point location failed")  # pragma: no cover

    def _insert(self, idx: int) -> None:
        p = self._verts[idx]
        start = self._locate(p)
        bad = {start}
        stack = [start]
        while stack:
            t = stack.pop()
            a, b, c = self._tris[t]
            for u, v in ((a, b), (b, c), (c, a)):
                nb = self._edge.get((v, u))
                if nb is None or nb in bad:
                    continue
                x, y, z = self._tris[nb]
                if in_circle(self._verts[x], self._verts[y], self._verts[z], p) > 0:
                    bad.add(nb)
                    stack.append(nb)
        boundary: list[Edge] = []
        for t in bad:
            a, b, c = self._tris[t]
            for u, v in ((a, b), (b, c), (c, a)):
                if self._edge.get((v, u), -1) not in bad:
                    boundary.append((u, v))
        for t in bad:
            self._remove_tri(t)
        for u, v in boundary:
            self._add_tri(u, v, idx)

    # ---- queries ------------------------------------------------------
    @property
    def triangles(self) -> list[tuple[int, int, int]]:
        """CCW triangles over the real points (super-triangle ones removed)."""
        return sorted(t for t in self._tris.values() if max(t) < self._n)

    def edges(self) -> set[Edge]:
        """Undirected Delaunay edges as sorted index pairs."""
        out: set[Edge] = set()
        for a, b, c in self.triangles:
            for u, v in ((a, b), (b, c), (c, a)):
                out.add((min(u, v), max(u, v)))
        return out

    def neighbors(self) -> dict[int, set[int]]:
        """Adjacency lists of the Delaunay graph."""
        if self._adj is not None:
            return self._adj
        adj: dict[int, set[int]] = {i: set() for i in range(self._n)}
        for u, v in self.edges():
            adj[u].add(v)
            adj[v].add(u)
        for dup, orig in self.duplicates.items():
            adj[dup] = set(adj[orig])
        self._adj = adj
        return adj

    def hull_indices(self) -> list[int]:
        """Boundary of the triangulation (the convex hull, CCW) as indices."""
        real = self.triangles
        directed = {(u, v) for a, b, c in real for u, v in ((a, b), (b, c), (c, a))}
        nxt = {u: v for (u, v) in directed if (v, u) not in directed}
        if not nxt:
            return []
        start = min(nxt)
        out = [start]
        cur = nxt[start]
        while cur != start and len(out) <= len(nxt):
            out.append(cur)
            cur = nxt[cur]
        return out

    def is_delaunay(self) -> bool:
        """Check the empty-circumcircle property against every site, O(n t)."""
        ids = [i for i in range(self._n) if i not in self.duplicates]
        for a, b, c in self.triangles:
            va, vb, vc = self.points[a], self.points[b], self.points[c]
            for i in ids:
                if i in (a, b, c):
                    continue
                if in_circle(va, vb, vc, self.points[i]) > 1e-9:
                    return False
        return True

    def locate(self, q: Point) -> tuple[int, int, int] | None:
        """Triangle containing ``q`` or None when outside the hull."""
        if not self._tris:
            return None
        t = self._locate(q)
        tri = self._tris[t]
        if max(tri) >= self._n:
            return None
        va, vb, vc = (self._verts[i] for i in tri)
        if min(orient(va, vb, q), orient(vb, vc, q), orient(vc, va, q)) < 0:
            return None
        return tri


def brute_force_delaunay(points: list[Point]) -> set[tuple[int, int, int]]:
    """O(n^4) reference: every triple with an empty circumcircle (general position)."""
    n = len(points)
    out: set[tuple[int, int, int]] = set()
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a, b, c = points[i], points[j], points[k]
                o = orient(a, b, c)
                if o == 0:
                    continue
                tri = (i, j, k) if o > 0 else (i, k, j)
                va, vb, vc = (points[x] for x in tri)
                if all(
                    in_circle(va, vb, vc, points[m]) <= 0
                    for m in range(n)
                    if m not in (i, j, k)
                ):
                    out.add(tri)
    return out
