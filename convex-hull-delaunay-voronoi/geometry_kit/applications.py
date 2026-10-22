"""Applications that exploit the Delaunay triangulation."""

from __future__ import annotations

from typing import Sequence

from .delaunay import Delaunay
from .hull import monotone_chain, point_in_convex
from .primitives import Point, circumcenter, dist


class _DSU:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> bool:
        a, b = self.find(a), self.find(b)
        if a == b:
            return False
        self.p[a] = b
        return True


def euclidean_mst(points: Sequence[Point], seed: int = 0) -> list[tuple[int, int, float]]:
    """Euclidean MST via Kruskal on Delaunay edges (EMST is a subgraph), O(n log n)."""
    d = Delaunay(points, seed=seed)
    edges = sorted((dist(d.points[u], d.points[v]), u, v) for u, v in d.edges())
    dsu = _DSU(len(points))
    for dup, orig in d.duplicates.items():
        dsu.union(dup, orig)
    out = [(dup, orig, 0.0) for dup, orig in d.duplicates.items()]
    for w, u, v in edges:
        if dsu.union(u, v):
            out.append((u, v, w))
    return out


def closest_pair(points: Sequence[Point], seed: int = 0) -> tuple[float, int, int]:
    """Closest pair: the minimum is always a Delaunay edge."""
    d = Delaunay(points, seed=seed)
    if d.duplicates:
        dup, orig = next(iter(d.duplicates.items()))
        return 0.0, orig, dup
    best = min((dist(d.points[u], d.points[v]), u, v) for u, v in d.edges())
    return best


def largest_empty_circle(points: Sequence[Point], seed: int = 0) -> tuple[float, Point]:
    """Largest circle centred inside the hull that contains no site.

    The optimum is at a Voronoi vertex inside the hull or on the intersection
    of a Voronoi edge with the hull boundary; here the vertex candidates are
    circumcentres inside the hull, which covers the interior case.
    """
    d = Delaunay(points, seed=seed)
    hull = monotone_chain(points)
    best = (0.0, hull[0] if hull else (0.0, 0.0))
    for a, b, c in d.triangles:
        pa, pb, pc = d.points[a], d.points[b], d.points[c]
        try:
            cc = circumcenter(pa, pb, pc)
        except ValueError:
            continue
        if point_in_convex(hull, cc):
            r = dist(cc, pa)
            if r > best[0]:
                best = (r, cc)
    return best


def nearest_site(d: Delaunay, q: Point, start: int = 0) -> int:
    """Nearest neighbour by greedy descent on the Delaunay graph.

    Greedy routing on a Delaunay graph always terminates at the true nearest
    site, because every non-nearest site has a neighbour closer to ``q``.
    """
    adj = d.neighbors()
    cur = start
    cd = dist(d.points[cur], q)
    while True:
        best, bd = cur, cd
        for j in adj[cur]:
            dj = dist(d.points[j], q)
            if dj < bd:
                best, bd = j, dj
        if best == cur:
            return cur
        cur, cd = best, bd
