"""Voronoi diagram obtained from the Delaunay graph, clipped to a bounding box."""

from __future__ import annotations

from typing import Iterable, Sequence

from .delaunay import Delaunay
from .primitives import Point, circumcenter, polygon_area

Polygon = list[Point]


def clip_halfplane(poly: Sequence[Point], p: Point, q: Point) -> Polygon:
    """Clip a convex polygon to the half-plane of points closer to ``p`` than ``q``.

    Sutherland-Hodgman against the perpendicular bisector of pq.
    """
    nx, ny = q[0] - p[0], q[1] - p[1]
    c = (q[0] ** 2 + q[1] ** 2 - p[0] ** 2 - p[1] ** 2) / 2.0

    def f(pt: Point) -> float:  # <= 0 means on p's side
        return nx * pt[0] + ny * pt[1] - c

    out: Polygon = []
    m = len(poly)
    for i in range(m):
        a, b = poly[i], poly[(i + 1) % m]
        fa, fb = f(a), f(b)
        if fa <= 0:
            out.append(a)
        if (fa < 0 < fb) or (fb < 0 < fa):
            t = fa / (fa - fb)
            out.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
    return out


class Voronoi:
    """Bounded Voronoi diagram: one convex polygon per site.

    Each cell starts as the bounding box and is clipped by the bisectors with
    the site's Delaunay neighbours, which is sufficient for correctness.
    """

    def __init__(self, points: Iterable[Point], box: tuple[float, float, float, float] | None = None, seed: int = 0) -> None:
        self.delaunay = Delaunay(points, seed=seed)
        self.sites = self.delaunay.points
        if box is None:
            xs = [p[0] for p in self.sites]
            ys = [p[1] for p in self.sites]
            padx = (max(xs) - min(xs)) * 0.1 + 1.0
            pady = (max(ys) - min(ys)) * 0.1 + 1.0
            box = (min(xs) - padx, min(ys) - pady, max(xs) + padx, max(ys) + pady)
        self.box = box
        x0, y0, x1, y1 = box
        self._rect: Polygon = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        self._adj = self.delaunay.neighbors()
        self._cells: dict[int, Polygon] = {}

    def cell(self, i: int) -> Polygon:
        """Voronoi cell of site ``i`` (CCW polygon clipped to the box)."""
        if i not in self._cells:
            poly = list(self._rect)
            p = self.sites[i]
            for j in self._adj[i]:
                poly = clip_halfplane(poly, p, self.sites[j])
                if not poly:
                    break
            self._cells[i] = poly
        return self._cells[i]

    def cells(self) -> list[Polygon]:
        return [self.cell(i) for i in range(len(self.sites))]

    def cell_areas(self) -> list[float]:
        return [abs(polygon_area(c)) if len(c) >= 3 else 0.0 for c in self.cells()]

    def vertices(self) -> list[Point]:
        """Unbounded Voronoi vertices: circumcentres of Delaunay triangles."""
        return [
            circumcenter(self.sites[a], self.sites[b], self.sites[c])
            for a, b, c in self.delaunay.triangles
        ]

    def lloyd_step(self) -> list[Point]:
        """One Lloyd relaxation step: move each site to its cell centroid."""
        out: list[Point] = []
        for i, cell in enumerate(self.cells()):
            a = polygon_area(cell)
            if abs(a) < 1e-15:
                out.append(self.sites[i])
                continue
            cx = cy = 0.0
            for k in range(len(cell)):
                x1, y1 = cell[k]
                x2, y2 = cell[(k + 1) % len(cell)]
                w = x1 * y2 - x2 * y1
                cx += (x1 + x2) * w
                cy += (y1 + y2) * w
            out.append((cx / (6 * a), cy / (6 * a)))
        return out
