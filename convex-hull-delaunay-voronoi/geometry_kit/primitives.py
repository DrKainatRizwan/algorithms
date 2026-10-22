"""Geometric predicates and small helpers shared by all modules."""

from __future__ import annotations

import math
from typing import Sequence

Point = tuple[float, float]


def orient(a: Point, b: Point, c: Point) -> float:
    """Twice the signed area of triangle abc (>0 for counter-clockwise)."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def in_circle(a: Point, b: Point, c: Point, d: Point) -> float:
    """Incircle determinant, translated to ``d`` for accuracy.

    For a counter-clockwise triangle abc the result is positive iff d lies
    strictly inside the circumcircle.
    """
    adx, ady = a[0] - d[0], a[1] - d[1]
    bdx, bdy = b[0] - d[0], b[1] - d[1]
    cdx, cdy = c[0] - d[0], c[1] - d[1]
    ad = adx * adx + ady * ady
    bd = bdx * bdx + bdy * bdy
    cd = cdx * cdx + cdy * cdy
    return (
        adx * (bdy * cd - bd * cdy)
        - ady * (bdx * cd - bd * cdx)
        + ad * (bdx * cdy - bdy * cdx)
    )


def circumcenter(a: Point, b: Point, c: Point) -> Point:
    """Circumcentre of triangle abc. Raises ValueError if degenerate."""
    bx, by = b[0] - a[0], b[1] - a[1]
    cx, cy = c[0] - a[0], c[1] - a[1]
    d = 2.0 * (bx * cy - by * cx)
    if d == 0.0:
        raise ValueError("collinear points have no circumcentre")
    b2 = bx * bx + by * by
    c2 = cx * cx + cy * cy
    ux = (cy * b2 - by * c2) / d
    uy = (bx * c2 - cx * b2) / d
    return (a[0] + ux, a[1] + uy)


def polygon_area(poly: Sequence[Point]) -> float:
    """Signed area of a polygon (shoelace); positive if counter-clockwise."""
    s = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return s / 2.0


def dist(a: Point, b: Point) -> float:
    """Euclidean distance."""
    return math.hypot(a[0] - b[0], a[1] - b[1])
