"""Convex hull algorithms (all return counter-clockwise hulls without collinear points)."""

from __future__ import annotations

from typing import Iterable, Sequence

from .primitives import Point, dist, orient, polygon_area


def _unique_sorted(points: Iterable[Point]) -> list[Point]:
    return sorted({(float(x), float(y)) for x, y in points})


def monotone_chain(points: Iterable[Point]) -> list[Point]:
    """Andrew's monotone chain, O(n log n). Starts at the lowest-leftmost point."""
    pts = _unique_sorted(points)
    if len(pts) <= 2:
        return pts
    lower: list[Point] = []
    for p in pts:
        while len(lower) >= 2 and orient(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[Point] = []
    for p in reversed(pts):
        while len(upper) >= 2 and orient(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def graham_scan(points: Iterable[Point]) -> list[Point]:
    """Graham scan: angular sort around the lowest point, O(n log n)."""
    pts = _unique_sorted(points)
    if len(pts) <= 2:
        return pts
    pivot = min(pts, key=lambda p: (p[1], p[0]))
    rest = [p for p in pts if p != pivot]

    import functools

    def cmp(a: Point, b: Point) -> int:
        o = orient(pivot, a, b)
        if o > 0:
            return -1
        if o < 0:
            return 1
        da, db = dist(pivot, a), dist(pivot, b)
        return -1 if da < db else (1 if da > db else 0)

    rest.sort(key=functools.cmp_to_key(cmp))
    stack = [pivot]
    for p in rest:
        while len(stack) >= 2 and orient(stack[-2], stack[-1], p) <= 0:
            stack.pop()
        stack.append(p)
    # all-collinear input degenerates to the two extremes
    if len(stack) == 2 or all(orient(pivot, rest[0], q) == 0 for q in rest):
        return monotone_chain(pts)
    i = stack.index(min(stack))
    return stack[i:] + stack[:i]


def jarvis_march(points: Iterable[Point]) -> list[Point]:
    """Gift wrapping, O(n h). Output-sensitive; good when the hull is tiny."""
    pts = _unique_sorted(points)
    if len(pts) <= 2:
        return pts
    start = pts[0]
    hull = [start]
    cur = start
    while True:
        nxt = pts[0] if pts[0] != cur else pts[1]
        for q in pts:
            if q == cur:
                continue
            o = orient(cur, nxt, q)
            if o < 0 or (o == 0 and dist(cur, q) > dist(cur, nxt)):
                nxt = q
        if nxt == start:
            break
        hull.append(nxt)
        cur = nxt
        if len(hull) > len(pts):  # pragma: no cover - defensive
            break
    # drop collinear intermediates produced by ties
    out: list[Point] = []
    for i, p in enumerate(hull):
        if orient(hull[i - 1], p, hull[(i + 1) % len(hull)]) != 0 or len(hull) <= 2:
            out.append(p)
    return out


def quickhull(points: Iterable[Point]) -> list[Point]:
    """Quickhull: divide and conquer on the farthest point from a chord."""
    pts = _unique_sorted(points)
    if len(pts) <= 2:
        return pts
    left, right = pts[0], pts[-1]

    def build(cands: Sequence[Point], a: Point, b: Point) -> list[Point]:
        # points strictly to the right of a->b (outside a CCW hull edge)
        if not cands:
            return []
        far = min(cands, key=lambda p: orient(a, b, p))
        s1 = [p for p in cands if orient(a, far, p) < 0]
        s2 = [p for p in cands if orient(far, b, p) < 0]
        return build(s1, a, far) + [far] + build(s2, far, b)

    below = [p for p in pts if orient(left, right, p) < 0]
    above = [p for p in pts if orient(left, right, p) > 0]
    # counter-clockwise: left -> (below chain) -> right -> (above chain)
    return [left] + build(below, left, right) + [right] + build(above, right, left)


def convex_hull(points: Iterable[Point], method: str = "monotone") -> list[Point]:
    """Dispatch to a named hull algorithm."""
    table = {
        "monotone": monotone_chain,
        "graham": graham_scan,
        "jarvis": jarvis_march,
        "quickhull": quickhull,
    }
    if method not in table:
        raise ValueError(f"unknown method {method!r}")
    return table[method](points)


def hull_area(hull: Sequence[Point]) -> float:
    """Area enclosed by a counter-clockwise hull."""
    return abs(polygon_area(hull)) if len(hull) >= 3 else 0.0


def hull_perimeter(hull: Sequence[Point]) -> float:
    """Perimeter of a hull (twice the length for a 2-point hull)."""
    n = len(hull)
    return sum(dist(hull[i], hull[(i + 1) % n]) for i in range(n)) if n > 1 else 0.0


def diameter(hull: Sequence[Point]) -> tuple[float, Point, Point]:
    """Farthest pair via rotating calipers, O(h). Returns (distance, p, q)."""
    n = len(hull)
    if n == 0:
        raise ValueError("empty hull")
    if n == 1:
        return 0.0, hull[0], hull[0]
    if n == 2:
        return dist(hull[0], hull[1]), hull[0], hull[1]
    best = (-1.0, hull[0], hull[0])
    j = 1
    for i in range(n):
        ni = (i + 1) % n
        # advance j while the triangle area (i, i+1, j) keeps growing
        while abs(orient(hull[i], hull[ni], hull[(j + 1) % n])) > abs(
            orient(hull[i], hull[ni], hull[j])
        ):
            j = (j + 1) % n
        for k in (i, ni):
            d = dist(hull[k], hull[j])
            if d > best[0]:
                best = (d, hull[k], hull[j])
    return best


def point_in_convex(hull: Sequence[Point], q: Point) -> bool:
    """Inclusive membership test for a CCW convex polygon in O(log h)."""
    n = len(hull)
    if n == 0:
        return False
    if n == 1:
        return q == hull[0]
    if n == 2:
        a, b = hull
        return orient(a, b, q) == 0 and min(a[0], b[0]) <= q[0] <= max(a[0], b[0]) and min(
            a[1], b[1]
        ) <= q[1] <= max(a[1], b[1])
    o = hull[0]
    if orient(o, hull[1], q) < 0 or orient(o, hull[-1], q) > 0:
        return False
    lo, hi = 1, n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if orient(o, hull[mid], q) >= 0:
            lo = mid
        else:
            hi = mid
    return orient(hull[lo], hull[hi], q) >= 0
