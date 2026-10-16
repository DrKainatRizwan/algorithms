"""Brute-force reference implementations used for validation and benchmarks."""
from __future__ import annotations

from typing import Sequence

Interval = tuple[float, float]
Rect = tuple[float, float, float, float]  # x1, y1, x2, y2


def stab(intervals: Sequence[Interval], x: float) -> list[int]:
    """Indices of closed intervals containing ``x``."""
    return [i for i, (lo, hi) in enumerate(intervals) if lo <= x <= hi]


def overlap(intervals: Sequence[Interval], lo: float, hi: float) -> list[int]:
    """Indices of closed intervals intersecting ``[lo, hi]``."""
    return [i for i, (a, b) in enumerate(intervals) if a <= hi and lo <= b]


def union_length(intervals: Sequence[Interval]) -> float:
    """Total length of the union of intervals."""
    total, cur_lo, cur_hi = 0.0, None, None
    for lo, hi in sorted(intervals):
        if cur_hi is None or lo > cur_hi:
            if cur_hi is not None:
                total += cur_hi - cur_lo
            cur_lo, cur_hi = lo, hi
        else:
            cur_hi = max(cur_hi, hi)
    if cur_hi is not None:
        total += cur_hi - cur_lo
    return total


def max_overlap(intervals: Sequence[Interval]) -> int:
    """Maximum number of closed intervals sharing a common point."""
    events = []
    for lo, hi in intervals:
        events.append((lo, 0))   # starts before ends at equal coordinate
        events.append((hi, 1))
    best = cur = 0
    for _, kind in sorted(events):
        cur += 1 if kind == 0 else -1
        best = max(best, cur)
    return best


def rect_union_area(rects: Sequence[Rect]) -> float:
    """Union area by coordinate compression, O(n^3)."""
    if not rects:
        return 0.0
    xs = sorted({r[0] for r in rects} | {r[2] for r in rects})
    ys = sorted({r[1] for r in rects} | {r[3] for r in rects})
    area = 0.0
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            cx, cy = xs[i], ys[j]
            if any(r[0] <= cx and xs[i + 1] <= r[2] and r[1] <= cy and ys[j + 1] <= r[3]
                   for r in rects):
                area += (xs[i + 1] - cx) * (ys[j + 1] - cy)
    return area


def rect_intersections(rects: Sequence[Rect]) -> list[tuple[int, int]]:
    """All index pairs of closed rectangles that intersect, O(n^2)."""
    out = []
    for i in range(len(rects)):
        a = rects[i]
        for j in range(i + 1, len(rects)):
            b = rects[j]
            if a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]:
                out.append((i, j))
    return out


def window_query(rects: Sequence[Rect], window: Rect) -> list[int]:
    """Indices of rectangles intersecting the query window."""
    return [i for i, r in enumerate(rects)
            if r[0] <= window[2] and window[0] <= r[2] and r[1] <= window[3] and window[1] <= r[3]]
