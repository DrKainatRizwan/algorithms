"""Plane-sweep algorithms built on the tree structures."""
from __future__ import annotations

from typing import Sequence

from .interval_tree import CenteredIntervalTree, DynamicIntervalTree
from .segment_tree import CoverageTree, MaxAddTree

Interval = tuple[float, float]
Rect = tuple[float, float, float, float]  # x1, y1, x2, y2 with x1<=x2, y1<=y2


def _check_rects(rects: Sequence[Rect]) -> None:
    for r in rects:
        if r[0] > r[2] or r[1] > r[3]:
            raise ValueError(f"invalid rectangle {r}")


def union_area(rects: Sequence[Rect]) -> float:
    """Area of the union of axis-aligned rectangles (Klee's measure in 2-D).

    Sweep over x; a CoverageTree over compressed y reports the covered
    y-length after each event.  O(n log n).
    """
    _check_rects(rects)
    rects = [r for r in rects if r[0] < r[2] and r[1] < r[3]]
    if not rects:
        return 0.0
    tree = CoverageTree([y for r in rects for y in (r[1], r[3])])
    events = []
    for x1, y1, x2, y2 in rects:
        events.append((x1, 1, y1, y2))
        events.append((x2, -1, y1, y2))
    events.sort()
    area, prev_x = 0.0, events[0][0]
    for x, d, y1, y2 in events:
        area += tree.covered * (x - prev_x)
        prev_x = x
        tree.add(y1, y2, d)
    return area


def union_length(intervals: Sequence[Interval]) -> float:
    """Total length of a union of 1-D intervals via the coverage tree."""
    ivs = [iv for iv in intervals if iv[0] < iv[1]]
    if not ivs:
        return 0.0
    tree = CoverageTree([p for iv in ivs for p in iv])
    for lo, hi in ivs:
        tree.add(lo, hi, 1)
    return tree.covered


def max_overlap_point(intervals: Sequence[Interval]) -> tuple[int, float]:
    """(depth, coordinate) of a point covered by the most closed intervals.

    Compress endpoints and use a lazy range-add/max tree; O(n log n).
    """
    if not intervals:
        return 0, float("nan")
    pts = sorted({p for iv in intervals for p in iv})
    index = {p: i for i, p in enumerate(pts)}
    tree = MaxAddTree(len(pts))
    for lo, hi in intervals:
        tree.add(index[lo], index[hi], 1)
    depth, pos = tree.global_max
    return depth, pts[pos]


def rect_intersections(rects: Sequence[Rect]) -> list[tuple[int, int]]:
    """All intersecting pairs ``(i, j)``, ``i < j``, of closed rectangles.

    Sweep over x keeping the y-intervals of active rectangles in a
    DynamicIntervalTree.  Output-sensitive: O((n + k) log n).
    """
    _check_rects(rects)
    events = []
    for i, (x1, _, x2, _) in enumerate(rects):
        events.append((x1, 0, i))   # insertions before removals at equal x
        events.append((x2, 1, i))
    events.sort()
    active = DynamicIntervalTree(seed=1)
    pairs: set[tuple[int, int]] = set()
    for _, kind, i in events:
        y1, y2 = rects[i][1], rects[i][3]
        if kind == 0:
            for j in active.overlap(y1, y2):
                pairs.add((min(i, j), max(i, j)))
            active.insert(y1, y2, key=i)
        else:
            active.remove(i)
    return sorted(pairs)


class RectangleIndex:
    """Static window-query index for rectangles.

    Rectangles are bucketed by a CenteredIntervalTree over x; the y-test is
    applied to the x-candidates.  Good when windows are thin in x.
    """

    def __init__(self, rects: Sequence[Rect]):
        _check_rects(rects)
        self.rects = list(rects)
        self._xtree = CenteredIntervalTree([(r[0], r[2]) for r in self.rects])

    def query(self, window: Rect) -> list[int]:
        """Indices of rectangles intersecting ``window``."""
        cand = self._xtree.overlap(window[0], window[2])
        return [i for i in cand
                if self.rects[i][1] <= window[3] and window[1] <= self.rects[i][3]]


def skyline(buildings: Sequence[tuple[float, float, float]]) -> list[tuple[float, float]]:
    """Skyline outline of ``(left, right, height)`` buildings.

    Returns key points ``(x, height)`` where the max height changes.  A
    lazy max-add tree over compressed x holds per-slab heights via a
    max-tag segment tree resolved top-down.
    """
    if not buildings:
        return []
    xs = sorted({x for l, r, _ in buildings for x in (l, r)})
    index = {x: i for i, x in enumerate(xs)}
    m = len(xs) - 1
    if m <= 0:
        return []
    # segment tree with "assign max" tags on canonical nodes, resolved at leaves
    tag = [0.0] * (4 * m)

    def upd(node: int, lo: int, hi: int, a: int, b: int, h: float) -> None:
        if b <= lo or hi <= a:
            return
        if a <= lo and hi <= b:
            tag[node] = max(tag[node], h)
            return
        mid = (lo + hi) // 2
        upd(2 * node, lo, mid, a, b, h)
        upd(2 * node + 1, mid, hi, a, b, h)

    for l, r, h in buildings:
        if l < r:
            upd(1, 0, m, index[l], index[r], h)
    heights = [0.0] * m

    def down(node: int, lo: int, hi: int, acc: float) -> None:
        acc = max(acc, tag[node])
        if hi - lo == 1:
            heights[lo] = acc
            return
        mid = (lo + hi) // 2
        down(2 * node, lo, mid, acc)
        down(2 * node + 1, mid, hi, acc)

    down(1, 0, m, 0.0)
    out: list[tuple[float, float]] = []
    prev = 0.0
    for i in range(m):
        if heights[i] != prev:
            out.append((xs[i], heights[i]))
            prev = heights[i]
    if prev != 0.0:
        out.append((xs[m], 0.0))
    return out
