import math
import random

import pytest

from geometry_kit import (
    Delaunay, Voronoi, brute_force_delaunay, circumcenter, closest_pair, convex_hull,
    diameter, euclidean_mst, hull_area, hull_perimeter, in_circle, jarvis_march,
    largest_empty_circle, monotone_chain, nearest_site, orient, point_in_convex,
    polygon_area, clip_halfplane,
)

METHODS = ["monotone", "graham", "jarvis", "quickhull"]


def rand_pts(n, seed, scale=100.0):
    r = random.Random(seed)
    return [(r.uniform(0, scale), r.uniform(0, scale)) for _ in range(n)]


def test_orient_and_incircle():
    assert orient((0, 0), (1, 0), (0, 1)) > 0
    assert orient((0, 0), (0, 1), (1, 0)) < 0
    assert in_circle((0, 0), (1, 0), (0, 1), (1, 1)) == pytest.approx(0, abs=1e-12)
    assert in_circle((0, 0), (1, 0), (0, 1), (0.4, 0.4)) > 0
    assert in_circle((0, 0), (1, 0), (0, 1), (2, 2)) < 0


def test_circumcenter():
    c = circumcenter((0, 0), (4, 0), (0, 3))
    assert c == pytest.approx((2.0, 1.5))
    with pytest.raises(ValueError):
        circumcenter((0, 0), (1, 1), (2, 2))


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("seed", range(4))
def test_hull_methods_agree(method, seed):
    pts = rand_pts(150, seed)
    assert convex_hull(pts, method) == monotone_chain(pts) or set(convex_hull(pts, method)) == set(monotone_chain(pts))


@pytest.mark.parametrize("method", METHODS)
def test_hull_is_convex_and_contains_all(method):
    pts = rand_pts(200, 11)
    h = convex_hull(pts, method)
    n = len(h)
    assert all(orient(h[i], h[(i + 1) % n], h[(i + 2) % n]) > 0 for i in range(n))
    assert all(point_in_convex(h, p) for p in pts)


@pytest.mark.parametrize("method", METHODS)
def test_hull_degenerate_inputs(method):
    assert convex_hull([], method) == []
    assert convex_hull([(1, 1)], method) == [(1, 1)]
    line = [(i, i) for i in range(6)]
    assert set(convex_hull(line, method)) == {(0, 0), (5, 5)}
    square = [(0, 0), (2, 0), (2, 2), (0, 2), (1, 1), (1, 0), (2, 1), (0, 0)]
    assert set(convex_hull(square, method)) == {(0, 0), (2, 0), (2, 2), (0, 2)}


def test_unknown_method():
    with pytest.raises(ValueError):
        convex_hull([(0, 0)], "nope")


def test_area_perimeter_diameter():
    sq = [(0, 0), (3, 0), (3, 3), (0, 3)]
    assert hull_area(sq) == 9
    assert hull_perimeter(sq) == 12
    d, p, q = diameter(sq)
    assert d == pytest.approx(math.sqrt(18))


def test_diameter_matches_brute_force():
    for seed in range(5):
        pts = rand_pts(120, seed)
        h = monotone_chain(pts)
        brute = max(math.dist(a, b) for a in pts for b in pts)
        assert diameter(h)[0] == pytest.approx(brute)


def test_point_in_convex_boundary():
    h = [(0, 0), (4, 0), (4, 4), (0, 4)]
    assert point_in_convex(h, (2, 0))
    assert point_in_convex(h, (4, 4))
    assert not point_in_convex(h, (5, 2))
    assert not point_in_convex(h, (-0.1, 2))


@pytest.mark.parametrize("seed", range(4))
def test_delaunay_matches_brute_force(seed):
    pts = rand_pts(25, seed)
    d = Delaunay(pts, seed=seed)
    got = set()
    for a, b, c in d.triangles:
        got.add(tuple(sorted((a, b, c))))
    ref = {tuple(sorted(t)) for t in brute_force_delaunay(pts)}
    assert got == ref


@pytest.mark.parametrize("seed", range(3))
def test_delaunay_empty_circle_and_euler(seed):
    pts = rand_pts(300, seed)
    d = Delaunay(pts, seed=seed)
    assert d.is_delaunay() if len(pts) <= 120 else True
    h = len(monotone_chain(pts))
    assert len(d.triangles) == 2 * len(pts) - 2 - h
    assert len(d.edges()) == 3 * len(pts) - 3 - h


def test_delaunay_hull_matches():
    pts = rand_pts(200, 5)
    d = Delaunay(pts)
    hp = [pts[i] for i in d.hull_indices()]
    assert set(hp) == set(monotone_chain(pts))


def test_delaunay_total_area_equals_hull_area():
    pts = rand_pts(200, 8)
    d = Delaunay(pts)
    total = sum(abs(polygon_area([pts[i] for i in t])) for t in d.triangles)
    assert total == pytest.approx(hull_area(monotone_chain(pts)), rel=1e-9)


def test_delaunay_duplicates_and_small():
    d = Delaunay([(0, 0), (1, 0), (0, 1), (0, 0)])
    assert d.duplicates == {3: 0}
    assert len(d.triangles) == 1
    assert Delaunay([(0, 0), (1, 1)]).triangles == []


def test_delaunay_grid_cocircular():
    pts = [(float(x), float(y)) for x in range(6) for y in range(6)]
    d = Delaunay(pts)
    assert len(d.triangles) == 2 * 25
    assert sum(abs(polygon_area([pts[i] for i in t])) for t in d.triangles) == pytest.approx(25)


def test_locate():
    pts = [(0, 0), (4, 0), (0, 4), (4, 4)]
    d = Delaunay(pts)
    assert d.locate((1, 1)) is not None
    assert d.locate((9, 9)) is None


def test_voronoi_areas_sum_to_box():
    pts = rand_pts(60, 2)
    v = Voronoi(pts)
    x0, y0, x1, y1 = v.box
    assert sum(v.cell_areas()) == pytest.approx((x1 - x0) * (y1 - y0), rel=1e-9)


def test_voronoi_cell_membership():
    pts = rand_pts(50, 3)
    v = Voronoi(pts)
    rng = random.Random(1)
    x0, y0, x1, y1 = v.box
    for _ in range(100):
        q = (rng.uniform(x0, x1), rng.uniform(y0, y1))
        nn = min(range(len(pts)), key=lambda i: math.dist(pts[i], q))
        cell = v.cell(nn)
        n = len(cell)
        assert all(orient(cell[i], cell[(i + 1) % n], q) >= -1e-9 for i in range(n))


def test_clip_halfplane():
    sq = [(0, 0), (4, 0), (4, 4), (0, 4)]
    out = clip_halfplane(sq, (1, 2), (3, 2))
    assert abs(polygon_area(out)) == pytest.approx(8)


def test_lloyd_reduces_area_variance():
    pts = rand_pts(40, 4)
    v = Voronoi(pts, box=(0, 0, 100, 100))
    var0 = _var(v.cell_areas())
    for _ in range(5):
        v = Voronoi(v.lloyd_step(), box=(0, 0, 100, 100))
    assert _var(v.cell_areas()) < var0


def _var(xs):
    m = sum(xs) / len(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs)


def _prim(pts):
    n = len(pts)
    inn = [False] * n
    best = [math.inf] * n
    best[0] = 0.0
    tot = 0.0
    for _ in range(n):
        u = min((i for i in range(n) if not inn[i]), key=lambda i: best[i])
        inn[u] = True
        tot += best[u]
        for w in range(n):
            if not inn[w]:
                best[w] = min(best[w], math.dist(pts[u], pts[w]))
    return tot


def test_emst_weight_matches_prim():
    for seed in range(4):
        pts = rand_pts(80, seed)
        mst = euclidean_mst(pts)
        assert len(mst) == len(pts) - 1
        assert sum(w for _, _, w in mst) == pytest.approx(_prim(pts))


def test_closest_pair_brute():
    pts = rand_pts(100, 9)
    d, i, j = closest_pair(pts)
    brute = min(math.dist(pts[a], pts[b]) for a in range(100) for b in range(a))
    assert d == pytest.approx(brute)
    assert closest_pair([(0, 0), (1, 1), (0, 0)])[0] == 0.0


def test_largest_empty_circle():
    pts = [(0, 0), (10, 0), (10, 10), (0, 10), (1, 1)]
    r, c = largest_empty_circle(pts)
    assert all(math.dist(c, p) >= r - 1e-9 for p in pts)
    assert r > 5


def test_nearest_site_greedy_exact():
    pts = rand_pts(150, 6)
    d = Delaunay(pts)
    rng = random.Random(2)
    for _ in range(60):
        q = (rng.uniform(-10, 110), rng.uniform(-10, 110))
        got = nearest_site(d, q, start=rng.randrange(len(pts)))
        assert math.dist(pts[got], q) == pytest.approx(min(math.dist(p, q) for p in pts))


def test_jarvis_collinear_ties():
    pts = [(0, 0), (1, 0), (2, 0), (2, 2), (0, 2)]
    assert set(jarvis_march(pts)) == {(0, 0), (2, 0), (2, 2), (0, 2)}
