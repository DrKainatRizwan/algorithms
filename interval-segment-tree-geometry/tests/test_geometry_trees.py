import random

import pytest

from geometry_trees import (CenteredIntervalTree, CoverageTree, DynamicIntervalTree,
                            MaxAddTree, RectangleIndex, SortedEndpointIndex,
                            max_overlap_point, rect_intersections, skyline, union_area,
                            union_length)
from geometry_trees import naive
from geometry_trees.bench import random_intervals, random_rects


def int_ivs(n, seed, span=60, max_len=10):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        lo = rng.randrange(span)
        out.append((lo, lo + rng.randrange(max_len)))
    return out


def test_centered_stab_matches_naive():
    ivs = int_ivs(200, 1)
    t = CenteredIntervalTree(ivs)
    for x in range(-2, 75):
        assert t.stab(x) == naive.stab(ivs, x)


def test_centered_overlap_matches_naive():
    ivs = int_ivs(200, 2)
    t = CenteredIntervalTree(ivs)
    rng = random.Random(3)
    for _ in range(200):
        a = rng.randrange(-5, 70)
        b = a + rng.randrange(0, 15)
        assert t.overlap(a, b) == naive.overlap(ivs, a, b)


def test_centered_empty_and_invalid():
    t = CenteredIntervalTree([])
    assert t.stab(1) == [] and t.overlap(0, 5) == [] and len(t) == 0
    with pytest.raises(ValueError):
        CenteredIntervalTree([(3, 1)])
    with pytest.raises(ValueError):
        CenteredIntervalTree([(0, 1)]).overlap(5, 1)


def test_centered_height_logarithmic():
    ivs = random_intervals(4000, random.Random(4))
    assert CenteredIntervalTree(ivs).height() <= 40


def test_dynamic_insert_query_matches_naive():
    ivs = int_ivs(150, 5)
    t = DynamicIntervalTree(seed=2)
    for lo, hi in ivs:
        t.insert(lo, hi)
    assert t.check_invariants()
    for a in range(-3, 70, 3):
        assert t.overlap(a, a + 4) == naive.overlap(ivs, a, a + 4)
        assert t.stab(a) == naive.stab(ivs, a)


def test_dynamic_random_insert_remove():
    rng = random.Random(6)
    t = DynamicIntervalTree(seed=3)
    live = {}
    for step in range(600):
        if live and rng.random() < 0.4:
            k = rng.choice(sorted(live))
            t.remove(k)
            del live[k]
        else:
            lo = rng.randrange(100)
            hi = lo + rng.randrange(12)
            live[t.insert(lo, hi)] = (lo, hi)
        if step % 50 == 0:
            assert t.check_invariants()
            a = rng.randrange(100)
            ids = sorted(k for k, (l, h) in live.items() if l <= a + 5 and a <= h)
            assert t.overlap(a, a + 5) == ids
    assert len(t) == len(live)


def test_dynamic_any_overlap_and_errors():
    t = DynamicIntervalTree()
    assert t.any_overlap(0, 1) is None
    k = t.insert(10, 20)
    assert t.any_overlap(15, 30) == k
    assert t.any_overlap(21, 30) is None
    with pytest.raises(KeyError):
        t.insert(1, 2, key=k)
    with pytest.raises(ValueError):
        t.insert(5, 1)
    t.remove(k)
    assert k not in t and t.any_overlap(0, 100) is None


def test_dynamic_depth_is_logarithmic_on_sorted_input():
    t = DynamicIntervalTree(seed=9)
    for i in range(2000):
        t.insert(i, i + 1)
    assert t.depth() < 45


def test_sorted_endpoint_counts():
    ivs = int_ivs(300, 7)
    idx = SortedEndpointIndex(ivs)
    for x in range(-2, 75):
        assert idx.count_stab(x) == len(naive.stab(ivs, x))
    for a in range(-2, 70, 4):
        assert idx.count_overlap(a, a + 6) == len(naive.overlap(ivs, a, a + 6))


def test_coverage_tree_union_length():
    for seed in range(5):
        ivs = [(float(a), float(b)) for a, b in int_ivs(40, seed)]
        assert union_length(ivs) == naive.union_length(ivs)


def test_coverage_tree_add_remove_returns_to_zero():
    t = CoverageTree([0, 2, 5, 9])
    t.add(0, 5, 1)
    t.add(2, 9, 1)
    assert t.covered == 9
    t.add(0, 5, -1)
    assert t.covered == 7
    t.add(2, 9, -1)
    assert t.covered == 0
    with pytest.raises(KeyError):
        t.add(1, 2, 1)


def test_max_add_tree_against_array():
    rng = random.Random(8)
    n = 50
    arr = [0] * n
    t = MaxAddTree(n)
    for _ in range(300):
        l = rng.randrange(n)
        r = rng.randrange(l, n)
        if rng.random() < 0.6:
            d = rng.randrange(-3, 5)
            t.add(l, r, d)
            for i in range(l, r + 1):
                arr[i] += d
        else:
            val, pos = t.query(l, r)
            assert val == max(arr[l:r + 1])
            assert arr[pos] == val and l <= pos <= r
    assert t.global_max[0] == max(arr)


def test_max_overlap_matches_naive():
    for seed in range(8):
        ivs = int_ivs(60, seed + 20)
        depth, x = max_overlap_point(ivs)
        assert depth == naive.max_overlap(ivs)
        assert len(naive.stab(ivs, x)) == depth


def test_union_area_matches_bruteforce():
    for seed in range(6):
        rng = random.Random(seed)
        rects = random_rects(25, rng, span=50, max_side=20)
        assert union_area(rects) == pytest.approx(naive.rect_union_area(rects))


def test_union_area_known_values():
    assert union_area([(0, 0, 2, 2), (1, 1, 3, 3)]) == 7
    assert union_area([(0, 0, 4, 4), (1, 1, 2, 2)]) == 16
    assert union_area([]) == 0
    assert union_area([(0, 0, 0, 5)]) == 0
    with pytest.raises(ValueError):
        union_area([(2, 0, 1, 1)])


def test_rect_intersections_match_naive():
    for seed in range(5):
        rng = random.Random(seed + 40)
        rects = random_rects(80, rng, span=100, max_side=25)
        assert rect_intersections(rects) == naive.rect_intersections(rects)


def test_rect_intersections_touching_counts():
    assert rect_intersections([(0, 0, 1, 1), (1, 1, 2, 2), (3, 3, 4, 4)]) == [(0, 1)]


def test_rectangle_index_window():
    rng = random.Random(50)
    rects = random_rects(300, rng, span=200, max_side=20)
    idx = RectangleIndex(rects)
    for _ in range(50):
        x, y = rng.uniform(0, 200), rng.uniform(0, 200)
        w = (x, y, x + rng.uniform(0, 40), y + rng.uniform(0, 40))
        assert idx.query(w) == naive.window_query(rects, w)


def test_skyline_classic_example():
    b = [(2, 9, 10), (3, 7, 15), (5, 12, 12), (15, 20, 10), (19, 24, 8)]
    assert skyline(b) == [(2, 10), (3, 15), (7, 12), (12, 0), (15, 10), (20, 8), (24, 0)]


def test_skyline_random_pointwise():
    rng = random.Random(60)
    bs = []
    for _ in range(30):
        l = rng.randrange(40)
        bs.append((l, l + rng.randrange(1, 10), rng.randrange(1, 20)))
    sk = skyline(bs)
    for x in [i + 0.5 for i in range(-1, 52)]:
        expect = max([h for l, r, h in bs if l <= x < r], default=0)
        cur = 0
        for px, ph in sk:
            if px <= x:
                cur = ph
        assert cur == expect
