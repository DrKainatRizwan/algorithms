import math

from skiplist_lab import (GeometricLevels, SkipList, expected_height, expected_search_cost,
                          height_tail_bound, level_histogram, measure_search_cost, optimal_p)
from skiplist_lab.analysis import geometric_chi_square


def test_tower_heights_follow_geometric_distribution():
    sl = SkipList(GeometricLevels(p=0.5, seed=42))
    for k in range(6000):
        sl.insert(k)
    heights = sl.tower_heights()
    assert geometric_chi_square(heights, 0.5) < 25  # ~ chi2 with <=12 dof
    assert abs(sum(heights) / len(heights) - 2.0) < 0.1


def test_height_close_to_theory():
    res = measure_search_cost(4096, p=0.5, seed=3)
    assert abs(res["height"] - res["predicted_height"]) < 5


def test_search_cost_within_pugh_bound():
    for p in (0.25, 0.5):
        res = measure_search_cost(4096, p=p, seed=8)
        # comparisons include one extra failed compare per level, allow slack
        assert res["mean_comparisons"] < 2.0 * res["predicted_steps"]


def test_search_cost_grows_logarithmically():
    a = measure_search_cost(1024, seed=1)["mean_comparisons"]
    b = measure_search_cost(16384, seed=1)["mean_comparisons"]
    assert b < a * 2.0  # 16x more keys, far less than 2x cost


def test_optimal_p_is_near_one_over_e():
    assert 0.3 <= optimal_p(100000) <= 0.45
    assert math.isclose(expected_search_cost(1, 0.5), 1.0)


def test_tail_bound_and_histogram():
    assert height_tail_bound(1000, 0.5, 30) < 1e-6
    assert height_tail_bound(1000, 0.5, 1) == 1.0
    assert level_histogram([1, 2, 1, 3, 1]) == {1: 3, 2: 1, 3: 1}
    assert expected_height(1024, 0.5) == 12.0
