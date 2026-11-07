import random
import pytest
from dlx_exact_cover import DLX, solve_dict, count_dict

# Knuth's example from "Dancing Links": 7 columns, 6 options, unique cover.
KNUTH = [[2, 4, 5], [0, 3, 6], [1, 2, 5], [0, 3], [1, 6], [3, 4, 6]]


def knuth():
    d = DLX(7)
    for o in KNUTH:
        d.add_option(o)
    return d


def test_knuth_example_unique_solution():
    sols = list(knuth().solve())
    assert sols == [[0, 3, 4]]


def test_solution_is_exact_cover():
    d = knuth()
    sol = d.first_solution()
    covered = sorted(c for i in sol for c in KNUTH[i])
    assert covered == list(range(7))


def test_structure_restored_after_search():
    d = knuth()
    before = (list(d.L), list(d.R), list(d.U), list(d.D), list(d.S))
    list(d.solve())
    assert (d.L, d.R, d.U, d.D, d.S) == before
    assert d.is_consistent()


def test_infeasible_returns_none():
    d = DLX(3)
    d.add_option([0, 1]); d.add_option([1, 2])
    assert d.first_solution() is None
    assert d.count() == 0


def test_empty_problem_has_one_empty_solution():
    assert list(DLX(0).solve()) == [[]]


def test_secondary_columns_at_most_once():
    d = DLX(2, 1)                       # column 2 is secondary
    d.add_option([0, 2]); d.add_option([1, 2]); d.add_option([1])
    sols = list(d.solve())
    assert sols == [[0, 2]]             # options 0 and 1 conflict on column 2


def test_limit_stops_enumeration():
    d = DLX(2)
    for o in ([0], [0], [1], [1]):
        d.add_option(o)
    assert d.count() == 4
    assert d.count(limit=2) == 2
    assert list(d.solve(limit=0)) == []


def test_invalid_options_rejected():
    d = DLX(3)
    with pytest.raises(ValueError):
        d.add_option([0, 0])
    with pytest.raises(ValueError):
        d.add_option([5])
    with pytest.raises(ValueError):
        d.add_option([])
    with pytest.raises(ValueError):
        DLX(2, heuristic="bogus")


@pytest.mark.parametrize("seed", range(8))
def test_matches_dict_baseline_on_random_instances(seed):
    rng = random.Random(seed)
    n_p, n_s = 6, 3
    opts = [rng.sample(range(n_p + n_s), rng.randint(1, 4)) for _ in range(14)]
    for h in ("mrv", "first"):
        d = DLX(n_p, n_s, heuristic=h)
        for o in opts:
            d.add_option(o)
        mine = sorted(map(tuple, d.solve()))
        ref = sorted(map(tuple, solve_dict(n_p, n_s, opts)))
        assert mine == ref


def test_stats_populated():
    d = knuth()
    d.count()
    assert d.stats.nodes > 0 and d.stats.solutions == 1 and d.stats.updates > 0


def test_mrv_not_worse_than_first_on_queens():
    from dlx_exact_cover.polyomino import _queens_dlx
    counts = {}
    for h in ("first", "mrv"):
        d = _queens_dlx(7); d.heuristic = h
        assert d.count() == 40
        counts[h] = d.stats.nodes
    assert counts["mrv"] <= counts["first"]


def test_count_dict_agrees():
    assert count_dict(7, 0, KNUTH) == 1
