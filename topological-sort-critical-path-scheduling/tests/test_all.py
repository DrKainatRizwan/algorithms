import itertools
import math

import numpy as np
import pytest

from project_scheduler import *
from project_scheduler.cpm import critical_paths, crash_sensitivity

CLASSIC = [
    Activity("A", 3), Activity("B", 4, ("A",)), Activity("C", 2, ("A",)),
    Activity("D", 5, ("B", "C")), Activity("E", 1, ("C",)), Activity("F", 2, ("D", "E")),
]


def is_topo(g, order):
    pos = {n: i for i, n in enumerate(order)}
    return len(order) == len(g) and all(pos[u] < pos[v] for u, v in g.edges())


def test_kahn_and_dfs_valid():
    g = DiGraph([("a", "b"), ("a", "c"), ("b", "d"), ("c", "d"), ("e", "c")])
    assert is_topo(g, kahn_order(g))
    assert is_topo(g, dfs_order(g))


def test_lexicographic_order():
    g = DiGraph([("b", "d"), ("a", "d"), ("c", "d")])
    assert kahn_order(g, key=lambda x: x) == ["a", "b", "c", "d"]


def test_cycle_detection():
    g = DiGraph([(1, 2), (2, 3), (3, 1), (3, 4)])
    cyc = find_cycle(g)
    assert cyc[0] == cyc[-1] and set(cyc) == {1, 2, 3}
    with pytest.raises(CycleError):
        kahn_order(g)
    with pytest.raises(CycleError):
        dfs_order(g)
    assert find_cycle(DiGraph([(1, 2)])) is None


def test_count_matches_enumeration_and_bruteforce():
    g = DiGraph([(0, 2), (1, 2), (2, 3), (1, 4)], nodes=range(5))
    orders = list(all_orders(g))
    brute = [p for p in itertools.permutations(range(5)) if is_topo(g, p)]
    assert len(orders) == len(brute) == count_orders(g)
    assert len({tuple(o) for o in orders}) == len(orders)


def test_count_orders_chain_and_antichain():
    assert count_orders(DiGraph([(i, i + 1) for i in range(7)])) == 1
    assert count_orders(DiGraph(nodes=range(6))) == math.factorial(6)
    assert count_orders(DiGraph([(1, 2), (2, 1)])) == 0


def test_levels():
    g = DiGraph([("a", "b"), ("b", "c"), ("a", "c")])
    assert longest_levels(g) == {"a": 0, "b": 1, "c": 2}


def test_cpm_classic():
    r = critical_path_method(CLASSIC)
    assert r.duration == 14
    assert r.critical_activities == ["A", "B", "D", "F"]
    assert r.total_float["C"] == 2 and r.total_float["E"] == 6
    assert r.es["D"] == 7 and r.lf["E"] == 12
    assert r.free_float["E"] == 6 and r.free_float["C"] == 0


def test_critical_paths_multiple():
    acts = [Activity("S", 1), Activity("X", 2, ("S",)), Activity("Y", 2, ("S",)), Activity("T", 1, ("X", "Y"))]
    assert sorted(critical_paths(acts)) == [["S", "X", "T"], ["S", "Y", "T"]]


def test_cpm_errors():
    with pytest.raises(CycleError):
        critical_path_method([Activity("a", 1, ("b",)), Activity("b", 1, ("a",))])
    with pytest.raises(KeyError):
        critical_path_method([Activity("a", 1, ("zz",))])
    with pytest.raises(ValueError):
        critical_path_method([Activity("a", -1)])


def test_sensitivity():
    assert crash_sensitivity(CLASSIC, "B", 2) == 2
    assert crash_sensitivity(CLASSIC, "C", 1) == 0


def test_float_property_random():
    acts = random_project(40, seed=3)
    r = critical_path_method(acts)
    assert all(v >= -1e-9 for v in r.total_float.values())
    assert r.critical_activities
    for a in acts:
        assert r.ef[a.name] <= r.lf[a.name] + 1e-9


def test_pert_basic():
    a = PertActivity("a", 1, 2, 9)
    assert a.mean == pytest.approx(3.0) and a.variance == pytest.approx((8 / 6) ** 2)
    with pytest.raises(ValueError):
        PertActivity("b", 3, 2, 1)
    res = pert_analysis([a, PertActivity("b", 2, 4, 6, ("a",))])
    assert res.expected == pytest.approx(7.0)
    assert res.prob_complete_by(7.0) == pytest.approx(0.5)
    assert res.prob_complete_by(res.deadline_for(0.9)) == pytest.approx(0.9, abs=1e-6)


def test_monte_carlo_matches_series_mean():
    acts = [PertActivity("a", 1, 2, 5), PertActivity("b", 2, 3, 8, ("a",))]
    s = monte_carlo_duration(acts, n=20000, seed=1)
    assert s.mean() == pytest.approx(pert_analysis(acts).expected, abs=0.05)
    assert np.array_equal(s, monte_carlo_duration(acts, n=20000, seed=1))


def test_merge_bias_parallel_paths():
    # Two parallel equal paths: simulated mean exceeds the PERT estimate
    acts = [PertActivity("a", 2, 4, 8), PertActivity("b", 2, 4, 8)]
    assert monte_carlo_duration(acts, 20000, 2).mean() > pert_analysis(acts).expected


def test_rcpsp_feasible_and_bounds():
    res = [Resource("R", 4), Resource("S", 3)]
    acts = random_project(30, seed=5, resources=("R", "S"))
    lb = resource_lower_bound(acts, res)
    for rule in ("lft", "lst", "mslk", "spt", "grd", "mts"):
        for fn in (schedule_serial, schedule_parallel):
            st = fn(acts, res, rule)
            assert validate_schedule(acts, res, st) == []
            assert max(st[a.name] + a.duration for a in acts) >= lb - 1e-9


def test_unconstrained_equals_cpm():
    acts = random_project(25, seed=8)
    st = schedule_serial(acts, [])
    assert max(st[a.name] + a.duration for a in acts) == critical_path_method(acts).duration


def test_resource_forces_serialisation():
    acts = [Activity("a", 3, (), {"R": 1}), Activity("b", 4, (), {"R": 1})]
    st = schedule_serial(acts, [Resource("R", 1)])
    assert max(st.values()) in (3, 4) and validate_schedule(acts, [Resource("R", 1)], st) == []
    stp = schedule_parallel(acts, [Resource("R", 1)])
    assert sorted(stp.values()) in ([0, 3], [0, 4])


def test_validator_detects_violations():
    acts = [Activity("a", 3, (), {"R": 2}), Activity("b", 2, ("a",), {"R": 2})]
    res = [Resource("R", 3)]
    assert validate_schedule(acts, res, {"a": 0, "b": 1})
    acts2 = [Activity("a", 3, (), {"R": 2}), Activity("b", 2, (), {"R": 2})]
    assert validate_schedule(acts2, res, {"a": 0, "b": 1})


def test_demand_exceeds_capacity():
    with pytest.raises(ValueError):
        schedule_serial([Activity("a", 1, (), {"R": 5})], [Resource("R", 2)])


def test_generator_deterministic_acyclic():
    a, b = random_project(50, seed=1), random_project(50, seed=1)
    assert a == b
    assert find_cycle(__import__("project_scheduler.cpm", fromlist=["x"]).build_graph(a)) is None
