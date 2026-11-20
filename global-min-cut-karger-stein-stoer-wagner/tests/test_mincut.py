import numpy as np
import pytest

from global_mincut import (Graph, GomoryHuTree, brute_force_min_cut,
                           complete_graph, count_min_cuts_estimate, cut_value,
                           cycle_graph, karger, karger_contract, karger_stein,
                           max_flow, planted_cut_graph, random_graph,
                           recommended_trials, stoer_wagner, stoer_wagner_heap)


def test_graph_merges_parallel_edges_and_drops_loops():
    g = Graph(3)
    g.add_edge(0, 1, 2)
    g.add_edge(1, 0, 3)
    g.add_edge(2, 2, 5)
    assert g.w[0, 1] == 5 and g.num_edges() == 1 and g.total_weight() == 5


def test_negative_weight_rejected():
    with pytest.raises(ValueError):
        Graph(2).add_edge(0, 1, -1)


def test_cut_value_matches_manual():
    g = Graph.from_edges(4, [(0, 1, 1), (1, 2, 4), (2, 3, 1), (0, 3, 2)])
    assert cut_value(g, {0, 1}) == 6
    assert cut_value(g, {0}) == 3


def test_cycle_min_cut_is_two():
    g = cycle_graph(9)
    assert stoer_wagner(g).value == 2
    assert stoer_wagner_heap(g).value == 2


def test_complete_graph_min_cut_is_n_minus_one():
    assert stoer_wagner(complete_graph(7)).value == 6


@pytest.mark.parametrize("seed", range(8))
def test_stoer_wagner_matches_brute_force(seed):
    g = random_graph(9, 0.4, seed, 1, 6)
    bf = brute_force_min_cut(g)
    for fn in (stoer_wagner, stoer_wagner_heap):
        r = fn(g)
        assert r.value == pytest.approx(bf.value)
        assert cut_value(g, r.side) == pytest.approx(r.value)


def test_heap_variant_handles_disconnected_graph():
    g = Graph.from_edges(5, [(0, 1, 2), (1, 2, 2), (3, 4, 1)])
    assert stoer_wagner_heap(g).value == 0
    assert stoer_wagner(g).value == 0


def test_planted_cut_recovered():
    g, planted = planted_cut_graph(10, 10, 0.7, 3, seed=1)
    r = stoer_wagner(g)
    assert r.value == planted
    assert set(r.side) in (set(range(10)), set(range(10, 20)))


def test_karger_cut_is_valid_and_upper_bounds_optimum():
    g = random_graph(12, 0.5, 3, 1, 4)
    opt = stoer_wagner(g).value
    rng = np.random.default_rng(0)
    for _ in range(20):
        r = karger_contract(g, rng)
        assert r.value >= opt - 1e-9
        assert cut_value(g, r.side) == pytest.approx(r.value)


def test_karger_finds_optimum_with_enough_trials():
    g = random_graph(10, 0.4, 5, 1, 3)
    opt = stoer_wagner(g).value
    assert karger(g, recommended_trials(10, 1e-4), seed=2).value == opt


def test_karger_stein_finds_optimum_and_is_consistent():
    g, planted = planted_cut_graph(15, 15, 0.5, 2, seed=4)
    r = karger_stein(g, seed=1)
    assert r.value == planted
    assert cut_value(g, r.side) == pytest.approx(r.value)


def test_karger_stein_deterministic_given_seed():
    g = random_graph(20, 0.3, 9, 1, 5)
    assert karger_stein(g, seed=7) == karger_stein(g, seed=7)


def test_recommended_trials_scaling():
    assert recommended_trials(20) > recommended_trials(10) > 1


def test_max_flow_simple_network():
    g = Graph.from_edges(4, [(0, 1, 3), (0, 2, 2), (1, 2, 1), (1, 3, 2), (2, 3, 3)])
    f, side = max_flow(g, 0, 3)
    assert f == 5
    assert cut_value(g, side) == f and 0 in side and 3 not in side


@pytest.mark.parametrize("seed", range(4))
def test_gomory_hu_pairwise_cuts_match_max_flow(seed):
    g = random_graph(8, 0.4, seed, 1, 5)
    tree = GomoryHuTree(g)
    for s in range(8):
        for t in range(s + 1, 8):
            assert tree.min_cut_value(s, t) == pytest.approx(max_flow(g, s, t)[0])


def test_gomory_hu_global_min_cut():
    g = random_graph(10, 0.35, 11, 1, 4)
    r = GomoryHuTree(g).global_min_cut()
    assert r.value == pytest.approx(stoer_wagner(g).value)
    assert cut_value(g, r.side) == pytest.approx(r.value)


def test_min_cut_histogram_on_cycle_is_bounded():
    g = cycle_graph(6)
    hist = count_min_cuts_estimate(g, 400, seed=0)
    assert 1 < len(hist) <= 15  # n(n-1)/2 bound for min cuts
