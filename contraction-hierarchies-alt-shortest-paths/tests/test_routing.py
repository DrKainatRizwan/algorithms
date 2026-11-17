import math
import random

import pytest

from road_routing import (ALT, ContractionHierarchy, Graph, bidirectional_dijkstra, dijkstra,
                          dijkstra_all, grid_road_network, largest_scc_subgraph, random_pairs,
                          run_queries)


@pytest.fixture(scope="module")
def road():
    return largest_scc_subgraph(grid_road_network(14, 14, seed=3))


@pytest.fixture(scope="module")
def ch(road):
    return ContractionHierarchy(road)


@pytest.fixture(scope="module")
def alt(road):
    return ALT(road, k=6, seed=1)


def path_cost(g, p):
    return sum(g.out[a][b] for a, b in zip(p, p[1:]))


def test_edge_min_and_negative():
    g = Graph(3)
    g.add_edge(0, 1, 5)
    g.add_edge(0, 1, 2)
    assert g.out[0][1] == 2 and g.inc[1][0] == 2
    with pytest.raises(ValueError):
        g.add_edge(0, 2, -1)


def test_dijkstra_small():
    g = Graph(4)
    for u, v, w in [(0, 1, 1), (1, 2, 2), (0, 2, 5), (2, 3, 1)]:
        g.add_edge(u, v, w)
    r = dijkstra(g, 0, 3)
    assert r.dist == 4 and r.path == [0, 1, 2, 3]
    assert dijkstra(g, 3, 0).dist == math.inf


def test_scc_strongly_connected(road):
    d = dijkstra_all(road, 0)
    assert all(x < math.inf for x in d)
    assert all(x < math.inf for x in dijkstra_all(road.reversed(), 0))


def test_bidirectional_matches(road):
    for s, t in random_pairs(road, 40, seed=5):
        a, b = dijkstra(road, s, t), bidirectional_dijkstra(road, s, t)
        assert b.dist == pytest.approx(a.dist)
        assert path_cost(road, b.path) == pytest.approx(a.dist)
        assert b.path[0] == s and b.path[-1] == t


def test_bidirectional_unreachable():
    g = Graph(3)
    g.add_edge(0, 1, 1)
    assert bidirectional_dijkstra(g, 0, 2).dist == math.inf


def test_alt_exact(road, alt):
    for s, t in random_pairs(road, 40, seed=6):
        r = alt.query(s, t)
        assert r.dist == pytest.approx(dijkstra(road, s, t).dist)
        assert path_cost(road, r.path) == pytest.approx(r.dist)


def test_alt_heuristic_admissible(road, alt):
    t = 17
    exact = dijkstra_all(road.reversed(), t)
    for v in range(0, road.n, 7):
        assert alt.heuristic(v, t) <= exact[v] + 1e-9


def test_alt_settles_fewer(road, alt):
    pairs = random_pairs(road, 30, seed=7)
    base = run_queries(lambda s, t: dijkstra(road, s, t), pairs)
    fast = run_queries(alt.query, pairs)
    assert fast["settled"] < base["settled"]


def test_alt_bad_strategy(road):
    with pytest.raises(ValueError):
        ALT(road, strategy="nope")


def test_alt_random_landmarks_exact(road):
    a = ALT(road, k=4, strategy="random", seed=2)
    s, t = 3, road.n - 4
    assert a.query(s, t).dist == pytest.approx(dijkstra(road, s, t).dist)


def test_ch_ranks_permutation(ch, road):
    assert sorted(ch.rank) == list(range(road.n))


def test_ch_exact_distances(ch, road):
    for s, t in random_pairs(road, 80, seed=8):
        assert ch.query(s, t).dist == pytest.approx(dijkstra(road, s, t).dist)


def test_ch_path_unpacks_to_original_edges(ch, road):
    for s, t in random_pairs(road, 40, seed=9):
        r = ch.query(s, t)
        assert r.path[0] == s and r.path[-1] == t
        assert path_cost(road, r.path) == pytest.approx(r.dist)


def test_ch_smaller_search_space(ch, road):
    pairs = random_pairs(road, 40, seed=10)
    base = run_queries(lambda s, t: dijkstra(road, s, t), pairs)
    fast = run_queries(ch.query, pairs)
    assert fast["settled"] < base["settled"] / 2


def test_ch_directed_asymmetric():
    rng = random.Random(4)
    g = Graph(30)
    for i in range(30):
        g.add_edge(i, (i + 1) % 30, rng.uniform(1, 5))
    for _ in range(60):
        g.add_edge(rng.randrange(30), rng.randrange(30), rng.uniform(1, 9))
    ch = ContractionHierarchy(g)
    for s in range(0, 30, 3):
        ref = dijkstra_all(g, s)
        for t in range(30):
            assert ch.query(s, t).dist == pytest.approx(ref[t])


def test_ch_trivial_and_stats(ch):
    assert ch.query(5, 5).path == [5]
    st = ch.stats()
    assert st["shortcuts"] == ch.num_shortcuts and st["vertices"] == ch.n
