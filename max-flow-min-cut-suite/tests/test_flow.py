import itertools
import pytest

from flow_suite import (SOLVERS, FlowNetwork, min_cut, verify_certificate, cut_capacity,
                        random_network, layered_network, bipartite_matching,
                        matching_from_flow, edge_disjoint_paths, dinic)

NAMES = list(SOLVERS)


def clrs():
    g = FlowNetwork(6)
    for u, v, c in [(0, 1, 16), (0, 2, 13), (1, 2, 10), (2, 1, 4), (1, 3, 12),
                    (3, 2, 9), (2, 4, 14), (4, 3, 7), (3, 5, 20), (4, 5, 4)]:
        g.add_edge(u, v, c)
    return g


@pytest.mark.parametrize("name", NAMES)
def test_clrs_example(name):
    g = clrs()
    assert SOLVERS[name](g, 0, 5) == 23
    assert g.check_flow(0, 5) == 23


@pytest.mark.parametrize("name", NAMES)
def test_disconnected_zero(name):
    g = FlowNetwork(4)
    g.add_edge(0, 1, 5)
    g.add_edge(2, 3, 5)
    assert SOLVERS[name](g, 0, 3) == 0


@pytest.mark.parametrize("name", NAMES)
def test_parallel_edges_and_antiparallel(name):
    g = FlowNetwork(3)
    g.add_edge(0, 1, 3)
    g.add_edge(0, 1, 4)
    g.add_edge(1, 0, 2)
    g.add_edge(1, 2, 5)
    assert SOLVERS[name](g, 0, 2) == 5


def test_solvers_agree_on_random_graphs():
    for seed in range(25):
        n = 6 + seed % 10
        vals = []
        for name in NAMES:
            g, s, t = random_network(n, 3 * n, 20, seed)
            v = SOLVERS[name](g, s, t)
            assert g.check_flow(s, t) == v
            vals.append(v)
        assert len(set(vals)) == 1, (seed, vals)


def test_layered_agreement():
    vals = set()
    for name in NAMES:
        g, s, t = layered_network(4, 5, 30, seed=3)
        vals.add(SOLVERS[name](g, s, t))
    assert len(vals) == 1


def test_max_flow_equals_min_cut_certificate():
    for seed in range(15):
        g, s, t = random_network(12, 40, 15, seed)
        cut = min_cut(g, s, t)
        assert verify_certificate(g, s, t, cut)
        assert cut_capacity(g, cut.source_side) == cut.value
        assert sum(c for _, _, c in cut.cut_edges) == cut.value


def test_min_cut_matches_brute_force():
    g, s, t = random_network(8, 18, 9, seed=11)
    best = None
    others = [v for v in range(g.n) if v not in (s, t)]
    for r in range(len(others) + 1):
        for comb in itertools.combinations(others, r):
            c = cut_capacity(g, {s, *comb})
            best = c if best is None else min(best, c)
    assert min_cut(g, s, t).value == best


def test_min_cut_does_not_mutate_input():
    g = clrs()
    before = list(g.cap)
    min_cut(g, 0, 5, solver=SOLVERS["push_relabel"])
    assert g.cap == before


def test_bipartite_matching():
    pairs = [(0, 0), (0, 1), (1, 0), (2, 1), (2, 2), (3, 2)]
    for name in NAMES:
        g, s, t = bipartite_matching(4, 3, pairs)
        assert SOLVERS[name](g, s, t) == 3
        m = matching_from_flow(g, 4, 3)
        assert len(m) == 3
        assert len({i for i, _ in m}) == 3 and len({j for _, j in m}) == 3
        assert all(p in pairs for p in m)


def test_edge_disjoint_paths():
    edges = [(0, 1), (0, 2), (1, 3), (2, 3), (1, 2), (3, 4), (3, 4)]
    assert edge_disjoint_paths(5, edges, 0, 4) == 2


def test_validation_errors():
    g = FlowNetwork(2)
    with pytest.raises(ValueError):
        g.add_edge(0, 5, 1)
    with pytest.raises(ValueError):
        g.add_edge(0, 1, -1)
    with pytest.raises(ValueError):
        dinic(g, 0, 0)
    with pytest.raises(ValueError):
        FlowNetwork(0)


def test_reset_allows_resolve():
    g = clrs()
    assert dinic(g, 0, 5) == 23
    g.reset()
    assert dinic(g, 0, 5) == 23
