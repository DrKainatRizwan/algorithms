import itertools
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mst_suite import (ALGORITHMS, Graph, RollbackUnionFind, TreePaths, UnionFind, bottleneck,
                       boruvka, disconnected_graph, euclidean_complete_graph, grid_graph,
                       is_minimum_spanning_forest, is_spanning_forest, kruskal, prim,
                       random_connected_graph, replacement_edges, reverse_delete,
                       second_best_mst, sensitivity)
from mst_suite.benchmark import adversarial_chain_steps, unionfind_steps


def brute_force_weight(g: Graph) -> float:
    best = math.inf
    for combo in itertools.combinations(range(g.m), g.n - 1):
        if is_spanning_forest(g, list(combo)):
            best = min(best, g.total_weight(list(combo)))
    return best


@pytest.mark.parametrize("seed", range(6))
def test_matches_brute_force(seed):
    g = random_connected_graph(6, 10, seed=seed)
    ref = brute_force_weight(g)
    for fn in ALGORITHMS.values():
        assert fn(g).weight == pytest.approx(ref)
    assert reverse_delete(g).weight == pytest.approx(ref)


@pytest.mark.parametrize("seed", range(5))
def test_all_algorithms_same_edges_with_ties(seed):
    g = random_connected_graph(40, 150, seed=seed, integer_weights=4)
    sets = [tuple(fn(g).edges) for fn in ALGORITHMS.values()]
    assert len(set(sets)) == 1
    assert tuple(reverse_delete(g).edges) == sets[0]


def test_certificate_accepts_mst_rejects_heavier():
    g = random_connected_graph(50, 200, seed=3)
    t = kruskal(g)
    assert is_minimum_spanning_forest(g, t.edges)
    tp = TreePaths(g, t.edges)
    non_tree = next(i for i in range(g.m) if i not in set(t.edges))
    u, v, _ = g.edges[non_tree]
    swapped = [e for e in t.edges if e != tp.path_max(u, v)] + [non_tree]
    assert is_spanning_forest(g, swapped)
    assert not is_minimum_spanning_forest(g, swapped)


def test_forest_on_disconnected_graph():
    g = disconnected_graph([5, 8, 1, 6], seed=2)
    assert g.components() == 4
    for fn in ALGORITHMS.values():
        r = fn(g)
        assert r.components == 4 and len(r.edges) == g.n - 4
        assert is_minimum_spanning_forest(g, r.edges)


def test_parallel_edges_and_self_loops():
    g = Graph(3)
    g.add_edge(0, 0, 0.1)
    g.add_edge(0, 1, 5)
    g.add_edge(0, 1, 2)
    g.add_edge(1, 2, 3)
    g.add_edge(1, 2, 3)
    for fn in ALGORITHMS.values():
        r = fn(g)
        assert r.weight == 5 and r.edges == [2, 3]


def test_edge_cases_empty_and_single():
    for n in (0, 1):
        g = Graph(n)
        for fn in ALGORITHMS.values():
            r = fn(g)
            assert r.edges == [] and r.weight == 0


def test_euclidean_and_grid():
    g, pts = euclidean_complete_graph(30, seed=1)
    assert g.m == 30 * 29 // 2
    assert prim(g).weight == pytest.approx(ALGORITHMS["prim_dense"](g).weight)
    gg = grid_graph(6, 7, seed=4)
    assert len(boruvka(gg).edges) == 41
    assert boruvka(gg).stats["rounds"] <= math.ceil(math.log2(42))


@pytest.mark.parametrize("union", ["naive", "size", "rank"])
@pytest.mark.parametrize("comp", ["none", "full", "halving", "splitting"])
def test_unionfind_variants_agree(union, comp):
    uf = UnionFind(30, union, comp)
    ref = UnionFind(30)
    pairs = [(i * 7 % 30, i * 11 % 30) for i in range(40)]
    for a, b in pairs:
        assert uf.union(a, b) == ref.union(a, b)
    assert uf.components == ref.components
    for a in range(30):
        for b in range(0, 30, 5):
            assert uf.connected(a, b) == ref.connected(a, b)
    assert kruskal(random_connected_graph(30, 80, 1), union, comp).weight == pytest.approx(
        kruskal(random_connected_graph(30, 80, 1)).weight)


def test_unionfind_invalid_rule():
    with pytest.raises(ValueError):
        UnionFind(3, "bogus")
    with pytest.raises(ValueError):
        UnionFind(3, "rank", "bogus")


def test_heuristics_reduce_work():
    none = adversarial_chain_steps(300, "naive", "none")
    best = adversarial_chain_steps(300, "rank", "full")
    assert best < none / 20
    uf = UnionFind(1024, "rank", "none")
    for i in range(1023):
        uf.union(i, i + 1)
    assert uf.max_depth() <= 10  # log2(1024)


def test_rollback_union_find():
    uf = RollbackUnionFind(6)
    uf.union(0, 1); uf.union(2, 3); uf.union(1, 3); uf.union(0, 2)
    assert uf.components == 3 and uf.find(0) == uf.find(3)
    uf.rollback()  # no-op union
    uf.rollback()
    assert uf.find(0) != uf.find(3) and uf.components == 4
    uf.rollback(); uf.rollback()
    assert uf.components == 6


def test_bottleneck_matches_bruteforce_minimax():
    g = random_connected_graph(12, 25, seed=9)
    t = kruskal(g).edges
    # minimax via Floyd-Warshall variant
    n = g.n
    d = [[math.inf] * n for _ in range(n)]
    for i in range(n):
        d[i][i] = 0
    for u, v, w in g.edges:
        d[u][v] = d[v][u] = min(d[u][v], w)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                d[i][j] = min(d[i][j], max(d[i][k], d[k][j]))
    for i in range(n):
        for j in range(n):
            assert bottleneck(g, t, i, j) == pytest.approx(d[i][j])


def test_second_best_matches_bruteforce():
    for seed in range(4):
        g = random_connected_graph(6, 10, seed=seed)
        t = kruskal(g)
        sb = second_best_mst(g, t.edges)
        weights = sorted(g.total_weight(list(c)) for c in itertools.combinations(range(g.m), g.n - 1)
                         if is_spanning_forest(g, list(c)))
        second = next(w for w in weights if w > weights[0] + 1e-12)
        assert sb.weight == pytest.approx(second)


def test_replacement_edges_and_sensitivity():
    g = random_connected_graph(25, 60, seed=5)
    t = kruskal(g)
    rep = replacement_edges(g, t.edges)
    sens = sensitivity(g, t.edges)
    for e in t.edges:
        rest = [x for x in t.edges if x != e]
        cands = [i for i in range(g.m) if i not in set(t.edges)
                 and is_spanning_forest(g, rest + [i])]
        if cands:
            best = min(g.edges[i][2] for i in cands)
            assert g.edges[rep[e]][2] == pytest.approx(best) == pytest.approx(sens[e])
        else:
            assert rep[e] is None and sens[e] == math.inf


def test_bridge_has_infinite_sensitivity():
    g = Graph(4)
    for u, v, w in [(0, 1, 1), (1, 2, 2), (2, 0, 3), (2, 3, 4)]:
        g.add_edge(u, v, w)
    t = kruskal(g)
    s = sensitivity(g, t.edges)
    assert s[3] == math.inf and s[1] == 3


def test_unionfind_steps_table():
    g = random_connected_graph(200, 800, seed=2)
    table = unionfind_steps(g)
    assert len(table) == 12
    assert table[("rank", "full")] <= table[("naive", "none")]
