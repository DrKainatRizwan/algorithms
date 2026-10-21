import random

from graph_sat import (Digraph, tarjan_scc, kosaraju_scc, gabow_scc, condensation,
                       random_digraph, transitive_closure, transitive_reduction_dag,
                       ReachabilityIndex)
from graph_sat.scc import same_partition, components, min_edges_to_strongly_connect
from graph_sat.applications import (orient_edges_for_strong_connectivity,
                                    is_strongly_connected, kernel_graph_summary)


def brute_scc(g):
    reach = [g.reachable_from(u) for u in range(g.n)]
    comp, k = [-1] * g.n, 0
    for u in range(g.n):
        if comp[u] == -1:
            for v in range(g.n):
                if v in reach[u] and u in reach[v]:
                    comp[v] = k
            k += 1
    return comp


def test_known_graph():
    g = Digraph(8, [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5), (5, 3), (6, 5), (6, 7), (7, 6)])
    for f in (tarjan_scc, kosaraju_scc, gabow_scc):
        comps = sorted(sorted(c) for c in components(f(g)))
        assert comps == [[0, 1, 2], [3, 4, 5], [6, 7]]


def test_algorithms_agree_with_brute_force():
    for seed in range(40):
        n = 2 + seed % 12
        g = random_digraph(n, random.Random(seed).randrange(0, 3 * n), seed)
        ref = brute_scc(g)
        for f in (tarjan_scc, kosaraju_scc, gabow_scc):
            assert same_partition(f(g), ref)


def test_deep_path_no_recursion_error():
    n = 50000
    g = Digraph(n, [(i, i + 1) for i in range(n - 1)] + [(n - 1, 0)])
    for f in (tarjan_scc, kosaraju_scc, gabow_scc):
        assert max(f(g)) == 0


def test_tarjan_reverse_topological_numbering():
    g = random_digraph(60, 120, 5)
    dag, comp = condensation(g, tarjan_scc(g))
    assert all(u > v for u, v in dag.edges())


def test_kosaraju_topological_numbering():
    g = random_digraph(60, 120, 6)
    dag, _ = condensation(g, kosaraju_scc(g))
    assert all(u < v for u, v in dag.edges())


def test_condensation_is_dag_and_deduplicated():
    g = random_digraph(80, 300, 7)
    dag, _ = condensation(g)
    assert dag.is_dag()
    assert len(set(dag.edges())) == dag.m


def test_transitive_closure_matches_bfs():
    g = random_digraph(40, 70, 11)
    cl = transitive_closure(g)
    for u in range(g.n):
        expect = sum(1 << v for v in g.reachable_from(u))
        assert cl[u] == expect


def test_transitive_reduction_preserves_reachability():
    rng = random.Random(3)
    dag = Digraph(25)
    for _ in range(90):
        a, b = sorted(rng.sample(range(25), 2))
        dag.add_edge(a, b)
    red = transitive_reduction_dag(dag)
    assert red.m <= len(set(dag.edges()))
    assert transitive_closure(red) == transitive_closure(dag)
    # minimality: removing any edge changes reachability
    base = transitive_closure(red)
    for u, v in list(red.edges()):
        red.adj[u].remove(v)
        assert transitive_closure(red) != base
        red.adj[u].append(v)


def test_reachability_index():
    g = Digraph(5, [(0, 1), (1, 0), (1, 2), (3, 4)])
    idx = ReachabilityIndex(g)
    assert idx.reaches(0, 2) and not idx.reaches(2, 0)
    assert idx.mutually_reachable(0, 1)
    assert idx.reachable_count(0) == 3


def test_min_edges_to_strongly_connect():
    assert min_edges_to_strongly_connect(Digraph(1)) == 0
    assert min_edges_to_strongly_connect(Digraph(4, [(0, 1), (1, 2), (2, 3)])) == 1
    assert min_edges_to_strongly_connect(Digraph(3)) == 3
    g = Digraph(4, [(0, 1), (0, 2), (0, 3)])
    assert min_edges_to_strongly_connect(g) == 3
    g.add_edge(3, 0)
    assert min_edges_to_strongly_connect(g) == 2


def test_robbins_orientation():
    cycle = [(i, (i + 1) % 6) for i in range(6)] + [(0, 3)]
    ori = orient_edges_for_strong_connectivity(6, cycle)
    assert ori is not None and is_strongly_connected(Digraph(6, ori))
    assert orient_edges_for_strong_connectivity(4, [(0, 1), (1, 2), (2, 0), (2, 3)]) is None


def test_summary():
    g = Digraph(4, [(0, 1), (1, 0), (1, 2)])
    s = kernel_graph_summary(g)
    assert s["components"] == 3 and s["largest"] == 2 and s["dag_edges"] == 1
