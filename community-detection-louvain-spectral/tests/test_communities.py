import numpy as np
import pytest

from graph_communities import (Graph, adjusted_rand_index, aggregate, conductance,
                               disconnected_communities, edge_betweenness, eigengap_k,
                               girvan_newman, label_propagation, louvain, modularity,
                               nmi, normalized_laplacian, planted_partition,
                               ring_of_cliques, spectral_clustering,
                               two_triangles_bridge)


def test_degree_and_total_weight_with_loop():
    g = Graph(2)
    g.add_edge(0, 1, 2.0)
    g.add_edge(0, 0, 1.5)
    assert g.degree(0) == pytest.approx(5.0)
    assert g.total_weight() == pytest.approx(3.5)


def test_modularity_known_value():
    g, truth = two_triangles_bridge()
    # in = 3 each, m = 7, tot = 7 each -> 2*(3/7 - 0.25)
    assert modularity(g, truth) == pytest.approx(2 * (3 / 7 - 0.25))
    assert modularity(g, np.zeros(6, dtype=int)) == pytest.approx(0.0)


def test_modularity_matches_matrix_formula():
    g, _ = planted_partition([10, 10], 0.6, 0.1, seed=3)
    lab = np.array([0] * 10 + [1] * 10)
    a = g.adjacency_matrix()
    k = a.sum(1)
    m2 = k.sum()
    same = lab[:, None] == lab[None, :]
    q = ((a - np.outer(k, k) / m2) * same).sum() / m2
    assert modularity(g, lab) == pytest.approx(q)


def test_nmi_ari_properties():
    a = [0, 0, 1, 1, 2, 2]
    b = [5, 5, 7, 7, 9, 9]
    assert nmi(a, b) == pytest.approx(1.0)
    assert adjusted_rand_index(a, b) == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    x, y = rng.integers(0, 4, 400), rng.integers(0, 4, 400)
    assert abs(adjusted_rand_index(x, y)) < 0.05
    assert nmi(x, y) < 0.05


def test_conductance():
    g, _ = two_triangles_bridge()
    assert conductance(g, [0, 1, 2]) == pytest.approx(1 / 7)


def test_louvain_recovers_planted_partition():
    g, truth = planted_partition([25, 25, 25, 25], 0.5, 0.02, seed=1)
    res = louvain(g, seed=0)
    assert nmi(res.labels, truth) > 0.95
    assert res.modularity == pytest.approx(modularity(g, res.labels))
    assert res.modularity >= modularity(g, truth) - 1e-9


def test_louvain_preserves_modularity_under_aggregation():
    g, _ = planted_partition([15, 15, 15], 0.5, 0.05, seed=2)
    rng = np.random.default_rng(0)
    from graph_communities import local_moving
    comm, _ = local_moving(g, rng)
    h = aggregate(g, comm)
    assert h.total_weight() == pytest.approx(g.total_weight())
    assert modularity(h, np.arange(h.n)) == pytest.approx(modularity(g, comm))


def test_louvain_deterministic_and_hierarchy():
    g, _ = planted_partition([20, 20, 20], 0.5, 0.03, seed=4)
    r1, r2 = louvain(g, seed=7), louvain(g, seed=7)
    assert np.array_equal(r1.labels, r2.labels)
    qs = [modularity(g, lv) for lv in r1.levels]
    assert all(b >= a - 1e-9 for a, b in zip(qs, qs[1:]))


def test_louvain_edgeless_graph():
    res = louvain(Graph(4))
    assert len(set(res.labels)) == 4 and res.modularity == 0.0


def test_resolution_limit_and_resolution_parameter():
    g, truth = ring_of_cliques(30, 5)
    merged = louvain(g, seed=0).labels
    assert len(set(merged)) < 30                     # cliques are merged
    fine = louvain(g, resolution=3.0, seed=0).labels
    assert nmi(fine, truth) > nmi(merged, truth)


def test_ring_of_cliques_small_exact():
    g, truth = ring_of_cliques(6, 6)
    assert nmi(louvain(g, seed=0).labels, truth) == pytest.approx(1.0)


def test_label_propagation_two_cliques():
    g, _ = two_triangles_bridge()
    lab = label_propagation(g, seed=0)
    for u in range(g.n):                 # fixed point: own label is a plurality label
        counts = np.bincount([lab[v] for v in g.adj[u]], minlength=lab.max() + 1)
        assert counts[lab[u]] == counts.max()
    g2, t2 = planted_partition([30, 30], 0.7, 0.01, seed=5)
    assert nmi(label_propagation(g2, seed=1), t2) > 0.9


def test_spectral_clustering_and_eigengap():
    g, truth = planted_partition([20, 20, 20], 0.6, 0.02, seed=6)
    assert eigengap_k(g) == 3
    assert nmi(spectral_clustering(g, 3, seed=0), truth) > 0.95
    assert nmi(spectral_clustering(g, seed=0), truth) > 0.95


def test_laplacian_spectrum_bounds():
    g, _ = planted_partition([10, 10], 0.5, 0.1, seed=0)
    vals = np.linalg.eigvalsh(normalized_laplacian(g))
    assert vals[0] == pytest.approx(0.0, abs=1e-9) and vals[-1] <= 2 + 1e-9


def test_edge_betweenness_bridge_is_max():
    g, _ = two_triangles_bridge()
    eb = edge_betweenness(g)
    assert eb[(2, 3)] == pytest.approx(9.0)
    assert max(eb, key=eb.get) == (2, 3)


def test_girvan_newman_splits_triangles():
    g, truth = two_triangles_bridge()
    labels, q, trace = girvan_newman(g)
    assert nmi(labels, truth) == pytest.approx(1.0)
    assert q == pytest.approx(modularity(g, truth))
    assert trace[0][0] == 1


def test_disconnected_community_detection():
    g = Graph.from_edges(4, [(0, 1), (2, 3)])
    assert disconnected_communities(g, [0, 0, 0, 0]) == [0]
    assert disconnected_communities(g, [0, 0, 1, 1]) == []
