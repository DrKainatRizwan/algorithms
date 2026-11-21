import numpy as np
import pytest

from hnsw_index import (BruteForceIndex, HNSWIndex, benchmark_index,
                        cosine_dist, gaussian_clusters, get_metric, l2_sq,
                        recall_at_k, uniform_cube)
from hnsw_index.analysis import degree_stats, greedy_hops, reciprocal_fraction
from hnsw_index.persist import load_index, save_index


@pytest.fixture(scope="module")
def built():
    X = gaussian_clusters(1200, 16, clusters=12, seed=1)
    idx = HNSWIndex(16, M=12, ef_construction=80, seed=2)
    idx.add_batch(X)
    Q = gaussian_clusters(60, 16, clusters=12, seed=1)[:60] + 0.05
    return X, Q, idx


def test_metrics_match_numpy():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 5)); q = rng.standard_normal(5)
    assert np.allclose(l2_sq(q, X), ((X - q) ** 2).sum(1))
    c = X @ q / (np.linalg.norm(X, axis=1) * np.linalg.norm(q))
    assert np.allclose(cosine_dist(q, X), 1 - c)
    with pytest.raises(ValueError):
        get_metric("nope")


def test_bruteforce_sorted_and_correct():
    X = uniform_cube(200, 4, seed=3)
    bf = BruteForceIndex(); bf.add(X)
    ids, d = bf.query(X[7], 5)
    assert ids[0] == 7 and d[0] == 0
    assert np.all(np.diff(d) >= 0)
    full = np.argsort(l2_sq(X[7], X), kind="stable")[:5]
    assert list(ids) == list(full)


def test_single_point_and_empty():
    idx = HNSWIndex(3)
    assert idx.query(np.zeros(3), 3)[0].size == 0
    idx.add(np.ones(3))
    ids, d = idx.query(np.zeros(3), 3)
    assert list(ids) == [0] and d[0] == pytest.approx(3.0)


def test_dimension_validation():
    with pytest.raises(ValueError):
        HNSWIndex(3).add(np.ones(4))
    with pytest.raises(ValueError):
        HNSWIndex(3, M=1)


def test_exact_self_match(built):
    X, _, idx = built
    for i in (0, 100, 999):
        ids, d = idx.query(X[i], 1, ef=50)
        assert ids[0] == i and d[0] == 0


def test_high_recall(built):
    X, Q, idx = built
    bf = BruteForceIndex(); bf.add(X)
    truth = bf.batch_query(Q, 10)
    found = np.array([idx.query(q, 10, ef=100)[0] for q in Q])
    assert recall_at_k(found, truth) > 0.95


def test_recall_monotone_in_ef(built):
    X, Q, idx = built
    rows = benchmark_index(idx, X, Q, 10, [10, 40, 160])
    r = [x["recall"] for x in rows]
    assert r[0] <= r[1] + 0.02 and r[1] <= r[2] + 0.02
    assert rows[2]["dist_evals"] > rows[0]["dist_evals"]


def test_sublinear_distance_evals(built):
    X, Q, idx = built
    rows = benchmark_index(idx, X, Q, 10, [50])
    assert rows[0]["dist_evals"] < 0.5 * len(X)


def test_degree_bounds_and_connectivity(built):
    _, _, idx = built
    stats = degree_stats(idx)
    assert stats[0]["max"] <= idx.M0
    for layer in range(1, idx.max_level + 1):
        assert stats[layer]["max"] <= idx.M
    assert idx.is_connected_layer0()
    assert 0 < reciprocal_fraction(idx) <= 1


def test_level_distribution_geometric(built):
    _, _, idx = built
    h = idx.level_histogram()
    assert h[0] > 0.8 * len(idx)          # ~ 1 - 1/M of nodes only on layer 0
    assert idx.level_of(idx.entry) == idx.max_level


def test_determinism():
    X = uniform_cube(300, 6, seed=5)
    a = HNSWIndex(6, seed=9); a.add_batch(X)
    b = HNSWIndex(6, seed=9); b.add_batch(X)
    assert a._links == b._links and a.entry == b.entry


def test_delete_hides_nodes(built):
    X = uniform_cube(400, 5, seed=11)
    idx = HNSWIndex(5, seed=1); idx.add_batch(X)
    idx.delete(10)
    ids, _ = idx.query(X[10], 5, ef=60)
    assert 10 not in ids and len(ids) == 5 and len(idx) == 399
    with pytest.raises(IndexError):
        idx.delete(10_000)


def test_cosine_metric_recall():
    X = np.random.default_rng(4).standard_normal((600, 12))
    idx = HNSWIndex(12, metric="cosine", seed=0); idx.add_batch(X)
    bf = BruteForceIndex("cosine"); bf.add(X)
    Q = X[:30] + 0.1
    found = np.array([idx.query(q, 5, ef=80)[0] for q in Q])
    assert recall_at_k(found, bf.batch_query(Q, 5)) > 0.9


def test_persistence_roundtrip(tmp_path, built):
    X, Q, idx = built
    p = str(tmp_path / "i.npz")
    save_index(idx, p)
    again = load_index(p)
    assert again._links == idx._links and again.entry == idx.entry
    for q in Q[:5]:
        assert list(again.query(q, 5)[0]) == list(idx.query(q, 5)[0])


def test_heuristic_selection_helps_on_clusters():
    X = gaussian_clusters(1500, 16, clusters=15, spread=0.8, seed=6)
    Q = gaussian_clusters(50, 16, clusters=15, spread=0.8, seed=6) + 0.1
    bf = BruteForceIndex(); bf.add(X)
    truth = bf.batch_query(Q, 10)
    rec = {}
    for heur in (True, False):
        idx = HNSWIndex(16, M=8, ef_construction=60, heuristic=heur, seed=1)
        idx.add_batch(X)
        rec[heur] = recall_at_k(np.array([idx.query(q, 10, ef=50)[0] for q in Q]), truth)
    assert rec[True] > rec[False]
    assert rec[True] > 0.95


def test_greedy_hops_positive_and_bounded(built):
    X, _, idx = built
    h = greedy_hops(idx, X[5] + 0.01)
    assert 0 <= h < 100
