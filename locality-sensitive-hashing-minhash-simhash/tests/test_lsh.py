import numpy as np
import pytest

from lsh_toolkit import (HyperplaneHasher, MinHasher, MinHashLSHIndex, PStableHasher,
                         SimHashIndex, SimHasher, VectorLSHIndex, char_shingles,
                         collision_probability, exact_knn, fnv1a_64, hamming, jaccard,
                         make_clusters, make_documents, optimal_bands, recall_at_k,
                         word_shingles)


def test_fnv_known_values():
    assert fnv1a_64("") == 0xCBF29CE484222325
    assert fnv1a_64("a") == 0xAF63DC4C8601EC8C


def test_shingles_and_jaccard():
    a, b = char_shingles("hello world", 3), char_shingles("hello  WORLD", 3)
    assert a == b and jaccard(a, b) == 1.0
    assert jaccard({1, 2, 3}, {2, 3, 4}) == 0.5
    assert jaccard(set(), set()) == 1.0
    assert len(word_shingles("a b c d e", 3)) == 3
    with pytest.raises(ValueError):
        char_shingles("x", 0)


def test_minhash_estimates_jaccard():
    mh = MinHasher(512, seed=1)
    a, b = set(range(0, 600)), set(range(300, 900))
    est = mh.estimate(mh.signature(a), mh.signature(b))
    assert abs(est - jaccard(a, b)) < 4 * mh.standard_error(jaccard(a, b))


def test_minhash_identical_disjoint_and_merge():
    mh = MinHasher(128, seed=2)
    a, b = set(range(100)), set(range(1000, 1100))
    sa, sb = mh.signature(a), mh.signature(b)
    assert mh.estimate(sa, mh.signature(a)) == 1.0
    assert mh.estimate(sa, sb) < 0.05
    assert np.array_equal(mh.merge(sa, sb), mh.signature(a | b))


def test_minhash_deterministic_and_empty():
    s1 = MinHasher(64, seed=5).signature({1, 2, 3})
    s2 = MinHasher(64, seed=5).signature([3, 2, 1, 1])
    assert np.array_equal(s1, s2)
    assert MinHasher(8).signature([]).shape == (8,)


def test_scurve_and_optimal_bands():
    assert collision_probability(0.0, 20, 5) == 0.0
    assert collision_probability(1.0, 20, 5) == 1.0
    b, r = optimal_bands(128, 0.8)
    assert b * r <= 128
    # the S-curve midpoint should be near the requested threshold
    assert abs((1 / b) ** (1 / r) - 0.8) < 0.15
    b2, r2 = optimal_bands(128, 0.4)
    assert (1 / b2) ** (1 / r2) < (1 / b) ** (1 / r)


def test_minhash_index_finds_near_duplicates():
    docs, group = make_documents(30, 2, 0.03, seed=3)
    mh = MinHasher(128, seed=0)
    idx = MinHashLSHIndex.for_threshold(mh, 0.6)
    sh = [char_shingles(d, 5) for d in docs]
    for i, s in enumerate(sh):
        idx.insert(i, s)
    assert len(idx) == len(docs)
    hits = 0
    total = 0
    for i, g in enumerate(group):
        res = {k for k, _ in idx.query(sh[i], 0.5)}
        mates = {j for j, gj in enumerate(group) if gj == g}
        hits += len(res & mates)
        total += len(mates)
        # unrelated documents should essentially never be returned
        assert not any(group[j] != g for j in res)
    assert hits / total > 0.95
    with pytest.raises(KeyError):
        idx.insert(0, sh[0])


def test_minhash_index_param_validation():
    with pytest.raises(ValueError):
        MinHashLSHIndex(MinHasher(16), 5, 5)


def test_simhash_similar_vs_different():
    sh = SimHasher(64)
    docs, group = make_documents(2, 1, 0.02, length=300, seed=7)
    f = [sh.text_fingerprint(d) for d in docs]
    assert hamming(f[0], f[1]) < hamming(f[0], f[2])
    assert sh.similarity(f[0], f[0]) == 1.0


def test_simhash_index_is_exact():
    rng = np.random.default_rng(0)
    fps = [int(x) for x in rng.integers(0, 2**63, size=500, dtype=np.int64)]
    base = fps[10]
    for flips in ([1], [3, 40], [5, 6, 7]):
        v = base
        for b in flips:
            v ^= 1 << b
        fps.append(v)
    idx = SimHashIndex(64, k=3)
    for f in fps:
        idx.add(f)
    res = idx.query(base)
    brute = sorted((i, hamming(base, f)) for i, f in enumerate(fps) if hamming(base, f) <= 3)
    assert sorted(res) == brute and len(res) >= 4


def test_simhash_index_validation():
    with pytest.raises(ValueError):
        SimHashIndex(8, k=8)
    with pytest.raises(ValueError):
        SimHasher(65)


def test_hyperplane_collision_matches_theory():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(50)
    y = x + 0.8 * rng.standard_normal(50)
    h = HyperplaneHasher(50, 4000, seed=1)
    emp = 1 - np.mean(h.bits(x) != h.bits(y))
    assert abs(emp - h.bit_collision_probability(x, y)) < 0.03
    cos = x @ y / np.linalg.norm(x) / np.linalg.norm(y)
    assert abs(h.estimate_cosine(x, y) - cos) < 0.06


def test_pstable_probability_monotone_and_empirical():
    h = PStableHasher(30, 5000, width=4.0, seed=2)
    assert h.collision_probability(1) > h.collision_probability(3) > h.collision_probability(8)
    rng = np.random.default_rng(1)
    x = rng.standard_normal(30)
    d = rng.standard_normal(30)
    y = x + 3.0 * d / np.linalg.norm(d)
    emp = np.mean(h.codes(x) == h.codes(y))
    assert abs(emp - h.collision_probability(3.0)) < 0.03


@pytest.mark.parametrize("metric,kw", [("cosine", {}), ("l2", {"width": 6.0})])
def test_vector_index_recall(metric, kw):
    data = make_clusters(1500, 24, seed=4)
    idx = VectorLSHIndex(24, num_tables=14, hashes_per_table=6 if metric == "cosine" else 4,
                         metric=metric, seed=1, **kw).fit(data)
    rng = np.random.default_rng(9)
    recs, cands = [], []
    for qi in rng.choice(len(data), 40, replace=False):
        q = data[qi] + 0.05 * rng.standard_normal(24)
        ids, _, nc = idx.query(q, 5)
        recs.append(recall_at_k(ids, exact_knn(data, q, 5, metric)))
        cands.append(nc)
    assert np.mean(recs) > 0.8
    assert np.mean(cands) < len(data)


def test_vector_index_empty_and_bad_metric():
    idx = VectorLSHIndex(4, num_tables=1, hashes_per_table=30).fit(np.eye(4))
    ids, d, n = idx.query(np.array([-1.0, -1.0, -1.0, -1.0]), 3)
    assert n == len(ids)
    with pytest.raises(ValueError):
        VectorLSHIndex(4, metric="manhattan")


def test_recall_helper():
    assert recall_at_k(np.array([1, 2]), np.array([2, 3])) == 0.5
    assert recall_at_k(np.array([]), np.array([])) == 1.0
