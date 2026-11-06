import numpy as np
import pytest

from backdoor_detection import (MLP, activation_clustering, apply_trigger, attack_success_rate,
                                clean_accuracy, evaluate_detection, kmeans, make_dataset, pca,
                                poison_dataset, sanitize_and_retrain, silhouette_score,
                                spectral_signature, train)

C, T = 4, 0


@pytest.fixture(scope="module")
def setup():
    Xtr, ytr = make_dataset(300, C, seed=1)
    Xte, yte = make_dataset(100, C, seed=2)
    pd = poison_dataset(Xtr, ytr, T, 0.10, seed=3)
    m = train(pd.X, pd.y, C, seed=0)
    return pd, m, Xte, yte


def test_dataset_deterministic_and_balanced():
    X1, y1 = make_dataset(20, 3, seed=5)
    X2, y2 = make_dataset(20, 3, seed=5)
    assert np.array_equal(X1, X2) and np.array_equal(y1, y2)
    assert np.bincount(y1).tolist() == [20, 20, 20]


def test_trigger_only_changes_corner():
    X, _ = make_dataset(5, 2, seed=0)
    Xt = apply_trigger(X, size=3, value=4.0)
    a, b = X.reshape(-1, 8, 8), Xt.reshape(-1, 8, 8)
    assert np.all(b[:, -3:, -3:] == 4.0)
    assert np.array_equal(a[:, :5], b[:, :5]) and np.array_equal(a[:, :, :5], b[:, :, :5])


def test_poison_rate_and_labels():
    X, y = make_dataset(100, C, seed=0)
    pd = poison_dataset(X, y, T, 0.1, seed=0)
    assert abs(pd.is_poison.mean() - 0.1) < 0.005
    assert np.all(pd.y[pd.is_poison] == T)
    with pytest.raises(ValueError):
        poison_dataset(X, y, T, 1.0)


def test_gradient_check():
    rng = np.random.default_rng(0)
    m = MLP([5, 6, 3], seed=1)
    X, y = rng.normal(size=(8, 5)), rng.integers(0, 3, 8)
    W0 = m.W[0].copy()
    base = m.loss(X, y)
    eps = 1e-6
    m.W[0][1, 2] += eps
    num = (m.loss(X, y) - base) / eps
    m.W[0] = W0.copy()
    m.step(X, y, lr=1.0)
    analytic = W0[1, 2] - m.W[0][1, 2]
    assert analytic == pytest.approx(num, rel=1e-3, abs=1e-6)


def test_backdoor_is_implanted(setup):
    pd, m, Xte, yte = setup
    assert clean_accuracy(m, Xte, yte) > 0.9
    assert attack_success_rate(m, Xte, yte, T) > 0.9


def test_pca_orthonormal_and_ratio():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 6)) * np.array([5, 3, 1, 1, 0.5, 0.1])
    Z, comps, ratio = pca(X, 3)
    assert Z.shape == (100, 3)
    assert np.allclose(comps @ comps.T, np.eye(3), atol=1e-10)
    assert np.all(np.diff(ratio) <= 0) and ratio.sum() <= 1.0


def test_kmeans_recovers_blobs():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 0.3, (50, 2)), rng.normal(5, 0.3, (30, 2))])
    labels, centers, _ = kmeans(X, 2, seed=0)
    assert sorted(np.bincount(labels)) == [30, 50]
    assert len(set(labels[:50])) == 1


def test_kmeans_invalid_k():
    with pytest.raises(ValueError):
        kmeans(np.zeros((3, 2)), 5)


def test_silhouette_range_and_separation():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 0.2, (30, 2)), rng.normal(6, 0.2, (30, 2))])
    good = np.repeat([0, 1], 30)
    bad = rng.integers(0, 2, 60)
    assert silhouette_score(X, good) > 0.9
    assert silhouette_score(X, bad) < 0.2
    assert silhouette_score(X, np.zeros(60, int)) == 0.0


def test_activation_clustering_finds_poison(setup):
    pd, m, _, _ = setup
    res = activation_clustering(m, pd.X, pd.y, C)
    ev = evaluate_detection(res.flagged, pd.is_poison)
    assert ev["precision"] > 0.9 and ev["recall"] > 0.9
    assert [r.label for r in res.reports if r.suspicious] == [T]


def test_clean_data_not_flagged():
    X, y = make_dataset(300, C, seed=1)
    m = train(X, y, C, seed=0)
    res = activation_clustering(m, X, y, C)
    assert res.mask_size == 0


def test_spectral_signature_flag_count(setup):
    pd, m, _, _ = setup
    res = spectral_signature(m, pd.X, pd.y, C, eps=0.1)
    expected = sum(int(np.ceil(1.5 * 0.1 * (pd.y == c).sum())) for c in range(C))
    assert res.mask_size == expected


def test_evaluate_detection_exact():
    truth = np.array([1, 1, 0, 0, 0, 1], bool)
    r = evaluate_detection(np.array([0, 2]), truth)
    assert (r["tp"], r["fp"], r["fn"]) == (1, 1, 2)
    assert r["precision"] == 0.5 and r["recall"] == pytest.approx(1 / 3)


def test_sanitise_removes_backdoor(setup):
    pd, m, Xte, yte = setup
    res = activation_clustering(m, pd.X, pd.y, C)
    m2 = sanitize_and_retrain(pd.X, pd.y, res.flagged, C, seed=0)
    assert attack_success_rate(m2, Xte, yte, T) < 0.2
    assert clean_accuracy(m2, Xte, yte) > 0.9
