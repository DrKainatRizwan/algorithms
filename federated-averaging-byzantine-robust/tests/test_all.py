import numpy as np
import pytest

from byzantine_fedavg import (
    ATTACKS, AGGREGATORS, FederatedSimulator, SoftmaxModel, coordinate_median,
    fedavg_mean, geometric_median, krum, little_is_enough, make_blobs_dataset,
    multi_krum, partition_iid, partition_label_skew, sign_flip_attack,
    trimmed_mean,
)
from byzantine_fedavg.aggregators import krum_scores


def _updates(seed=0, n=10, d=5):
    return np.random.default_rng(seed).normal(size=(n, d))


def test_mean_matches_numpy_and_weighted():
    u = _updates()
    assert np.allclose(fedavg_mean(u), u.mean(0))
    w = np.arange(1, 11, dtype=float)
    assert np.allclose(fedavg_mean(u, weights=w), (w[:, None] * u).sum(0) / w.sum())


def test_median_resists_outliers():
    u = np.vstack([np.zeros((7, 3)), np.full((3, 3), 1e6)])
    assert np.allclose(coordinate_median(u), 0)


def test_trimmed_mean_known_value():
    u = np.array([[1.0], [2.0], [3.0], [4.0], [100.0]])
    assert trimmed_mean(u, f=1)[0] == pytest.approx(3.0)
    with pytest.raises(ValueError):
        trimmed_mean(u, f=3)


def test_krum_picks_honest_cluster_member():
    rng = np.random.default_rng(1)
    honest = rng.normal(0, 0.1, size=(8, 4))
    bad = np.full((2, 4), 50.0)
    out = krum(np.vstack([bad, honest]), f=2)
    assert any(np.allclose(out, h) for h in honest)


def test_krum_scores_validation_and_order():
    u = _updates(n=6)
    with pytest.raises(ValueError):
        krum_scores(u, f=4)
    assert krum_scores(u, f=1).shape == (6,)


def test_multi_krum_excludes_outliers():
    rng = np.random.default_rng(2)
    honest = rng.normal(0, 0.1, size=(8, 4))
    bad = np.full((2, 4), 50.0)
    out = multi_krum(np.vstack([bad, honest]), f=2)
    assert np.abs(out).max() < 1.0


def test_geometric_median_robust_and_collinear():
    pts = np.array([[0.0, 0], [1, 0], [2, 0], [1000, 0]])
    gm = geometric_median(pts)
    assert 0.9 < gm[0] < 2.1 and abs(gm[1]) < 1e-9


def test_aggregators_are_permutation_invariant():
    u = _updates(seed=3)
    perm = np.random.default_rng(4).permutation(len(u))
    for name, agg in AGGREGATORS.items():
        a, b = agg(u, f=2), agg(u[perm], f=2)
        assert np.allclose(a, b, atol=1e-6), name


def test_attack_shapes_and_effects():
    honest = _updates(n=8)
    rng = np.random.default_rng(0)
    for name, atk in ATTACKS.items():
        assert atk(honest, 3, rng).shape == (3, honest.shape[1]), name
    assert np.allclose(sign_flip_attack(honest, 1, rng)[0], -honest.mean(0))
    lie = little_is_enough(honest, 1, rng, z=1.0)[0]
    assert np.allclose(lie, honest.mean(0) - honest.std(0))


def test_gradient_matches_finite_differences():
    X, y = make_blobs_dataset(50, 4, 3, seed=1)
    m = SoftmaxModel(4, 3)
    w = np.random.default_rng(0).normal(size=m.dim)
    _, g = m.loss_and_grad(w, X, y, l2=0.01)
    for i in [0, 5, m.dim - 1]:
        e = np.zeros(m.dim); e[i] = 1e-6
        num = (m.loss_and_grad(w + e, X, y, 0.01)[0]
               - m.loss_and_grad(w - e, X, y, 0.01)[0]) / 2e-6
        assert num == pytest.approx(g[i], abs=1e-5)


def test_partitions_cover_all_samples():
    X, y = make_blobs_dataset(500, seed=2)
    for shards in (partition_iid(500, 10), partition_label_skew(y, 10, 0.3)):
        allidx = np.concatenate(shards)
        assert sorted(allidx.tolist()) == list(range(500))
        assert all(len(s) > 0 for s in shards)


def _setup(agg, n_byz, attack, rounds=8, seed=0):
    X, y = make_blobs_dataset(1500, 8, 3, seed=seed)
    Xtr, ytr, Xte, yte = X[:1200], y[:1200], X[1200:], y[1200:]
    sim = FederatedSimulator(
        SoftmaxModel(8, 3), Xtr, ytr, partition_iid(1200, 10, seed), Xte, yte,
        agg, n_byzantine=n_byz, attack=attack, seed=seed)
    return sim.run(rounds)


def test_clean_fedavg_learns():
    assert _setup(fedavg_mean, 0, None).final_accuracy > 0.9


def test_mean_breaks_but_robust_survives_signflip():
    atk = ATTACKS["scaled_negative"]
    bad = _setup(fedavg_mean, 3, atk).final_accuracy
    good = _setup(trimmed_mean, 3, atk).final_accuracy
    assert bad < 0.6 and good > 0.9


def test_determinism():
    a = _setup(krum, 2, ATTACKS["gaussian"])
    b = _setup(krum, 2, ATTACKS["gaussian"])
    assert a.accuracy == b.accuracy


def test_simulator_validation():
    X, y = make_blobs_dataset(100, 4, 2)
    with pytest.raises(ValueError):
        FederatedSimulator(SoftmaxModel(4, 2), X, y, partition_iid(100, 5),
                           X, y, krum, n_byzantine=1, attack=None)
