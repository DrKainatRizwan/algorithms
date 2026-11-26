import numpy as np
import pytest

from ldp_frequency import (base_cut, best_oracle, clip_normalize, grr_threshold,
                           l1_error, make_oracle, mse, norm_sub, project_simplex,
                           required_users, sample_users, topk_recall, true_frequencies,
                           variance_table, zipf_distribution)


def test_best_oracle_switches_at_grr_threshold():
    eps = 1.0
    t = grr_threshold(eps)  # ~10.15
    assert best_oracle(int(t) - 2, eps).name == "GRR"
    assert best_oracle(int(t) + 20, eps).name in {"OUE", "OLH"}
    assert best_oracle(5000, eps).name in {"OUE", "OLH"}


def test_oue_and_olh_have_near_equal_variance():
    t = variance_table(1000, 2.0, 10_000)
    assert t["OLH"] == pytest.approx(t["OUE"], rel=0.1)
    assert t["SUE"] > t["OUE"]
    assert t["GRR"] > t["OUE"]


def test_required_users_hits_target():
    o = make_oracle("OUE", 100, 1.0)
    n = required_users(o, 0.01)
    assert np.sqrt(o.variance(n)) <= 0.01
    assert np.sqrt(o.variance(n - 1)) > 0.01 - 1e-12


def test_project_simplex_properties():
    rng = np.random.default_rng(3)
    for _ in range(20):
        v = rng.normal(0.1, 0.3, size=15)
        p = project_simplex(v)
        assert p.sum() == pytest.approx(1.0)
        assert (p >= 0).all()
        # no other simplex point is closer
        q = rng.dirichlet(np.ones(15))
        assert np.linalg.norm(p - v) <= np.linalg.norm(q - v) + 1e-12


def test_norm_sub_equals_projection():
    rng = np.random.default_rng(4)
    for _ in range(20):
        v = rng.normal(0.05, 0.2, size=25)
        assert np.allclose(norm_sub(v), project_simplex(v), atol=1e-9)


def test_clip_normalize_and_base_cut():
    v = np.array([0.5, -0.2, 0.3, 0.01])
    c = clip_normalize(v)
    assert c.sum() == pytest.approx(1.0) and c[1] == 0
    assert np.allclose(clip_normalize(np.array([-1.0, -2.0])), 0.5)
    b = base_cut(v, 1000, variance=0.01, z=2.0)  # threshold 0.2
    assert b[3] == 0 and b.sum() == pytest.approx(1.0)


def test_postprocessing_reduces_error_on_sparse_data():
    d, n = 200, 20_000
    rng = np.random.default_rng(8)
    values = sample_users(zipf_distribution(d, 1.5), n, rng)
    truth = true_frequencies(values, d)
    est = make_oracle("OLH", d, 1.0).run(values, rng)
    assert mse(truth, project_simplex(est)) < mse(truth, est)


def test_metrics():
    t = np.array([0.5, 0.3, 0.2])
    assert mse(t, t) == 0 and l1_error(t, np.array([0.4, 0.4, 0.2])) == pytest.approx(0.2)
    assert topk_recall(t, np.array([0.1, 0.5, 0.4]), 2) == 0.5
