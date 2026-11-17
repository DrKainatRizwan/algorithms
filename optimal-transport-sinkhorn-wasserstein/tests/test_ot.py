import itertools

import numpy as np
import pytest

from optimal_transport import *
from optimal_transport.exact import plan_is_feasible


def rand_problem(n, m, seed):
    rng = np.random.default_rng(seed)
    a, b = rng.random(n) + 0.1, rng.random(m) + 0.1
    x, y = rng.standard_normal((n, 2)), rng.standard_normal((m, 2)) + 1
    return a / a.sum(), b / b.sum(), pairwise_cost(x, y)


def test_emd_assignment_matches_bruteforce():
    rng = np.random.default_rng(1)
    n = 6
    C = rng.random((n, n))
    best = min(sum(C[i, p[i]] for i in range(n)) for p in itertools.permutations(range(n))) / n
    val = emd(np.ones(n), np.ones(n), C, return_plan=False)
    assert val == pytest.approx(best, abs=1e-9)


@pytest.mark.parametrize("n,m,seed", [(5, 7, 0), (10, 10, 1), (3, 12, 2), (15, 8, 3)])
def test_emd_plan_feasible(n, m, seed):
    a, b, C = rand_problem(n, m, seed)
    val, P = emd(a, b, C)
    assert plan_is_feasible(P, a, b)
    assert val == pytest.approx((P * C).sum())


def test_emd_beats_independent_coupling():
    a, b, C = rand_problem(8, 9, 4)
    val, _ = emd(a, b, C)
    assert val <= (np.outer(a, b) * C).sum() + 1e-12


def test_emd_identical_measures_zero():
    a, _, _ = rand_problem(6, 6, 5)
    x = np.random.default_rng(5).standard_normal((6, 2))
    assert emd(a, a, pairwise_cost(x, x), return_plan=False) == pytest.approx(0, abs=1e-10)


def test_emd_matches_1d_closed_form():
    rng = np.random.default_rng(6)
    x, y = rng.standard_normal(12), rng.standard_normal(9) * 2
    a, b = rng.random(12) + .2, rng.random(9) + .2
    exact = wasserstein_exact(x, y, a, b, p=2.0)
    assert exact == pytest.approx(wasserstein_1d(x, y, a, b, p=2.0), abs=1e-9)


def test_emd_dual_lower_bound_via_sinkhorn():
    a, b, C = rand_problem(10, 10, 7)
    ex, _ = emd(a, b, C)
    r = sinkhorn(a, b, C, eps=0.05)
    assert r.cost >= ex - 1e-9


def test_sinkhorn_marginals_and_convergence():
    a, b, C = rand_problem(12, 15, 8)
    r = sinkhorn(a, b, C, eps=0.1)
    assert r.marginal_error < 1e-7
    assert r.plan.min() >= 0


def test_sinkhorn_approaches_exact_as_eps_shrinks():
    a, b, C = rand_problem(10, 10, 9)
    ex, _ = emd(a, b, C)
    errs = [sinkhorn(a, b, C, eps=e, max_iter=20000).cost - ex for e in (0.5, 0.1, 0.02)]
    assert errs[0] > errs[1] > errs[2] >= -1e-9
    assert errs[2] < 0.02


def test_log_domain_matches_classic():
    a, b, C = rand_problem(8, 8, 10)
    r1, r2 = sinkhorn(a, b, C, 0.2), sinkhorn_classic(a, b, C, 0.2)
    assert np.allclose(r1.plan, r2.plan, atol=1e-7)


def test_log_domain_stable_small_eps():
    a, b, C = rand_problem(8, 8, 11)
    r = sinkhorn(a, b, C, 1e-3, max_iter=20000)
    assert np.all(np.isfinite(r.plan))


def test_eps_scaling_matches_direct():
    a, b, C = rand_problem(10, 10, 12)
    r1 = sinkhorn_eps_scaling(a, b, C, eps=0.02)
    r2 = sinkhorn(a, b, C, 0.02, max_iter=50000)
    assert r1.cost == pytest.approx(r2.cost, abs=1e-5)


def test_sinkhorn_divergence_properties():
    rng = np.random.default_rng(13)
    x, y = rng.standard_normal((20, 2)), rng.standard_normal((20, 2)) + 1.5
    a = np.ones(20) / 20
    d = lambda u, v: sinkhorn_divergence(a, a, pairwise_cost(u, v), pairwise_cost(u, u), pairwise_cost(v, v), 0.3)
    assert d(x, x) == pytest.approx(0, abs=1e-8)
    assert d(x, y) > 0.1
    assert d(x, y) == pytest.approx(d(y, x), abs=1e-8)


def test_wasserstein_1d_shift():
    x = np.random.default_rng(14).standard_normal(50)
    assert wasserstein_1d(x, x + 2.5, p=2) == pytest.approx(2.5)
    assert wasserstein_1d(x, x + 2.5, p=1) == pytest.approx(2.5)


def test_wasserstein_1d_triangle_inequality():
    rng = np.random.default_rng(15)
    x, y, z = rng.standard_normal(30), rng.standard_normal(30) + 1, rng.standard_normal(30) * 2
    assert wasserstein_1d(x, z, p=1) <= wasserstein_1d(x, y, p=1) + wasserstein_1d(y, z, p=1) + 1e-12


def test_sliced_wasserstein_1d_equals_w():
    rng = np.random.default_rng(16)
    x, y = rng.standard_normal(40), rng.standard_normal(40) + 1
    assert sliced_wasserstein(x[:, None], y[:, None], n_proj=5) == pytest.approx(wasserstein_1d(x, y, p=2))


def test_sliced_lower_bounds_wasserstein():
    rng = np.random.default_rng(17)
    X, Y = rng.standard_normal((25, 3)), rng.standard_normal((25, 3)) + 1
    sw = sliced_wasserstein(X, Y, n_proj=200, seed=1)
    assert 0 < sw <= wasserstein_exact(X, Y, p=2.0) + 1e-9


def test_1d_barycenter_of_shifted_gaussians():
    rng = np.random.default_rng(18)
    z = rng.standard_normal(2000)
    bar = wasserstein_1d_barycenter([z - 3, z + 5], [0.5, 0.5], 500)
    assert bar.mean() == pytest.approx(1.0, abs=0.05)
    assert bar.std() == pytest.approx(z.std(), rel=0.05)


def test_entropic_barycenter_gaussians():
    n = 100
    t = np.linspace(0, 1, n)
    g = lambda m: (lambda h: h / h.sum())(np.exp(-(t - m) ** 2 / (2 * 0.04**2)))
    C = grid_cost(n, 2)
    bar = entropic_barycenter(np.vstack([g(0.2), g(0.8)]), C, eps=0.002, max_iter=3000)
    assert bar.sum() == pytest.approx(1.0)
    assert (bar * t).sum() == pytest.approx(0.5, abs=0.02)
    assert np.sqrt((bar * (t - 0.5) ** 2).sum()) < 0.08


def test_barycentric_map_recovers_translation():
    rng = np.random.default_rng(19)
    x = rng.standard_normal((15, 2))
    y = x + np.array([2.0, -1.0])
    a = np.ones(15)
    _, P = emd(a, a, pairwise_cost(x, y))
    assert np.allclose(barycentric_map(P, y), y, atol=1e-8)


def test_validation_errors():
    with pytest.raises(ValueError):
        emd(np.array([-1.0, 2]), np.ones(2), np.zeros((2, 2)))
    with pytest.raises(ValueError):
        sinkhorn(np.ones(2), np.ones(3), np.zeros((2, 2)))
    with pytest.raises(ValueError):
        sinkhorn(np.ones(2), np.ones(2), np.zeros((2, 2)), eps=0)
