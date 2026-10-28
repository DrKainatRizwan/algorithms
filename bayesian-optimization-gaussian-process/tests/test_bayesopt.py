import numpy as np
import pytest

from gp_bayesopt import (BayesianOptimizer, GaussianProcess, Matern32, Matern52, RBF, TEST_FUNCTIONS,
                         expected_improvement, latin_hypercube, nelder_mead, norm_cdf,
                         probability_of_improvement, random_search, upper_confidence_bound)


@pytest.mark.parametrize("K", [RBF, Matern32, Matern52])
def test_kernel_psd_and_symmetric(K):
    X = np.random.default_rng(0).random((20, 3))
    M = K(np.array([0.5, 1.0, 2.0]), 1.7)(X, X)
    assert np.allclose(M, M.T)
    assert np.linalg.eigvalsh(M).min() > -1e-8
    assert np.allclose(np.diag(M), 1.7)


def test_kernel_decays_with_distance():
    k = Matern52(1.0, 1.0)
    a, b, c = np.array([[0.0]]), np.array([[0.5]]), np.array([[3.0]])
    assert k(a, a)[0, 0] > k(a, b)[0, 0] > k(a, c)[0, 0] > 0


def test_gp_interpolates_noise_free():
    X = np.linspace(0, 1, 8)[:, None]
    y = np.sin(6 * X[:, 0])
    gp = GaussianProcess(RBF(0.3, 1.0), noise=1e-8).fit(X, y)
    mu, std = gp.predict(X)
    assert np.allclose(mu, y, atol=1e-3)
    assert std.max() < 1e-2


def test_gp_uncertainty_grows_away_from_data():
    X = np.array([[0.1], [0.2], [0.3]])
    gp = GaussianProcess(Matern52(0.2, 1.0)).fit(X, np.array([0.0, 1.0, 0.5]))
    _, s = gp.predict(np.array([[0.2], [0.9]]))
    assert s[1] > 10 * s[0]


def test_hyperparameter_optimisation_improves_lml():
    rng = np.random.default_rng(1)
    X = rng.random((25, 1))
    y = np.sin(8 * X[:, 0]) + 0.05 * rng.standard_normal(25)
    gp = GaussianProcess(RBF(5.0, 1.0), noise=1e-2).fit(X, y)
    before = gp.log_marginal_likelihood()
    gp.fit(X, y, optimize=True, seed=0)
    assert gp.log_marginal_likelihood() > before + 1.0
    assert gp.kernel.lengthscale[0] < 1.0


def test_posterior_samples_match_moments():
    X = np.array([[0.0], [1.0]])
    gp = GaussianProcess(RBF(0.5, 1.0), noise=1e-6).fit(X, np.array([0.0, 1.0]))
    Xs = np.array([[0.5], [2.0]])
    S = gp.sample_posterior(Xs, 4000, np.random.default_rng(3))
    mu, std = gp.predict(Xs)
    assert np.allclose(S.mean(0), mu, atol=0.1)
    assert np.allclose(S.std(0), std, rtol=0.1)


def test_norm_cdf_values():
    assert norm_cdf(np.array([0.0]))[0] == pytest.approx(0.5)
    assert norm_cdf(np.array([1.96]))[0] == pytest.approx(0.975, abs=1e-4)


def test_expected_improvement_properties():
    std = np.array([1.0, 1.0, 1.0])
    ei = expected_improvement(np.array([-1.0, 0.0, 1.0]), std, best=0.0)
    assert np.all(ei > 0) and ei[0] > ei[1] > ei[2]
    assert expected_improvement(np.array([1.0]), np.array([0.0]), 0.0)[0] == 0.0
    # closed form at mu=best: sigma * pdf(0)
    assert expected_improvement(np.array([0.0]), np.array([2.0]), 0.0)[0] == pytest.approx(2 / np.sqrt(2 * np.pi))


def test_pi_and_ucb_monotone():
    p = probability_of_improvement(np.array([-1.0, 1.0]), np.array([1.0, 1.0]), 0.0)
    assert p[0] > 0.5 > p[1]
    u = upper_confidence_bound(np.array([0.0, 0.0]), np.array([1.0, 2.0]), beta=4.0)
    assert u[1] > u[0]


def test_nelder_mead_respects_bounds_and_finds_min():
    x, v = nelder_mead(lambda z: float(((z - 0.3) ** 2).sum()), np.array([0.9, 0.9]),
                       bounds=(np.zeros(2), np.ones(2)))
    assert np.allclose(x, 0.3, atol=1e-3) and v < 1e-5
    x, _ = nelder_mead(lambda z: float(((z - 5) ** 2).sum()), np.array([0.5]), bounds=(np.zeros(1), np.ones(1)))
    assert x[0] == pytest.approx(1.0, abs=1e-6)


def test_latin_hypercube_stratified():
    u = latin_hypercube(10, 3, np.random.default_rng(0))
    for j in range(3):
        assert sorted((u[:, j] * 10).astype(int)) == list(range(10))


def test_test_function_optima():
    f, _, opt = TEST_FUNCTIONS["branin"]
    assert f(np.array([np.pi, 2.275])) == pytest.approx(opt, abs=1e-4)
    f, _, opt = TEST_FUNCTIONS["hartmann3"]
    assert f(np.array([0.114614, 0.555649, 0.852547])) == pytest.approx(opt, abs=1e-3)


def test_bo_solves_forrester():
    f, b, opt = TEST_FUNCTIONS["forrester"]
    bo = BayesianOptimizer(b, n_init=4, seed=0)
    _, y = bo.minimize(f, n_iter=12)
    assert y < opt + 0.05


def test_bo_beats_random_search_on_branin():
    f, b, opt = TEST_FUNCTIONS["branin"]
    bo = BayesianOptimizer(b, n_init=5, seed=1)
    _, y = bo.minimize(f, n_iter=25)
    rs = random_search(f, b, 30, seed=1)[-1]
    assert y < 1.0 and y < rs


@pytest.mark.parametrize("acq", ["ei", "pi", "ucb"])
def test_all_acquisitions_make_progress(acq):
    f, b, _ = TEST_FUNCTIONS["sphere3"]
    bo = BayesianOptimizer(b, acquisition=acq, n_init=5, seed=2)
    bo.minimize(f, n_iter=15)
    h = bo.history()
    assert h[-1] < h[4] and np.all(np.diff(h) <= 0)


def test_batch_ask_returns_distinct_points_in_bounds():
    f, b, _ = TEST_FUNCTIONS["branin"]
    bo = BayesianOptimizer(b, n_init=4, seed=0)
    X = bo.ask(4)
    bo.tell(X, [f(x) for x in X])
    B = bo.ask(3)
    assert B.shape == (3, 2)
    assert np.all(B >= b[0]) and np.all(B <= b[1])
    assert len({tuple(np.round(r, 6)) for r in B}) == 3


def test_determinism_and_bad_acquisition():
    f, b, _ = TEST_FUNCTIONS["forrester"]
    r = [BayesianOptimizer(b, n_init=3, seed=7).minimize(f, n_iter=4)[1] for _ in range(2)]
    assert r[0] == r[1]
    with pytest.raises(ValueError):
        BayesianOptimizer(b, acquisition="nope")
