import numpy as np
import pytest

from evolution_strategies import (BENCHMARKS, CMAES, SNES, XNES, CrossEntropyMethod,
                                  OnePlusOneES, cmaes_minimize, get_problem, ipop_cmaes)
from evolution_strategies.nes import expm_sym
from evolution_strategies.result import rank_utilities


def test_benchmark_optima():
    for name, p in BENCHMARKS.items():
        x = p.optimum_x(5)
        assert p.func(x) == pytest.approx(p.optimum_f, abs=1e-9), name


def test_unknown_benchmark():
    with pytest.raises(KeyError):
        get_problem("nope")


def test_cmaes_sphere():
    r = cmaes_minimize(get_problem("sphere").func, np.ones(5) * 3, 1.0, ftarget=1e-10, seed=1)
    assert r.success and r.fun < 1e-10


def test_cmaes_rotated_ellipsoid_learns_covariance():
    p = get_problem("rotated_ellipsoid")
    r = cmaes_minimize(p.func, np.ones(6), 1.0, ftarget=1e-8, max_evals=60000, seed=2)
    assert r.success


def test_cmaes_rosenbrock():
    p = get_problem("rosenbrock")
    r = cmaes_minimize(p.func, np.zeros(4), 0.5, ftarget=1e-8, max_evals=40000, seed=3)
    assert r.success
    assert np.allclose(r.x, 1.0, atol=1e-3)


def test_cmaes_deterministic():
    f = get_problem("cigar").func
    a = cmaes_minimize(f, np.ones(4), 1.0, max_evals=2000, seed=7)
    b = cmaes_minimize(f, np.ones(4), 1.0, max_evals=2000, seed=7)
    assert a.fun == b.fun and np.array_equal(a.x, b.x)


def test_cmaes_weights_and_params():
    es = CMAES(np.zeros(10), 1.0)
    assert es.weights.sum() == pytest.approx(1.0)
    assert np.all(np.diff(es.weights) <= 0)
    assert es.c1 + es.cmu <= 1.0
    assert es.lam == 4 + int(3 * np.log(10))


def test_cmaes_covariance_stays_spd_and_adapts():
    p = get_problem("rotated_ellipsoid")
    es = CMAES(np.ones(5), 1.0, seed=0)
    for _ in range(150):
        X = es.ask()
        es.tell(X, [p.func(x) for x in X])
    assert np.all(np.linalg.eigvalsh(es.C) > 0)
    assert es.condition_number > 100


def test_cmaes_input_validation():
    with pytest.raises(ValueError):
        CMAES(np.zeros(3), -1.0)
    es = CMAES(np.zeros(3), 1.0)
    with pytest.raises(ValueError):
        es.tell(np.zeros((2, 3)), np.zeros(2))


def test_rank_utilities_zero_sum_and_monotone():
    u = rank_utilities(12)
    assert u.sum() == pytest.approx(0.0, abs=1e-12)
    assert np.all(np.diff(u) <= 1e-15)


def test_expm_sym_matches_series():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((4, 4)) * 0.3
    a = (a + a.T) / 2
    series, term = np.eye(4), np.eye(4)
    for k in range(1, 40):
        term = term @ a / k
        series += term
    assert np.allclose(expm_sym(a), series)


def test_xnes_sphere_and_det_shape():
    r = XNES(np.ones(4) * 2, 1.0, seed=0)
    res = r.minimize(get_problem("sphere").func, max_evals=20000, ftarget=1e-8)
    assert res.success
    assert np.linalg.det(r.B) == pytest.approx(1.0, rel=1e-6)


def test_snes_separable_ellipsoid():
    res = SNES(np.ones(8), 1.0, seed=0).minimize(get_problem("ellipsoid").func,
                                                 max_evals=100000, ftarget=1e-6)
    assert res.success


def test_nes_requires_ask_first():
    with pytest.raises(RuntimeError):
        XNES(np.zeros(2), 1.0).tell(np.zeros((6, 2)), np.zeros(6))


def test_one_plus_one_es():
    res = OnePlusOneES(np.ones(5) * 2, 1.0, seed=0).minimize(get_problem("sphere").func,
                                                             max_evals=20000, ftarget=1e-8)
    assert res.success
    assert res.history[-1] <= res.history[0]


def test_cross_entropy_method_sphere():
    res = CrossEntropyMethod(np.ones(4) * 2, 1.0, seed=0).minimize(
        get_problem("sphere").func, max_evals=20000, ftarget=1e-6)
    assert res.success


def test_ipop_beats_single_run_on_rastrigin():
    p = get_problem("rastrigin")
    single = cmaes_minimize(p.func, np.full(5, 3.0), 1.0, max_evals=30000, seed=0)
    ipop = ipop_cmaes(p.func, 5, (-5.12, 5.12), 2.0, max_evals=150000,
                      ftarget=1e-6, seed=0)
    assert ipop.fun <= single.fun
    assert ipop.fun < 1.0
