import numpy as np
import pytest

from swarm_evolution_opt import (PSO, DifferentialEvolution, compare_optimizers, de_minimize,
                                 get_benchmark, list_benchmarks, pso_minimize, summarize,
                                 wilcoxon_rank_sum)
from swarm_evolution_opt.de import _distinct_indices
from swarm_evolution_opt.pso import _neighbour_best


@pytest.mark.parametrize("name", list_benchmarks())
def test_benchmark_optimum_value(name):
    b = get_benchmark(name)
    for d in (2, 5):
        val = b.func(b.x_opt(d)[None, :])[0]
        assert abs(val - b.f_opt) < 1e-6


def test_unknown_benchmark():
    with pytest.raises(KeyError):
        get_benchmark("nope")


def test_pso_sphere():
    r = pso_minimize(get_benchmark("sphere").func, -5, 5, dim=5, max_iter=300, seed=1)
    assert r.fun < 1e-8 and r.x.shape == (5,)


def test_pso_deterministic():
    f = get_benchmark("rastrigin").func
    a = pso_minimize(f, -5.12, 5.12, dim=4, max_iter=50, seed=3)
    b = pso_minimize(f, -5.12, 5.12, dim=4, max_iter=50, seed=3)
    assert a.fun == b.fun and np.array_equal(a.x, b.x)


@pytest.mark.parametrize("boundary", ["clip", "reflect", "random"])
def test_pso_respects_bounds(boundary):
    seen = []

    def f(X):
        seen.append(X.copy())
        return np.sum((X - 4.0) ** 2, axis=1)  # optimum outside the box

    r = pso_minimize(f, -1, 1, dim=3, boundary=boundary, max_iter=60, seed=0)
    allx = np.vstack(seen)
    assert allx.min() >= -1 - 1e-12 and allx.max() <= 1 + 1e-12
    assert np.all(r.x <= 1)


@pytest.mark.parametrize("topology", ["global", "ring", "von_neumann"])
def test_pso_topologies_converge(topology):
    r = pso_minimize(get_benchmark("sphere").func, -5, 5, dim=3, topology=topology,
                     n_particles=25, max_iter=300, seed=2)
    assert r.fun < 1e-4


def test_pso_constriction_and_decay():
    f = get_benchmark("sphere").func
    r1 = pso_minimize(f, -5, 5, dim=4, variant="constriction", c1=2.05, c2=2.05, max_iter=200, seed=0)
    r2 = pso_minimize(f, -5, 5, dim=4, variant="decay", w=0.9, w_end=0.4, max_iter=200, seed=0)
    assert r1.fun < 1e-6 and r2.fun < 1e-6
    with pytest.raises(ValueError):
        PSO(f, -5, 5, dim=2, variant="constriction")  # default c1+c2 < 4


def test_ring_neighbour_best_wraps():
    x = np.arange(6, dtype=float)[:, None]
    f = np.array([5, 4, 3, 0, 3, 1.0])
    nb = _neighbour_best(x, f, "ring", 1)
    assert nb[:, 0].tolist() == [5, 2, 3, 3, 3, 5]


def test_budget_and_history_monotone():
    r = pso_minimize(get_benchmark("ackley").func, -32, 32, dim=5, max_iter=1000, max_nfev=600, seed=0)
    assert r.nfev <= 600 + 30 and r.message == "evaluation budget exhausted"
    assert all(b <= a for a, b in zip(r.history, r.history[1:]))


def test_de_sphere_all_strategies():
    f = get_benchmark("sphere").func
    for s in ("rand1", "best1", "current_to_best1", "rand2", "best2"):
        r = de_minimize(f, -5, 5, dim=5, strategy=s, pop_size=30, max_iter=400, F=0.5, seed=4)
        assert r.fun < 1e-6, s


def test_de_exponential_crossover_and_tol():
    r = de_minimize(get_benchmark("sphere").func, -5, 5, dim=5, crossover="exp", tol=1e-6,
                    max_iter=2000, seed=0)
    assert r.fun <= 1e-6 and r.message == "target tolerance reached"


def test_de_rosenbrock_jde():
    r = de_minimize(get_benchmark("rosenbrock").func, -2.048, 2.048, dim=4, adaptive=True,
                    pop_size=40, max_iter=1500, seed=5)
    assert r.fun < 1e-3


def test_de_bounds_and_validation():
    seen = []

    def f(X):
        seen.append(X.copy())
        return np.sum((X + 3) ** 2, axis=1)

    de_minimize(f, -1, 1, dim=3, max_iter=50, seed=0)
    assert np.vstack(seen).min() >= -1 and np.vstack(seen).max() <= 1
    with pytest.raises(ValueError):
        DifferentialEvolution(f, -1, 1, dim=2, strategy="bogus")
    with pytest.raises(ValueError):
        DifferentialEvolution(f, -1, 1, dim=2, pop_size=3)
    with pytest.raises(ValueError):
        DifferentialEvolution(f, 1, -1, dim=2)


def test_distinct_indices():
    idx = _distinct_indices(np.random.default_rng(0), 10, 4)
    for i, row in enumerate(idx):
        assert i not in row and len(set(row)) == 4


def test_de_beats_pso_on_rastrigin_budget():
    cfg = {"pso": ("pso", {}), "de": ("de", {"adaptive": True})}
    errs = compare_optimizers(cfg, "rastrigin", 6, list(range(5)), max_nfev=15000)
    assert errs["de"].mean() < errs["pso"].mean()


def test_summarize_keys():
    s = summarize({"a": np.array([1.0, 3.0])})
    assert s["a"]["mean"] == 2.0 and s["a"]["best"] == 1.0


def test_rank_sum_test():
    a = np.arange(10, dtype=float)
    z, p = wilcoxon_rank_sum(a, a + 100)
    assert z < 0 and p < 0.001
    z2, p2 = wilcoxon_rank_sum(a, a)
    assert abs(z2) < 1e-9 and p2 > 0.99
