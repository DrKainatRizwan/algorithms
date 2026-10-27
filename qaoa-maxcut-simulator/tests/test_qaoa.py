import numpy as np
import pytest

from qaoa_maxcut import *
from qaoa_maxcut.statevector import apply_sum_x


def test_cut_values_triangle():
    g = complete_graph(3)
    cv = cut_values(g)
    assert cv.max() == 2 and cv[0] == 0 and cv[0b101] == 2


def test_bruteforce_ring_even_odd():
    assert brute_force_maxcut(ring_graph(6))[0] == 6
    assert brute_force_maxcut(ring_graph(5))[0] == 4


def test_regular_graph_degrees():
    g = random_regular_graph(10, 3, seed=2)
    assert set(g.degrees()) == {3} and g.m == 15


def test_graph_validation():
    with pytest.raises(ValueError):
        Graph(3, [(1, 1, 1.0)])
    with pytest.raises(ValueError):
        random_regular_graph(5, 3)


def test_state_normalised_and_mixer_unitary():
    q = QAOA(erdos_renyi(6, 0.5, seed=1), p=2)
    psi = q.state(np.array([0.3, 0.7, 0.2, 0.5]))
    assert np.isclose(np.linalg.norm(psi), 1.0)


def test_zero_params_gives_half_total_weight():
    g = erdos_renyi(7, 0.6, seed=3, weighted=True)
    q = QAOA(g, 1)
    assert np.isclose(q.energy(np.zeros(2)), g.total_weight() / 2)


def test_mixer_matches_dense_matrix():
    n = 3
    rng = np.random.default_rng(0)
    psi = rng.normal(size=8) + 1j * rng.normal(size=8)
    x = np.array([[0, 1], [1, 0]])
    B = sum(np.kron(np.kron(np.eye(2 ** i), x), np.eye(2 ** (n - i - 1))) for i in range(n))
    w, v = np.linalg.eigh(B)
    U = (v * np.exp(-1j * 0.37 * w)) @ v.conj().T
    assert np.allclose(apply_mixer(psi, n, 0.37), U @ psi)
    assert np.allclose(apply_sum_x(psi, n), B @ psi)


@pytest.mark.parametrize("p", [1, 2, 3])
def test_adjoint_gradient_matches_finite_difference(p):
    q = QAOA(erdos_renyi(6, 0.6, seed=4, weighted=True), p)
    x = np.random.default_rng(p).uniform(0, 1, 2 * p)
    assert np.allclose(q.gradient(x), q.numeric_gradient(x), atol=1e-6)


@pytest.mark.parametrize("make", [lambda: ring_graph(7), lambda: random_regular_graph(8, 3, 1),
                                  lambda: complete_graph(5), lambda: erdos_renyi(8, 0.4, 5)])
def test_p1_closed_form_matches_simulation(make):
    g = make()
    q = QAOA(g, 1)
    for gam, bet in [(0.4, 0.3), (1.1, 0.7), (-0.6, 0.2)]:
        assert np.isclose(q.energy(np.array([gam, bet])), p1_expectation(g, gam, bet), atol=1e-9)


def test_ring_p1_reaches_three_quarters():
    g = ring_graph(10)
    q = QAOA(g, 1)
    gs = np.linspace(0, np.pi, 61)
    bs = np.linspace(0, np.pi / 2, 61)
    best = landscape_p1(q, gs, bs).max()
    assert np.isclose(best / g.m, 0.75, atol=2e-3)


def test_optimisation_improves_and_depth_helps():
    g = random_regular_graph(8, 3, seed=7)
    q1, q2 = QAOA(g, 1), QAOA(g, 2)
    r1 = multistart(q1.energy, q1.gradient, 2, starts=4, seed=1)
    x0 = interp_init(r1.x)
    r2 = adam_ascent(q2.energy, q2.gradient, x0, iters=150)
    assert r1.value > g.m / 2 + 0.5
    assert r2.value >= r1.value - 1e-6
    assert q2.approximation_ratio(r2.x) > 0.8


def test_nelder_mead_matches_adam_optimum():
    g = ring_graph(6)
    q = QAOA(g, 1)
    r = nelder_mead(q.energy, np.array([0.5, 0.4]))
    a = multistart(q.energy, q.gradient, 2, starts=4, seed=0)
    assert abs(r.value - a.value) < 1e-3


def test_interp_shape_and_endpoints():
    x = np.array([0.1, 0.2, 0.3, 0.4])
    y = interp_init(x)
    assert y.shape == (6,) and y[0] == 0.1 and y[2] == 0.2 and y[3] == 0.3 and y[5] == 0.4


def test_sampling_deterministic_and_consistent():
    g = ring_graph(6)
    q = QAOA(g, 1)
    x = np.array([0.5, 0.4])
    a = q.sample_energy(x, 4000, seed=3)
    assert a == q.sample_energy(x, 4000, seed=3)
    assert abs(a[0] - q.energy(x)) < 0.15
    assert a[1] <= q.optimum


def test_classical_baselines_bounds():
    g = random_regular_graph(10, 3, seed=5)
    opt = brute_force_maxcut(g)[0]
    gv, bits = greedy_maxcut(g, 0)
    lv, lbits = local_search_maxcut(g, 5, 0)
    assert g.cut_of(bits) == gv <= opt
    assert opt >= lv >= g.m / 2 and g.cut_of(lbits) == lv


def test_success_probability_in_range_and_improves():
    q = QAOA(ring_graph(6), 2)
    r = multistart(q.energy, q.gradient, 4, starts=6, seed=2, iters=200)
    assert q.success_probability(r.x) > q.success_probability(np.zeros(4))
