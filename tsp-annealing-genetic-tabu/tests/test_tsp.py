import numpy as np
import pytest

from tsp_metaheuristics import (brute_force, circle_instance, edge_recombination, genetic_algorithm,
                                held_karp, is_valid_tour, nearest_neighbour, or_opt,
                                order_crossover, random_instance, simulated_annealing, tabu_search,
                                tour_length, two_opt, two_opt_delta)


def test_circle_optimal_length():
    inst = circle_instance(12)
    expected = 12 * 2 * np.sin(np.pi / 12)
    assert tour_length(inst, list(range(12))) == pytest.approx(expected)


def test_held_karp_matches_brute_force():
    inst = random_instance(8, seed=3)
    hk, t = held_karp(inst)
    bf, _ = brute_force(inst)
    assert hk == pytest.approx(bf)
    assert is_valid_tour(inst, t) and tour_length(inst, t) == pytest.approx(hk)


def test_held_karp_circle():
    inst = circle_instance(10)
    assert held_karp(inst)[0] == pytest.approx(10 * 2 * np.sin(np.pi / 10))


def test_nearest_neighbour_valid():
    inst = random_instance(30, 1)
    assert is_valid_tour(inst, nearest_neighbour(inst))


def test_two_opt_delta_matches_recompute():
    inst = random_instance(15, 2)
    rng = np.random.default_rng(0)
    t = rng.permutation(15)
    for _ in range(30):
        i, j = sorted(rng.choice(np.arange(1, 15), 2, replace=False))
        new = t.copy()
        new[i:j + 1] = new[i:j + 1][::-1]
        assert tour_length(inst, new) - tour_length(inst, t) == pytest.approx(
            two_opt_delta(inst, t, int(i), int(j)))


def test_two_opt_improves_and_is_local_optimum():
    inst = random_instance(40, 4)
    t0 = np.random.default_rng(1).permutation(40)
    t1 = two_opt(inst, t0)
    assert is_valid_tour(inst, t1)
    assert tour_length(inst, t1) < tour_length(inst, t0)
    n = 40
    assert all(two_opt_delta(inst, t1, i, j) >= -1e-9 for i in range(1, n - 1) for j in range(i + 1, n))


def test_or_opt_never_worse_and_valid():
    inst = random_instance(30, 5)
    t0 = nearest_neighbour(inst)
    t1 = or_opt(inst, t0)
    assert is_valid_tour(inst, t1)
    assert tour_length(inst, t1) <= tour_length(inst, t0) + 1e-9


def test_crossovers_produce_permutations():
    rng = np.random.default_rng(0)
    for _ in range(50):
        p1, p2 = rng.permutation(12), rng.permutation(12)
        for cx in (order_crossover, edge_recombination):
            c = cx(p1, p2, rng)
            assert sorted(c.tolist()) == list(range(12))


def test_order_crossover_keeps_slice():
    rng = np.random.default_rng(7)
    p1, p2 = np.arange(10), np.arange(10)[::-1].copy()
    c = order_crossover(p1, p2, rng)
    assert sorted(c.tolist()) == list(range(10))


def test_annealing_valid_deterministic_and_length_consistent():
    inst = random_instance(25, 6)
    a = simulated_annealing(inst, 5000, seed=1)
    b = simulated_annealing(inst, 5000, seed=1)
    assert np.array_equal(a.tour, b.tour)
    assert is_valid_tour(inst, a.tour)
    assert a.length == pytest.approx(tour_length(inst, a.tour))
    assert a.history[-1] <= a.history[0] + 1e-12


def test_annealing_finds_optimum_on_small_instance():
    inst = random_instance(9, 8)
    opt, _ = held_karp(inst)
    r = simulated_annealing(inst, 20000, seed=0)
    assert r.length <= opt * 1.02


def test_genetic_improves_and_valid():
    inst = random_instance(20, 9)
    r = genetic_algorithm(inst, 30, 40, seed=2)
    assert is_valid_tour(inst, r.tour)
    assert r.history[-1] <= r.history[0]
    assert r.length == pytest.approx(tour_length(inst, r.tour))


@pytest.mark.parametrize("cx", ["ox", "erx"])
def test_genetic_circle_near_optimal(cx):
    inst = circle_instance(14)
    r = genetic_algorithm(inst, 40, 60, seed=0, crossover=cx)
    assert r.length <= 14 * 2 * np.sin(np.pi / 14) * 1.05


def test_tabu_valid_and_monotone_history():
    inst = random_instance(20, 10)
    r = tabu_search(inst, 60, seed=0)
    assert is_valid_tour(inst, r.tour)
    assert all(x >= y - 1e-12 for x, y in zip(r.history, r.history[1:]))
    assert r.length <= tour_length(inst, nearest_neighbour(inst)) + 1e-9


def test_tabu_small_optimum():
    inst = random_instance(9, 11)
    opt, _ = held_karp(inst)
    assert tabu_search(inst, 80, seed=0).length <= opt * 1.05
