import math

import numpy as np
import pytest

from gossip_sim import (Replica, complete_graph, erdos_renyi, mean_field_curve,
                        mean_field_rounds, push_rounds_estimate, push_sum,
                        ring_lattice, rumor_mongering, rumor_residue, run_anti_entropy,
                        run_trials, spread, watts_strogatz, barabasi_albert)


def rng(s=0):
    return np.random.default_rng(s)


def test_graph_shapes():
    g = complete_graph(10)
    assert g.edge_count() == 45 and g.is_connected()
    r = ring_lattice(20, 2)
    assert all(r.degree(v) == 4 for v in range(20))
    assert r.diameter_from(0) == 5


def test_random_graphs_connected_and_symmetric():
    g = erdos_renyi(100, 0.1, rng(1))
    for u in range(g.n):
        for v in g.adj[u]:
            assert u in g.adj[v]
    ba = barabasi_albert(200, 3, rng(2))
    assert ba.is_connected()
    ws = watts_strogatz(100, 3, 0.2, rng(3))
    assert ws.edge_count() <= 300 and ws.edge_count() > 250


def test_push_reaches_everyone_complete():
    res = spread(complete_graph(200), "push", rng(4))
    assert res.rounds_to_full is not None and res.coverage == 1.0
    assert res.informed_history == sorted(res.informed_history)


@pytest.mark.parametrize("mode", ["push", "pull", "push-pull"])
def test_modes_complete(mode):
    res = spread(complete_graph(128), mode, rng(5))
    assert res.rounds_to_full is not None


def test_push_pull_faster_than_push():
    g = complete_graph(500)
    a = run_trials(g, "push", 20, seed=1)
    b = run_trials(g, "push-pull", 20, seed=1)
    assert b.mean_rounds < a.mean_rounds


def test_push_rounds_match_theory():
    n = 512
    s = run_trials(complete_graph(n), "push", 20, seed=7)
    assert abs(s.mean_rounds - push_rounds_estimate(n)) < 4


def test_mean_field_tracks_simulation():
    n = 1000
    curve = mean_field_curve("push-pull", n, 12)
    res = spread(complete_graph(n), "push-pull", rng(8))
    mid = res.informed_history[6] / n
    assert abs(mid - curve[6]) < 0.35
    assert mean_field_rounds("push-pull", n) < mean_field_rounds("push", n)


def test_loss_slows_and_crash_handled():
    g = complete_graph(300)
    clean = run_trials(g, "push-pull", 15, seed=3)
    lossy = run_trials(g, "push-pull", 15, seed=3, loss=0.5)
    assert lossy.mean_rounds > clean.mean_rounds
    crashed = list(range(250, 300))
    r = spread(g, "push-pull", rng(9), crashed=crashed)
    assert r.n_alive == 250 and r.rounds_to_full is not None


def test_ring_is_slow():
    r = spread(ring_lattice(100, 1), "push-pull", rng(10), max_rounds=500)
    assert r.rounds_to_full is not None and r.rounds_to_full >= 25


def test_invalid_args():
    g = complete_graph(5)
    with pytest.raises(ValueError):
        spread(g, "flood", rng())
    with pytest.raises(ValueError):
        spread(g, "push", rng(), crashed=[0])
    with pytest.raises(ValueError):
        rumor_mongering(g, rng(), k=0)


def test_rumor_residue_theory():
    assert rumor_residue(1) == pytest.approx(0.2032, abs=1e-3)
    assert rumor_residue(3) < rumor_residue(2) < rumor_residue(1)
    s = rumor_residue(2)
    assert s == pytest.approx(math.exp(-3 * (1 - s)), abs=1e-9)


def test_rumor_mongering_matches_theory():
    g = complete_graph(2000)
    res = np.mean([rumor_mongering(g, rng(i), k=2).residue for i in range(5)])
    assert abs(res - rumor_residue(2)) < 0.03


@pytest.mark.parametrize("variant", ["feedback-coin", "feedback-counter", "blind-coin"])
def test_rumor_variants_terminate(variant):
    r = rumor_mongering(complete_graph(300), rng(11), k=3, variant=variant)
    assert 0.0 <= r.residue < 0.5 and r.rounds < 1000


def test_push_sum_mass_conservation_and_convergence():
    g = complete_graph(100)
    x = rng(12).normal(10, 3, 100)
    res = push_sum(g, x, rng(13), tol=1e-6)
    assert res.converged
    assert res.mass_s == pytest.approx(x.sum(), rel=1e-9)
    assert res.mass_w == pytest.approx(100.0, rel=1e-9)
    assert np.allclose(res.estimates, x.mean(), atol=1e-4)


def test_push_sum_sparse_graph_slower():
    x = rng(14).random(60)
    fast = push_sum(complete_graph(60), x, rng(15))
    slow = push_sum(ring_lattice(60, 2), x, rng(15), tol=1e-4, max_rounds=3000)
    fast = push_sum(complete_graph(60), x, rng(15), tol=1e-4)
    assert slow.converged and slow.rounds > fast.rounds


def test_anti_entropy_converges_lww():
    gen = rng(16)
    reps = [Replica(i) for i in range(30)]
    for i, r in enumerate(reps):
        r.put(f"k{i % 5}", f"v{i}", ts=i)
    rounds = run_anti_entropy(reps, gen)
    assert rounds > 0
    for k in range(5):
        winner = max(i for i in range(30) if i % 5 == k)
        assert all(r.data[f"k{k}"][2] == f"v{winner}" for r in reps)


def test_replica_merge_and_digest():
    a, b = Replica(0), Replica(1)
    a.put("x", "1", 5)
    b.put("x", "2", 5)  # same ts, higher origin id wins
    assert a.digest() != b.digest()
    assert a.merge_entry("x", b.data["x"]) and not b.merge_entry("x", a.data["x"])
    assert a.digest() == b.digest()


def test_determinism():
    g = complete_graph(100)
    a = spread(g, "push-pull", rng(99))
    b = spread(g, "push-pull", rng(99))
    assert a.informed_history == b.informed_history and a.messages == b.messages
