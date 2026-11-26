import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from model_based_rl import (DynaQ, DynaQPlus, GridWorld, PrioritizedSweeping,
                            QLearning, TabularModel, blocking_maze_layouts,
                            policy_iteration, run_episodes, run_steps,
                            value_iteration)
from model_based_rl.dp import optimal_steps, policy_evaluation
from model_based_rl.envs import dyna_maze
from model_based_rl.experiments import greedy_rollout_length


def test_env_walls_and_bounds():
    env = GridWorld(["S#G"])
    rng = np.random.default_rng(0)
    env.reset()
    s, r, d = env.step(1, rng)  # wall on the right
    assert s == env.start and r == 0 and not d
    s, _, _ = env.step(0, rng)  # top boundary
    assert s == env.start


def test_env_goal_reward():
    env = GridWorld(["SG"])
    s, r, d = env.step(1, np.random.default_rng(0))
    assert (s, r, d) == (1, 1.0, True)


def test_transition_model_is_stochastic_matrix():
    env = GridWorld(dyna_maze(), slip=0.2)
    P, R, term = env.transition_model()
    assert np.allclose(P.sum(axis=2), 1.0)
    assert term.sum() == 1 and (R >= 0).all()


def test_layout_validation():
    with pytest.raises(ValueError):
        GridWorld(["..."])
    with pytest.raises(ValueError):
        GridWorld(["S.G", ".."])
    with pytest.raises(ValueError):
        GridWorld(["SG"], slip=1.0)


def test_value_iteration_matches_policy_iteration():
    env = GridWorld(dyna_maze(), slip=0.1)
    P, R, term = env.transition_model()
    V, Q, _ = value_iteration(P, R, term, 0.95)
    V2, pol, _ = policy_iteration(P, R, term, 0.95)
    assert np.allclose(V, V2, atol=1e-7)
    assert np.allclose(policy_evaluation(P, R, term, pol, 0.95), V2)


def test_optimal_value_known_corridor():
    env = GridWorld(["S...G"])
    P, R, term = env.transition_model()
    V, _, _ = value_iteration(P, R, term, 0.9)
    assert V[env.start] == pytest.approx(0.9**3)
    assert optimal_steps(P, R, term, env.start, 0.9) == 4


def test_model_last_vs_counts():
    m = TabularModel("last")
    m.update(0, 1, 0.0, 2, False)
    m.update(0, 1, 1.0, 3, True)
    assert m.sample(0, 1, np.random.default_rng(0)) == (1.0, 3, True)
    assert len(m) == 1 and (0, 1) in m
    assert (0, 1) not in m.predecessors[2]
    c = TabularModel("counts")
    for _ in range(3):
        c.update(0, 0, 0.0, 1, False)
    c.update(0, 0, 1.0, 2, True)
    dist = {(s2, d): (p, r) for p, r, s2, d in c.distribution(0, 0)}
    assert dist[(1, False)][0] == pytest.approx(0.75)
    assert dist[(2, True)] == (pytest.approx(0.25), 1.0)
    with pytest.raises(ValueError):
        TabularModel("x")


def test_q_update_rule():
    ag = QLearning(3, 2, alpha=0.5, gamma=0.9, seed=0)
    ag.Q[1] = [0.0, 2.0]
    ag.observe(0, 0, 1.0, 1, False)
    assert ag.Q[0, 0] == pytest.approx(0.5 * (1 + 0.9 * 2))
    ag.observe(0, 1, 1.0, 1, True)
    assert ag.Q[0, 1] == pytest.approx(0.5)


def test_determinism():
    out = []
    for _ in range(2):
        env = GridWorld(dyna_maze())
        ag = DynaQ(env.n_states, 4, alpha=0.5, seed=3, n_planning=5)
        out.append(run_episodes(env, ag, 10, seed=3))
    assert np.array_equal(out[0], out[1])


def _mean_curve(mk, n_ep=20):
    return np.mean(
        [run_episodes(GridWorld(dyna_maze()), mk(s), n_ep, seed=s) for s in range(5)], axis=0
    ).mean()


def test_planning_beats_direct_learning():
    n = GridWorld(dyna_maze()).n_states
    q = _mean_curve(lambda s: QLearning(n, 4, alpha=0.5, seed=s))
    d = _mean_curve(lambda s: DynaQ(n, 4, alpha=0.5, seed=s, n_planning=20))
    assert d < 0.6 * q


def test_prioritized_sweeping_efficient():
    n = GridWorld(dyna_maze()).n_states
    q = _mean_curve(lambda s: QLearning(n, 4, alpha=0.5, seed=s))
    ps = _mean_curve(lambda s: PrioritizedSweeping(n, 4, seed=s, n_planning=5))
    assert ps < 0.5 * q


def test_agents_converge_to_optimal_policy_length():
    env = GridWorld(dyna_maze())
    P, R, term = env.transition_model()
    opt = optimal_steps(P, R, term, env.start)
    for cls, kw in [(DynaQ, {"n_planning": 20}), (PrioritizedSweeping, {"n_planning": 10})]:
        ag = cls(env.n_states, 4, alpha=0.5, epsilon=0.1, seed=1, **kw)
        run_episodes(env, ag, 100, seed=1)
        length = greedy_rollout_length(env, ag.policy())
        assert length is not None and opt <= length <= 1.5 * opt
        # Sampled backups can never overestimate the optimal values.
        V, Qs, _ = value_iteration(P, R, term)
        assert (ag.Q <= Qs + 1e-9).all()


def test_prioritized_sweeping_values_near_optimal():
    env = GridWorld(["S...G"])
    ag = PrioritizedSweeping(env.n_states, 4, gamma=0.9, epsilon=0.3, seed=0, n_planning=20)
    run_episodes(env, ag, 60, seed=0)
    P, R, term = env.transition_model()
    V, _, _ = value_iteration(P, R, term, 0.9)
    assert ag.Q[env.start].max() == pytest.approx(V[env.start], abs=1e-3)
    assert ag.planning_updates > 0


def test_counts_model_on_stochastic_env_learns():
    env = GridWorld(["S...G"], slip=0.2)
    ag = DynaQ(env.n_states, 4, alpha=0.3, gamma=0.9, epsilon=0.2, seed=2,
               n_planning=10, model_mode="counts")
    run_episodes(env, ag, 150, seed=2)
    assert ag.policy()[0] == 1 and ag.policy()[2] == 1


def test_dyna_q_plus_adapts_better_to_blocking():
    before, after = blocking_maze_layouts()
    tot = {}
    for name, cls, kw in [("q", DynaQ, {}), ("plus", DynaQPlus, {"kappa": 1e-3})]:
        res = []
        for sd in range(6):
            e = GridWorld(before)
            ag = cls(e.n_states, 4, alpha=1.0, seed=sd, n_planning=10, **kw)
            cum = run_steps(e, ag, 3000, seed=sd, switch_at=1000,
                            on_switch=lambda en: en.set_layout(after))
            res.append(cum[-1] - cum[999])
        tot[name] = np.mean(res)
    assert tot["plus"] > tot["q"]


def test_set_layout_shape_check():
    env = GridWorld(["S.G"])
    with pytest.raises(ValueError):
        env.set_layout(["S..G"])
    env.set_layout(["S#G"])
    assert env.walls[0, 1]


def test_render_contains_policy_arrows():
    env = GridWorld(["S.G"])
    assert env.render() == "S.G"
    assert env.render(np.array([1, 1, 1])) == ">>G"
