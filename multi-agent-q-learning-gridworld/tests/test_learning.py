import numpy as np

from marl_gridworld import (EpsilonSchedule, IndependentQLearner, JointQLearner,
                            evaluate, make_env, train)


def test_epsilon_schedule_linear():
    s = EpsilonSchedule(1.0, 0.1, 10)
    assert s(0) == 1.0 and abs(s(5) - 0.55) < 1e-12 and abs(s(100) - 0.1) < 1e-12


def test_td_update_moves_toward_target():
    a = IndependentQLearner(5, alpha=0.5, gamma=0.9)
    td = a.update("s", 2, 1.0, "t", True)
    assert td == 1.0 and a.values("s")[2] == 0.5


def test_hysteretic_negative_updates_are_smaller():
    a = IndependentQLearner(5, alpha=0.5, beta=0.1)
    a.update("s", 0, -1.0, "t", True)
    assert np.isclose(a.values("s")[0], -0.1)


def test_greedy_tie_breaking_is_seeded():
    a = IndependentQLearner(5, seed=3)
    b = IndependentQLearner(5, seed=3)
    assert [a.greedy("x") for _ in range(10)] == [b.greedy("x") for _ in range(10)]


def test_joint_action_indexing():
    j = JointQLearner(2, 5)
    assert len(j.joint) == 25
    j.update("s", [1, 3], 2.0, "t", True)
    assert j.values("s")[j.joint.index((1, 3))] > 0


def test_training_is_deterministic():
    def run():
        env = make_env("crossing")
        ag = [IndependentQLearner(5, seed=i) for i in range(2)]
        return train(env, ag, 150).returns
    assert run() == run()


def test_joint_learner_solves_crossing():
    env = make_env("crossing")
    j = JointQLearner(2, 5, seed=0)
    log = train(env, j, 1500, obs_mode="global")
    assert log.success_rate(200) > 0.9
    res = evaluate(env, j)
    assert res["success"] == 1.0 and res["collisions"] == 0


def test_independent_learners_with_local_view_solve_bay():
    env = make_env("bay")
    ag = [IndependentQLearner(5, seed=i) for i in range(2)]
    log = train(env, ag, 3000, obs_mode="local")
    assert log.success_rate(300) > 0.8
    assert evaluate(env, ag, 3)["success"] == 1.0


def test_bay_requires_coordination_blind_agents_fail():
    env = make_env("bay")
    ag = [IndependentQLearner(5, seed=i) for i in range(2)]
    log = train(env, ag, 2000, obs_mode="own")
    assert log.success_rate(300) < 0.2


def test_learning_improves_returns():
    env = make_env("crossing")
    ag = [IndependentQLearner(5, seed=i) for i in range(2)]
    log = train(env, ag, 1500)
    assert np.mean(log.returns[-200:]) > np.mean(log.returns[:200])
