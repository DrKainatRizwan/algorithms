import numpy as np
import pytest

from deep_qlearning import (Adam, DQNAgent, DQNConfig, MLP, MazeEnv, PrioritizedReplayBuffer,
                            ReplayBuffer, evaluate, generate_maze, shortest_path_length, train)


def test_maze_shape_and_endpoints():
    m = generate_maze(5, seed=1)
    assert m.shape == (11, 11)
    assert m.is_free(*m.start) and m.is_free(*m.goal)


def test_maze_solvable_and_deterministic():
    for seed in range(5):
        assert shortest_path_length(generate_maze(6, seed=seed)) > 0
    assert (generate_maze(4, 7).walls == generate_maze(4, 7).walls).all()


def test_extra_openings_do_not_lengthen_path():
    a = generate_maze(6, seed=2)
    b = generate_maze(6, seed=2, extra_openings=10)
    assert shortest_path_length(b) <= shortest_path_length(a)
    assert b.walls.sum() < a.walls.sum()


def test_env_wall_bump_and_goal():
    env = MazeEnv(generate_maze(3, seed=0))
    env.reset()
    _, r, done = env.step(0)  # up from (1,1) is the border
    assert r == MazeEnv.WALL_REWARD and env.pos == env.maze.start and not done
    env.pos = (env.maze.goal[0] - 1, env.maze.goal[1])
    _, r, done = env.step(2)
    assert done and r == MazeEnv.GOAL_REWARD


def test_env_timeout_and_invalid_action():
    env = MazeEnv(generate_maze(3, seed=0), max_steps=3)
    env.reset()
    done = False
    for _ in range(3):
        _, _, done = env.step(0)
    assert done
    with pytest.raises(ValueError):
        env.step(9)


def test_replay_ring_overwrites():
    buf = ReplayBuffer(3, 2)
    for i in range(5):
        buf.add(np.full(2, i), 0, float(i), np.full(2, i), False)
    assert len(buf) == 3
    assert set(buf.r) == {2.0, 3.0, 4.0}
    s, a, r, s2, d = buf.sample(8)
    assert s.shape == (8, 2)


def test_prioritized_prefers_high_td():
    buf = PrioritizedReplayBuffer(10, 1, seed=0)
    for i in range(10):
        buf.add(np.array([i]), 0, 0.0, np.array([i]), False)
    buf.sample(10)
    buf.last_idx = np.arange(10)
    buf.update_priorities(np.array([0.0] * 9 + [50.0]))
    s, *_ = buf.sample(500)
    assert (s[:, 0] == 9).mean() > 0.5
    assert buf.last_weights.max() == pytest.approx(1.0)


def test_mlp_gradient_check():
    rng = np.random.default_rng(0)
    net = MLP([4, 5, 3], seed=0)
    x = rng.normal(size=(6, 4))
    t = rng.normal(size=(6, 3))
    loss = lambda: 0.5 * ((net.forward(x) - t) ** 2).sum()
    grads = net.backward(net.forward(x) - t)
    for p, g in zip(net.params, grads):
        flat = p.reshape(-1)
        for k in range(0, flat.size, 3):
            old = flat[k]
            flat[k] = old + 1e-6; lp = loss()
            flat[k] = old - 1e-6; lm = loss()
            flat[k] = old
            assert (lp - lm) / 2e-6 == pytest.approx(g.reshape(-1)[k], abs=1e-4)


def test_adam_minimises_quadratic():
    p = [np.array([5.0, -3.0])]
    opt = Adam(p, lr=0.1)
    for _ in range(500):
        opt.step([2 * p[0]])
    assert np.abs(p[0]).max() < 0.05


def test_adam_clipping_bounds_update():
    p = [np.zeros(2)]
    Adam(p, lr=1.0, clip=1.0).step([np.array([1e6, 0.0])])
    assert np.isfinite(p[0]).all()


def test_target_sync_and_soft_update():
    a, b = MLP([3, 4, 2], seed=0), MLP([3, 4, 2], seed=5)
    b.soft_update(a, 1.0)
    assert all((p == q).all() for p, q in zip(a.params, b.params))


def test_double_dqn_target_formula():
    ag = DQNAgent(4, 2, DQNConfig(gamma=0.5, double=True))
    s2 = np.eye(4)
    best = ag.q.forward(s2).argmax(axis=1)
    expected = 1.0 + 0.5 * ag.target.forward(s2)[np.arange(4), best]
    assert np.allclose(ag.targets(np.ones(4), s2, np.zeros(4)), expected)
    assert np.allclose(ag.targets(np.ones(4), s2, np.ones(4)), 1.0)


def test_epsilon_schedule():
    ag = DQNAgent(4, 2, DQNConfig(eps_start=1.0, eps_end=0.1, eps_decay_steps=100))
    assert ag.epsilon() == 1.0
    ag.steps = 50
    assert ag.epsilon() == pytest.approx(0.55)
    ag.steps = 1000
    assert ag.epsilon() == pytest.approx(0.1)


@pytest.mark.parametrize("kw", [dict(double=True), dict(double=False), dict(prioritized=True)])
def test_agent_learns_small_maze(kw):
    maze = generate_maze(3, seed=1)
    env = MazeEnv(maze, max_steps=80)
    ag = DQNAgent(env.obs_dim, env.n_actions, DQNConfig(seed=0, eps_decay_steps=3000, **kw))
    train(env, ag, 250)
    ev = evaluate(env, ag)
    assert ev["success"] == 1.0
    assert ev["steps"] <= shortest_path_length(maze) + 4
