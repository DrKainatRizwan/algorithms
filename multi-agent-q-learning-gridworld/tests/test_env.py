import numpy as np
import pytest

from marl_gridworld import CoopGridWorld, make_env, parse_layout, optimal_makespan, bfs_distances, moving_average

UP, DOWN, LEFT, RIGHT, STAY = range(5)


def small(rows, **kw):
    w, s, g = parse_layout(rows)
    return CoopGridWorld(w, s, g, **kw)


def test_parse_layout_roundtrip():
    walls, starts, goals = parse_layout(["0.#", "..a"])
    assert walls.sum() == 1 and starts == [(0, 0)] and goals == [(1, 2)]


def test_parse_rejects_bad_layouts():
    with pytest.raises(ValueError):
        parse_layout(["0.", "..a"[:1]])
    with pytest.raises(ValueError):
        parse_layout(["0?a"])
    with pytest.raises(ValueError):
        parse_layout(["0.."])


def test_wall_and_bounds_block_movement():
    env = small(["0#a"])
    env.reset()
    env.step([RIGHT])
    assert env.pos[0] == (0, 0)
    env.step([UP])
    assert env.pos[0] == (0, 0)


def test_same_cell_conflict_blocks_both():
    env = small(["0.1", "a.b"])
    env.reset()
    _, r, _, info = env.step([RIGHT, LEFT])
    assert env.pos == [(0, 0), (0, 2)] and info["blocked"] == 2 and r < -0.3


def test_swap_is_blocked():
    env = small(["01", "ba"])
    env.reset()
    env.step([RIGHT, LEFT])
    assert env.pos == [(0, 0), (0, 1)]


def test_moving_into_stationary_agent_blocked():
    env = small(["01.", "ab."])
    env.reset()
    env.step([RIGHT, STAY])
    assert env.pos[0] == (0, 0)


def test_chain_follow_blocked_by_blocked_leader():
    env = small(["012", "abc"])
    env.reset()
    env.step([RIGHT, RIGHT, STAY])
    assert env.pos == [(0, 0), (0, 1), (0, 2)]


def test_goal_reached_removes_agent_and_gives_bonus():
    env = small(["0a"])
    env.reset()
    _, r, done, info = env.step([RIGHT])
    assert done and info["success"] and not env.active[0]
    assert r == pytest.approx(-0.05 + 1.0 + 3.0)


def test_timeout_ends_episode():
    env = small(["0.a"], max_steps=3)
    env.reset()
    for _ in range(3):
        _, _, done, info = env.step([STAY])
    assert done and not info["success"]


def test_local_observation_sees_neighbours():
    env = small(["01.", "ab."])
    env.reset()
    assert env.observe(0, "local") == ((0, 0), (0, 0, 0, 1))
    assert env.observe(0, "own") == ((0, 0),)
    assert env.observe(0, "global") == ((0, 0), (0, 1))
    with pytest.raises(ValueError):
        env.observe(0, "bogus")


def test_bfs_and_makespan():
    env = make_env("bay")
    assert optimal_makespan(env) == 9
    d = bfs_distances(env.walls, env.starts[0])
    assert d[env.goals[0]] == 9 and d[env.starts[0]] == 0


def test_moving_average():
    out = moving_average([1, 2, 3, 4], 2)
    assert np.allclose(out, [1, 1.5, 2.5, 3.5])
