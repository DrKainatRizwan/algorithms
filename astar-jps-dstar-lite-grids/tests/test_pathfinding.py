import math

import numpy as np
import pytest

from grid_pathfinding import (DStarLite, Grid, astar, cave_map, dijkstra, jps,
                              manhattan, maze_map, navigate_dstar, navigate_replan,
                              octile, random_obstacles, random_query, room_map)
from grid_pathfinding.grid import SQRT2
from grid_pathfinding.jps import JumpPointSearch


def maps(seed):
    return [random_obstacles(30, 30, 0.1 + 0.1 * (seed % 4), seed), cave_map(40, 30, seed),
            room_map(50, 40, seed), maze_map(31, 31, seed, 0.1)]


def test_corner_cutting_forbidden():
    g = Grid.from_strings([".#", "#."])
    assert all((nx, ny) != (1, 1) for nx, ny, _ in g.neighbors(0, 0))
    assert astar(g, (0, 0), (1, 1)).path is None


def test_open_field_cost_is_octile():
    g = Grid(20, 20)
    r = astar(g, (0, 0), (10, 4))
    assert r.cost == pytest.approx(octile((0, 0), (10, 4)))
    assert r.cost == pytest.approx(6 + 4 * SQRT2)


def test_astar_matches_dijkstra_and_expands_less():
    for seed in range(15):
        for g in maps(seed):
            s, t = random_query(g, np.random.default_rng(seed), 0.4)
            a, d = astar(g, s, t), dijkstra(g, s, t)
            assert a.cost == pytest.approx(d.cost)
            assert a.expanded <= d.expanded
            assert g.is_valid_path(a.path)
            assert g.path_cost(a.path) == pytest.approx(a.cost)


def test_unreachable_and_blocked_endpoints():
    g = Grid.from_strings(["..#..", "..#..", "..#.."])
    assert not astar(g, (0, 0), (4, 0)).found
    assert not jps(g, (0, 0), (4, 0)).found
    assert not astar(g, (2, 0), (0, 0)).found
    assert not jps(g, (0, 0), (2, 1)).found


def test_start_equals_goal():
    g = Grid(5, 5)
    assert astar(g, (2, 2), (2, 2)).path == [(2, 2)]
    assert jps(g, (2, 2), (2, 2)).cost == 0.0


def test_weighted_astar_bound_and_effort():
    g = random_obstacles(60, 60, 0.2, 3)
    s, t = random_query(g, np.random.default_rng(3))
    opt = astar(g, s, t)
    w = astar(g, s, t, weight=3.0)
    assert opt.cost - 1e-9 <= w.cost <= 3.0 * opt.cost + 1e-9
    assert w.expanded <= opt.expanded
    with pytest.raises(ValueError):
        astar(g, s, t, weight=0.5)


def test_manhattan_not_admissible_overestimates():
    assert manhattan((0, 0), (5, 5)) > octile((0, 0), (5, 5))


def test_jps_optimal_on_many_maps():
    for seed in range(40):
        for g in maps(seed):
            s, t = random_query(g, np.random.default_rng(seed + 7), 0.3)
            a, j = astar(g, s, t), jps(g, s, t)
            assert j.cost == pytest.approx(a.cost, abs=1e-9)
            assert g.is_valid_path(j.path)
            assert g.path_cost(j.path) == pytest.approx(j.cost)
            assert j.path[0] == s and j.path[-1] == t


def test_jps_expands_far_fewer_nodes_on_structured_maps():
    for g in (room_map(120, 120, 3), cave_map(120, 120, 3)):
        rng = np.random.default_rng(0)
        pairs = [random_query(g, rng) for _ in range(8)]
        a = sum(astar(g, s, t).expanded for s, t in pairs)
        j = sum(jps(g, s, t).expanded for s, t in pairs)
        assert j * 4 < a


def test_jps_expand_interpolates():
    path = JumpPointSearch.expand([(0, 0), (3, 3), (3, 6)])
    assert path == [(0, 0), (1, 1), (2, 2), (3, 3), (3, 4), (3, 5), (3, 6)]


def test_generators_deterministic_and_connected_rooms():
    assert (room_map(40, 40, 5).to_array() == room_map(40, 40, 5).to_array()).all()
    assert (cave_map(40, 40, 5).to_array() != cave_map(40, 40, 6).to_array()).any()
    g = room_map(60, 40, 2)
    cells = g.free_cells()
    assert len(g.reachable(cells[0])) == len(cells)


def test_maze_is_odd_sized_and_solvable():
    g = maze_map(30, 30, 1, braid=0.0)
    assert g.width % 2 == 1 and g.height % 2 == 1
    assert len(g.reachable((1, 1))) == g.free_count()


def test_dstar_static_matches_astar():
    for seed in range(15):
        for g in maps(seed):
            s, t = random_query(g, np.random.default_rng(seed), 0.4)
            d = DStarLite(g.copy(), s, t)
            d.compute_shortest_path()
            assert d.cost_to_goal() == pytest.approx(astar(g, s, t).cost)
            p = d.extract_path()
            assert p is not None and g.path_cost(p) == pytest.approx(d.cost_to_goal())


def test_dstar_incremental_updates_match_fresh_search():
    rng = np.random.default_rng(4)
    g = random_obstacles(35, 35, 0.15, 4)
    s, t = random_query(g, rng, 0.7)
    d = DStarLite(g.copy(), s, t)
    d.compute_shortest_path()
    for _ in range(25):
        x, y = int(rng.integers(35)), int(rng.integers(35))
        if (x, y) in (s, t):
            continue
        d.update_cells([(x, y, bool(rng.random() < 0.6))])
        d.compute_shortest_path()
        ref = astar(d.grid, s, t).cost
        assert d.cost_to_goal() == pytest.approx(ref) or (math.isinf(ref) and math.isinf(d.cost_to_goal()))


def test_dstar_moving_start_keeps_correct_costs():
    rng = np.random.default_rng(8)
    g = random_obstacles(30, 30, 0.2, 8)
    s, t = random_query(g, rng, 0.8)
    d = DStarLite(g.copy(), s, t)
    d.compute_shortest_path()
    for step in range(10):
        nxt = d.next_step()
        if nxt is None or nxt == t:
            break
        d.move_start(nxt)
        bx, by = int(rng.integers(30)), int(rng.integers(30))
        if (bx, by) not in (nxt, t):
            d.update_cells([(bx, by, True)])
        d.compute_shortest_path()
        assert d.cost_to_goal() == pytest.approx(astar(d.grid, nxt, t).cost)


def test_dstar_detects_blocked_goal_region():
    g = Grid(10, 3)
    d = DStarLite(g, (0, 1), (9, 1))
    d.compute_shortest_path()
    assert d.cost_to_goal() == pytest.approx(9.0)
    d.update_cells([(5, y, True) for y in range(3)])
    d.compute_shortest_path()
    assert math.isinf(d.cost_to_goal()) and d.extract_path() is None


def test_navigation_reaches_goal_in_unknown_terrain():
    for seed in range(6):
        g = random_obstacles(40, 40, 0.2, seed)
        s, t = random_query(g, np.random.default_rng(seed), 0.7)
        a = navigate_dstar(g, s, t, radius=3)
        b = navigate_replan(g, s, t, radius=3)
        assert a.success and b.success
        for r in (a, b):
            assert r.trajectory[0] == s and r.trajectory[-1] == t
            assert g.is_valid_path(r.trajectory)
            assert g.path_cost(r.trajectory) == pytest.approx(r.travelled)
        assert a.travelled >= astar(g, s, t).cost - 1e-9


def test_navigation_with_full_knowledge_is_optimal():
    g = random_obstacles(40, 40, 0.2, 2)
    s, t = random_query(g, np.random.default_rng(2), 0.7)
    r = navigate_dstar(g, s, t, radius=1, known=g)
    assert r.travelled == pytest.approx(astar(g, s, t).cost)
    assert r.replans == 0


def test_navigation_fails_when_goal_walled_off():
    g = Grid.from_strings(["....#...", "....#...", "....#..."])
    r = navigate_dstar(g, (0, 1), (7, 1), radius=2)
    assert not r.success
