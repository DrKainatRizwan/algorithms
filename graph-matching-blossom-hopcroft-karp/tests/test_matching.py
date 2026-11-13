import random

import pytest

from graph_matching import (
    all_stable_matchings, blocking_pairs, brute_force_max_matching,
    edmonds_blossom, gale_shapley, greedy_matching, hall_violator,
    hopcroft_karp, hospitals_residents, is_matching, is_vertex_cover,
    konig_vertex_cover, kuhn_matching, random_bipartite, random_graph,
    random_preferences,
)
from graph_matching.bench import general_size, matching_size


def bip_edges(adj):
    return [(u, v) for u, a in enumerate(adj) for v in a]


@pytest.mark.parametrize("seed", range(10))
def test_hk_equals_kuhn(seed):
    adj = random_bipartite(30, 25, 0.1, seed)
    assert matching_size(hopcroft_karp(adj, 25)) == matching_size(kuhn_matching(adj, 25))


def test_hk_valid_matching():
    adj = random_bipartite(40, 40, 0.08, 1)
    m = hopcroft_karp(adj, 40)
    used = [v for v in m if v != -1]
    assert len(used) == len(set(used))
    assert all(m[u] in adj[u] for u in range(40) if m[u] != -1)


def test_hk_perfect_on_complete():
    adj = [list(range(6)) for _ in range(6)]
    assert matching_size(hopcroft_karp(adj, 6)) == 6


def test_hk_empty_and_isolated():
    assert hopcroft_karp([[], []], 3) == [-1, -1]
    assert hopcroft_karp([], 0) == []


def test_hk_needs_augmentation():
    adj = [[0, 1], [0]]
    assert matching_size(hopcroft_karp(adj, 2)) == 2


@pytest.mark.parametrize("seed", range(8))
def test_konig_cover(seed):
    adj = random_bipartite(15, 12, 0.15, seed)
    m = hopcroft_karp(adj, 12)
    left, right = konig_vertex_cover(adj, 12, m)
    assert len(left) + len(right) == matching_size(m)
    for u, v in bip_edges(adj):
        assert u in left or v in right


def test_hall_violator():
    adj = [[0], [0], [0, 1]]
    m = hopcroft_karp(adj, 2)
    s = hall_violator(adj, 2, m)
    assert s is not None
    nbrs = {v for u in s for v in adj[u]}
    assert len(nbrs) < len(s)
    assert hall_violator([[0], [1]], 2, hopcroft_karp([[0], [1]], 2)) is None


@pytest.mark.parametrize("seed", range(25))
def test_blossom_matches_brute_force(seed):
    n = 6 + seed % 9
    adj = random_graph(n, 0.25 + 0.02 * (seed % 5), seed)
    mate = edmonds_blossom(adj)
    assert is_matching(adj, mate)
    assert general_size(mate) == brute_force_max_matching(adj)


def test_blossom_odd_cycle():
    adj = [[1, 4], [0, 2], [1, 3], [2, 4], [3, 0]]
    assert general_size(edmonds_blossom(adj, warm_start=False)) == 2


def test_blossom_two_triangles():
    edges = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5), (5, 6), (6, 4)]
    adj = [[] for _ in range(7)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    for ws in (True, False):
        mate = edmonds_blossom(adj, warm_start=ws)
        assert is_matching(adj, mate)
        assert general_size(mate) == 3


def test_blossom_on_bipartite_agrees_with_hk():
    nl, nr = 12, 12
    b = random_bipartite(nl, nr, 0.2, 5)
    adj = [[nl + v for v in a] for a in b] + [[] for _ in range(nr)]
    for u, a in enumerate(b):
        for v in a:
            adj[nl + v].append(u)
    assert general_size(edmonds_blossom(adj)) == matching_size(hopcroft_karp(b, nr))


def test_greedy_is_maximal_not_larger():
    adj = random_graph(14, 0.3, 3)
    g = greedy_matching(adj)
    assert is_matching(adj, g)
    assert general_size(g) <= brute_force_max_matching(adj)
    for u in range(14):
        for v in adj[u]:
            assert g[u] != -1 or g[v] != -1


def test_is_matching_rejects_bad():
    adj = [[1], [0], []]
    assert not is_matching(adj, [1, -1, -1])
    assert not is_matching(adj, [2, -1, 0])


def test_is_vertex_cover():
    assert is_vertex_cover([(0, 1), (1, 2)], {1})
    assert not is_vertex_cover([(0, 1), (2, 3)], {1})


@pytest.mark.parametrize("seed", range(10))
def test_gale_shapley_stable(seed):
    a, b = random_preferences(12, seed)
    m = gale_shapley(a, b)
    assert sorted(m) == list(range(12))
    assert blocking_pairs(m, a, b) == []


def test_gale_shapley_proposer_optimal():
    for seed in range(8):
        a, b = random_preferences(6, seed)
        m = gale_shapley(a, b)
        for other in all_stable_matchings(a, b):
            for p in range(6):
                assert a[p].index(m[p]) <= a[p].index(other[p])


def test_blocking_pair_detected():
    a = [[0, 1], [0, 1]]
    b = [[0, 1], [0, 1]]
    assert blocking_pairs([1, 0], a, b) == [(0, 0)]


def test_all_stable_limit():
    a, b = random_preferences(9, 0)
    with pytest.raises(ValueError):
        all_stable_matchings(a, b)


def test_hospitals_residents_capacity_and_stability():
    rng = random.Random(4)
    nr, nh = 20, 4
    caps = [6, 5, 5, 4]
    rp = [rng.sample(range(nh), nh) for _ in range(nr)]
    hp = [rng.sample(range(nr), nr) for _ in range(nh)]
    res = hospitals_residents(rp, hp, caps)
    assert all(len(r) <= c for r, c in zip(res, caps))
    assert sum(len(r) for r in res) == nr
    where = {r: h for h, rs in enumerate(res) for r in rs}
    for r in range(nr):
        for h in rp[r]:
            if h == where[r]:
                break
            full = len(res[h]) == caps[h]
            worst = max(hp[h].index(x) for x in res[h])
            assert full and worst < hp[h].index(r)
