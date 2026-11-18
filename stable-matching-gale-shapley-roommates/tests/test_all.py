import pytest
from stable_matching import *
from stable_matching.lattice import find_all_rotations
from stable_matching.preferences import validate_complete

def test_textbook_instance():
    men = [[0, 1, 2], [1, 0, 2], [0, 1, 2]]
    women = [[1, 0, 2], [0, 1, 2], [0, 1, 2]]
    r = gale_shapley(men, women)
    assert is_stable(men, women, r.match)
    assert r.match == [1, 0, 2] or sorted(r.match) == [0, 1, 2]

@pytest.mark.parametrize("seed", range(15))
def test_gs_stable_random(seed):
    men, women = random_marriage_instance(12, seed)
    r = gale_shapley(men, women)
    assert sorted(r.match) == list(range(12))
    assert blocking_pairs(men, women, r.match) == []

@pytest.mark.parametrize("seed", range(10))
def test_optimality_against_brute_force(seed):
    men, women = random_marriage_instance(6, seed)
    allm = brute_force_marriage(men, women)
    mr = rank_table(men)
    best = gale_shapley(men, women).match
    for m in allm:
        assert all(mr[i][best[i]] <= mr[i][m[i]] for i in range(6))

@pytest.mark.parametrize("seed", range(12))
def test_enumeration_matches_brute_force(seed):
    men, women = random_marriage_instance(6, seed)
    assert enumerate_stable_matchings(men, women) == sorted(brute_force_marriage(men, women))

def test_enumeration_n7_and_rotations():
    men, women = random_marriage_instance(7, 3)
    assert enumerate_stable_matchings(men, women) == sorted(brute_force_marriage(men, women))
    assert len(find_all_rotations(men, women)) >= len(find_rotations(men, women))

def test_cyclic_instance_has_many_matchings():
    n = 4
    men = [[(i + k) % n for k in range(n)] for i in range(n)]
    women = [[(i - k - 1) % n for k in range(n)] for i in range(n)]
    allm = enumerate_stable_matchings(men, women)
    assert len(allm) == len(brute_force_marriage(men, women)) > 1
    info = stable_matching_lattice(men, women)
    assert info["count"] == len(allm) and info["rotations"] >= 1
    assert find_rotations(men, women)

def test_criteria_are_stable_and_optimal():
    men, women = random_marriage_instance(7, 5)
    for f in (egalitarian_matching, minimum_regret_matching, sex_equal_matching):
        assert is_stable(men, women, f(men, women))
    mr, wr = rank_table(men), rank_table(women)
    cost = lambda m: sum(mr[i][m[i]] + wr[m[i]][i] for i in range(7))
    eg = egalitarian_matching(men, women)
    assert all(cost(eg) <= cost(m) for m in enumerate_stable_matchings(men, women))

def test_incomplete_lists():
    men = [[0], [0, 1], [1]]
    women = [[1, 0], [2, 1]]
    r = gale_shapley(men, women)
    assert r.match == [-1, 0, 1]
    assert is_stable(men, women, r.match)
    assert not is_stable(men, women, [0, -1, 1])

def test_validation():
    with pytest.raises(ValueError):
        validate_complete([[0, 0]], [[0]])

@pytest.mark.parametrize("seed", range(60))
def test_roommates_agree_with_brute_force(seed):
    prefs = random_roommates_instance(8, seed)
    bf = brute_force_roommates(prefs)
    m = irving_roommates(prefs)
    assert (m is not None) == bool(bf)
    if m is not None:
        assert roommates_blocking_pairs(prefs, m) == []

def test_roommates_known_unsolvable():
    prefs = [[1, 2, 3], [2, 0, 3], [0, 1, 3], [0, 1, 2]]
    assert irving_roommates(prefs) is None
    assert brute_force_roommates(prefs) == []

def test_roommates_odd():
    assert irving_roommates([[1, 2], [0, 2], [0, 1]]) is None

def test_roommates_solvable_instance():
    prefs = [[1, 2, 3], [0, 2, 3], [3, 0, 1], [2, 0, 1]]
    assert irving_roommates(prefs) == [1, 0, 3, 2]

@pytest.mark.parametrize("seed", range(10))
def test_hr_stable_both_orientations(seed):
    caps = [2, 1, 3, 2]
    res, hosp = random_hr_instance(10, caps, seed)
    a = hospitals_residents(res, hosp, caps, True)
    b = hospitals_residents(res, hosp, caps, False)
    for x in (a, b):
        assert hr_blocking_pairs(res, hosp, caps, x) == []
        assert all(len(x[h]) <= caps[h] for h in range(4))
    # rural hospitals theorem: same number of filled seats at each hospital
    assert [len(x) for x in a] == [len(x) for x in b]
    rr = rank_table(res)
    wa = {r: h for h, rs in enumerate(a) for r in rs}
    wb = {r: h for h, rs in enumerate(b) for r in rs}
    assert set(wa) == set(wb)
    assert all(rr[r][wa[r]] <= rr[r][wb[r]] for r in wa)

def test_proposal_stats_scale():
    s = proposal_statistics([10, 40], trials=5, seed=1)
    assert s[40][0] > s[10][0] > 10

@pytest.mark.parametrize("seed", range(30))
def test_roommates_large_valid(seed):
    prefs = random_roommates_instance(30, seed)
    m = irving_roommates(prefs)
    if m is not None:
        assert all(m[m[i]] == i and m[i] != i for i in range(30))
        assert roommates_blocking_pairs(prefs, m) == []
