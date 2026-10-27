import itertools
import math
import random

import numpy as np
import pytest

from dp_toolkit import *


def rnd_items(rng, n, wmax=12, vmax=30):
    return [rng.randint(1, wmax) for _ in range(n)], [rng.randint(1, vmax) for _ in range(n)]


def test_knapsack01_matches_bruteforce():
    rng = random.Random(1)
    for _ in range(40):
        w, v = rnd_items(rng, rng.randint(1, 9))
        cap = rng.randint(0, 30)
        res = knapsack_01(w, v, cap)
        assert res.value == knapsack_01_bruteforce(w, v, cap)
        assert res.weight(w) <= cap
        assert sum(c * x for c, x in zip(res.counts, v)) == res.value


def test_knapsack01_edge_cases():
    assert knapsack_01([], [], 5).value == 0
    assert knapsack_01([10], [5], 3).value == 0
    assert knapsack_01([0], [7], 0).value == 7
    with pytest.raises(ValueError):
        knapsack_01([1, 2], [1], 3)


def test_unbounded_known_value_and_reconstruction():
    res = knapsack_unbounded([5, 10, 15], [10, 30, 20], 100)
    assert res.value == 300
    assert res.weight([5, 10, 15]) <= 100
    with pytest.raises(ValueError):
        knapsack_unbounded([0], [1], 5)


def test_unbounded_dominates_01():
    rng = random.Random(2)
    for _ in range(20):
        w, v = rnd_items(rng, 5)
        assert knapsack_unbounded(w, v, 25).value >= knapsack_01(w, v, 25).value


def test_bounded_equals_expanded_01():
    rng = random.Random(3)
    for _ in range(25):
        n = 4
        w, v = rnd_items(rng, n)
        lim = [rng.randint(0, 5) for _ in range(n)]
        cap = rng.randint(5, 40)
        res = knapsack_bounded(w, v, lim, cap)
        ew = [w[i] for i in range(n) for _ in range(lim[i])]
        ev = [v[i] for i in range(n) for _ in range(lim[i])]
        assert res.value == knapsack_01(ew, ev, cap).value
        assert all(c <= l for c, l in zip(res.counts, lim))
        assert res.weight(w) <= cap


def test_subset_sum_bitset():
    nums = [3, 34, 4, 12, 5, 2]
    idx = subset_sum_bitset(nums, 9)
    assert idx is not None and sum(nums[i] for i in idx) == 9
    assert subset_sum_bitset([2, 4, 6], 5) is None
    assert subset_sum_bitset([], 0) == []


def test_edit_distance_known():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("", "abc") == 3
    assert edit_distance("abc", "abc") == 0
    assert edit_distance("flaw", "lawn") == 2


def test_edit_distance_weighted_symmetry():
    # swapping strings swaps insert/delete costs
    a, b = "intention", "execution"
    assert edit_distance(a, b, ins=2, dele=3, sub=4) == edit_distance(b, a, ins=3, dele=2, sub=4)
    assert edit_distance("ab", "", dele=5) == 10


def test_edit_script_round_trip():
    rng = random.Random(4)
    for _ in range(60):
        a = "".join(rng.choice("abc") for _ in range(rng.randint(0, 9)))
        b = "".join(rng.choice("abc") for _ in range(rng.randint(0, 9)))
        s = edit_script(a, b)
        assert len(s) == edit_distance(a, b)
        assert apply_script(a, s) == b


def test_banded_matches_full_when_within_band():
    rng = random.Random(5)
    for _ in range(50):
        a = "".join(rng.choice("ab") for _ in range(rng.randint(0, 12)))
        b = "".join(rng.choice("ab") for _ in range(rng.randint(0, 12)))
        d = edit_distance(a, b)
        assert banded_edit_distance(a, b, d) == d
        if d > 0:
            assert banded_edit_distance(a, b, d - 1) is None


def is_subseq(s, t):
    it = iter(t)
    return all(c in it for c in s)


def test_lcs_variants_agree():
    rng = random.Random(6)
    for _ in range(40):
        a = [rng.choice("ACGT") for _ in range(rng.randint(0, 14))]
        b = [rng.choice("ACGT") for _ in range(rng.randint(0, 14))]
        x, y = lcs(a, b), lcs_hirschberg(a, b)
        assert len(x) == len(y)
        assert is_subseq(y, a) and is_subseq(y, b)


def test_lis():
    assert len(lis([10, 9, 2, 5, 3, 7, 101, 18])) == 4
    assert lis([]) == []
    assert lis([5, 5, 5]) == [5]
    out = lis([3, 1, 4, 1, 5, 9, 2, 6])
    assert all(x < y for x, y in zip(out, out[1:])) and is_subseq(out, [3, 1, 4, 1, 5, 9, 2, 6])


def test_matrix_chain_classic():
    cost, split = matrix_chain([10, 30, 5, 60])
    assert cost == 4500
    assert parenthesize(split) == "((A0 x A1) x A2)"
    assert matrix_chain([40, 20, 30, 10, 30])[0] == 26000


def test_matrix_chain_vs_bruteforce():
    rng = random.Random(7)

    def brute(d, i, j):
        if i == j:
            return 0
        return min(brute(d, i, k) + brute(d, k + 1, j) + d[i] * d[k + 1] * d[j + 1] for k in range(i, j))

    for _ in range(15):
        d = [rng.randint(1, 20) for _ in range(rng.randint(2, 8))]
        assert matrix_chain(d)[0] == brute(d, 0, len(d) - 2)


def test_optimal_bst_clrs_example():
    p = [0.15, 0.10, 0.05, 0.10, 0.20]
    q = [0.05, 0.10, 0.05, 0.05, 0.05, 0.10]
    cost, root = optimal_bst(p, q)
    assert cost == pytest.approx(2.75)
    assert root[1][5] == 2


def test_burst_balloons():
    best, order = burst_balloons([3, 1, 5, 8])
    assert best == 167
    # replay order
    total = 0
    alive = list(range(4))
    nums = [3, 1, 5, 8]
    for idx in order:
        p = alive.index(idx)
        l = nums[alive[p - 1]] if p > 0 else 1
        r = nums[alive[p + 1]] if p + 1 < len(alive) else 1
        total += l * nums[idx] * r
        alive.remove(idx)
    assert total == best and sorted(order) == [0, 1, 2, 3]


def test_polygon_triangulation_square():
    sq = [(0, 0), (1, 0), (1, 1), (0, 1)]
    # one diagonal: two triangles of perimeter 2+sqrt2
    assert polygon_triangulation(sq) == pytest.approx(2 * (2 + math.sqrt(2)))


def test_held_karp_matches_permutations():
    rng = np.random.default_rng(8)
    for n in (2, 4, 6, 7):
        pts = rng.random((n, 2))
        d = np.linalg.norm(pts[:, None] - pts[None], axis=2)
        best, tour = held_karp(d)
        brute = min(
            sum(d[a, b] for a, b in zip((0, *p), (*p, 0))) for p in itertools.permutations(range(1, n))
        )
        assert best == pytest.approx(brute)
        assert sorted(tour) == list(range(n))
        assert sum(d[tour[i], tour[(i + 1) % n]] for i in range(n)) == pytest.approx(best)


def test_assignment_dp_vs_permutations():
    rng = np.random.default_rng(9)
    c = rng.integers(1, 50, (6, 6)).astype(float)
    best, a = assignment_dp(c)
    brute = min(sum(c[i, p[i]] for i in range(6)) for p in itertools.permutations(range(6)))
    assert best == brute and sorted(a) == list(range(6))


def test_sum_over_subsets():
    rng = random.Random(10)
    f = [rng.randint(0, 9) for _ in range(16)]
    g = sum_over_subsets(f, 4)
    for m in range(16):
        assert g[m] == sum(f[s] for s in range(16) if s & m == s)


def test_hamiltonian_path_counts():
    n = 5
    complete = [[int(i != j) for j in range(n)] for i in range(n)]
    assert count_hamiltonian_paths(complete) == math.factorial(n)
    path = [[int(abs(i - j) == 1) for j in range(n)] for i in range(n)]
    assert count_hamiltonian_paths(path) == 2


def test_min_set_cover():
    sets = [[0, 1, 2], [2, 3], [3, 4], [0, 4], [1, 3]]
    sel = min_set_cover(5, sets)
    assert sel is not None and len(sel) == 2
    assert set().union(*(sets[i] for i in sel)) == set(range(5))
    assert min_set_cover(3, [[0], [1]]) is None


def test_memo_counts_subproblems():
    m = Memo(lambda n: n if n < 2 else m(n - 1) + m(n - 2))
    assert m(30) == 832040
    assert m.misses == 31 and m.hits > 0
