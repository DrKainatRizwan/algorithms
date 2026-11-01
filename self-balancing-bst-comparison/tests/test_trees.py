import math
import random

import pytest

from balanced_trees import (RedBlackTree, SplayTree, Treap, compare_all,
                            sequential, uniform, zipf_accesses)

FACTORIES = [lambda: Treap(seed=3), SplayTree, RedBlackTree]
IDS = ["treap", "splay", "rb"]


@pytest.fixture(params=FACTORIES, ids=IDS)
def tree(request):
    return request.param()


def test_empty(tree):
    assert len(tree) == 0 and not tree
    assert tree.height() == 0
    assert tree.minimum() is None and tree.maximum() is None
    assert tree.get(1, "d") == "d"
    assert not tree.delete(1)
    tree.validate()


def test_insert_get_replace(tree):
    assert tree.insert(5, "a")
    assert not tree.insert(5, "b")
    assert tree.get(5) == "b" and len(tree) == 1 and 5 in tree and 6 not in tree


def test_matches_dict_under_random_ops(tree):
    rng = random.Random(11)
    ref = {}
    for step in range(3000):
        k = rng.randrange(300)
        op = rng.random()
        if op < 0.5:
            assert tree.insert(k, step) == (k not in ref)
            ref[k] = step
        elif op < 0.8:
            assert tree.delete(k) == (k in ref)
            ref.pop(k, None)
        else:
            assert tree.get(k) == ref.get(k)
        if step % 250 == 0:
            tree.validate()
    tree.validate()
    assert list(tree.items()) == sorted(ref.items())
    assert len(tree) == len(ref)


def test_delete_everything(tree):
    keys = uniform(200, seed=2)
    for k in keys:
        tree.insert(k)
    for k in uniform(200, seed=5):
        assert tree.delete(k)
        tree.validate()
    assert len(tree) == 0 and tree.height() == 0


def test_floor_ceiling_range(tree):
    for k in range(0, 100, 10):
        tree.insert(k)
    assert tree.floor(35) == 30 and tree.ceiling(35) == 40
    assert tree.floor(-1) is None and tree.ceiling(91) is None
    assert tree.floor(40) == 40 and tree.ceiling(40) == 40
    assert tree.range(15, 55) == [20, 30, 40, 50]
    assert tree.range(200, 300) == []
    assert tree.minimum() == 0 and tree.maximum() == 90


def test_red_black_height_bound_on_sequential():
    t = RedBlackTree()
    n = 4096
    for k in sequential(n):
        t.insert(k)
    t.validate()
    assert t.height() <= 2 * math.log2(n + 1)
    assert t.black_height() >= math.log2(n + 1) / 2


def test_red_black_rotation_limits():
    t = RedBlackTree()
    worst = 0
    for k in uniform(1000, seed=4):
        before = t.rotations
        t.insert(k)
        worst = max(worst, t.rotations - before)
    assert worst <= 2
    worst = 0
    for k in uniform(1000, seed=9):
        before = t.rotations
        t.delete(k)
        worst = max(worst, t.rotations - before)
    assert worst <= 3


def test_treap_height_logarithmic_on_sorted_input():
    t = Treap(seed=1)
    n = 4096
    for k in sequential(n):
        t.insert(k)
    t.validate()
    assert t.height() < 5 * math.log2(n)


def test_treap_rank_select():
    t = Treap(seed=8)
    keys = uniform(500, seed=3)
    for k in keys:
        t.insert(k * 2)
    for i in range(500):
        assert t.select(i) == 2 * i
        assert t.rank(2 * i) == i
        assert t.rank(2 * i + 1) == i + 1
    with pytest.raises(IndexError):
        t.select(500)


def test_treap_split_concat_roundtrip():
    t = Treap(seed=2)
    for k in uniform(300, seed=6):
        t.insert(k)
    left, right = t.split(120)
    left.validate(); right.validate()
    assert left.keys() == list(range(120)) and right.keys() == list(range(120, 300))
    assert len(t) == 0
    joined = Treap.concat(left, right)
    joined.validate()
    assert joined.keys() == list(range(300))
    a, b = Treap(), Treap()
    a.insert(5); b.insert(3)
    with pytest.raises(ValueError):
        Treap.concat(a, b)


def test_treap_deterministic_given_seed():
    def build(seed):
        t = Treap(seed=seed)
        for k in range(100):
            t.insert(k)
        return t.height()
    assert build(4) == build(4)


def test_splay_moves_accessed_key_to_root():
    t = SplayTree()
    for k in uniform(100, seed=1):
        t.insert(k)
    t.get(42)
    assert t._root.key == 42
    t.insert(1000)
    assert t._root.key == 1000
    t.get(-5)  # miss: neighbour surfaces, structure stays valid
    t.validate()


def test_splay_sequential_insert_is_a_path_but_access_is_cheap_after_repeat():
    t = SplayTree()
    for k in sequential(500):
        t.insert(k)
    assert t.height() == 500  # worst-case shape, amortisation pays elsewhere
    t.get(0)
    t.reset_counters()
    for _ in range(100):
        t.get(0)
    assert t.comparisons <= 2 * 100  # root hit, O(1) each


def test_splay_beats_static_trees_on_skewed_access():
    res = {(r.workload, r.name): r for r in compare_all(n=1000, seed=3)}
    assert res[("working-set", "splay")].access_cmp < res[("working-set", "red-black")].access_cmp
    assert res[("zipf", "splay")].access_cmp < res[("zipf", "red-black")].access_cmp * 1.2


def test_zipf_generator_is_skewed_and_deterministic():
    a = zipf_accesses(100, 2000, seed=1)
    assert a == zipf_accesses(100, 2000, seed=1)
    assert a.count(0) > a.count(50) * 5


def test_string_keys(tree):
    for w in "the quick brown fox jumps over the lazy dog".split():
        tree.insert(w)
    tree.validate()
    assert tree.keys() == sorted(set("the quick brown fox jumps over the lazy dog".split()))
