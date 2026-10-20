import random

import pytest

from skiplist_lab import GeometricLevels, HashLevels, SkipList


def build(n=500, seed=1, **kw):
    sl = SkipList(GeometricLevels(seed=seed, **kw))
    rng = random.Random(seed)
    keys = rng.sample(range(10 * n), n)
    for k in keys:
        sl.insert(k, k * 2)
    return sl, sorted(keys)


def test_insert_get_sorted_iteration():
    sl, keys = build()
    assert list(sl) == keys
    assert len(sl) == len(keys)
    assert all(sl[k] == 2 * k for k in keys)
    sl.check_invariants()


def test_overwrite_returns_false_and_updates():
    sl = SkipList()
    assert sl.insert(5, "a") is True
    assert sl.insert(5, "b") is False
    assert sl[5] == "b" and len(sl) == 1


def test_delete_matches_reference_model():
    rng = random.Random(7)
    sl, model = SkipList(GeometricLevels(seed=3)), set()
    for _ in range(3000):
        k = rng.randrange(150)
        if rng.random() < 0.55:
            assert sl.insert(k) == (k not in model)
            model.add(k)
        else:
            assert sl.delete(k) == (k in model)
            model.discard(k)
    assert list(sl) == sorted(model)
    sl.check_invariants()


def test_missing_key_errors():
    sl = SkipList()
    with pytest.raises(KeyError):
        sl[1]
    with pytest.raises(KeyError):
        del sl[1]
    with pytest.raises(KeyError):
        sl.pop_min()


def test_rank_and_kth_agree_with_sorted_order():
    sl, keys = build(400, seed=4)
    for i in range(0, 400, 7):
        assert sl.kth(i)[0] == keys[i]
        assert sl.rank(keys[i]) == i
    assert sl.kth(-1)[0] == keys[-1]
    with pytest.raises(IndexError):
        sl.kth(400)
    sl.check_invariants()


def test_spans_survive_deletions():
    sl, keys = build(300, seed=9)
    rng = random.Random(2)
    for k in rng.sample(keys, 150):
        sl.delete(k)
        keys.remove(k)
    sl.check_invariants()
    assert [sl.kth(i)[0] for i in range(len(keys))] == keys


def test_floor_ceiling_and_range():
    sl = SkipList()
    for k in (10, 20, 30, 40):
        sl.insert(k, str(k))
    assert sl.floor(25) == 20 and sl.floor(20) == 20 and sl.floor(5) is None
    assert sl.ceiling(25) == 30 and sl.ceiling(41) is None
    assert [k for k, _ in sl.range(15, 40)] == [20, 30]


def test_pop_min_drains_in_order():
    sl, keys = build(100, seed=5)
    out = [sl.pop_min()[0] for _ in range(100)]
    assert out == keys and len(sl) == 0 and sl.height == 1


def test_hash_levels_history_independent():
    keys = list(range(300))
    a, b = SkipList(HashLevels()), SkipList(HashLevels())
    for k in keys:
        a.insert(k)
    for k in reversed(keys):
        b.insert(k)
    for k in range(300, 400):  # insert then delete extras
        b.insert(k)
    for k in range(300, 400):
        b.delete(k)
    assert a.tower_heights() == b.tower_heights()
    assert a.level_sizes() == b.level_sizes()


def test_policy_validation():
    with pytest.raises(ValueError):
        GeometricLevels(p=1.0)
    with pytest.raises(ValueError):
        HashLevels(max_level=0)


def test_string_keys_with_hash_levels():
    sl = SkipList(HashLevels())
    words = ["pear", "apple", "fig", "kiwi"]
    for w in words:
        sl.insert(w)
    assert list(sl) == sorted(words)
