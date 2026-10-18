import random

import pytest

from bplus_engine import (BPlusTree, BufferPool, KVStore, PageOverflow, bulk_load, chunk_sizes,
                          dumps, encode_key, loads, max_keys_for_page, record_trace)


@pytest.mark.parametrize("m", [3, 4, 5, 8, 33])
def test_random_insert_delete_matches_dict(m):
    rng = random.Random(m)
    t, ref = BPlusTree(m), {}
    for step in range(3000):
        k = rng.randrange(500)
        if rng.random() < 0.55:
            assert t.insert(k, step) == (k not in ref)
            ref[k] = step
        else:
            assert t.delete(k) == (k in ref)
            ref.pop(k, None)
        if step % 200 == 0:
            t.check()
    t.check()
    assert len(t) == len(ref)
    assert list(t.items()) == sorted(ref.items())


def test_get_contains_and_missing():
    t = BPlusTree(4)
    for i in range(0, 100, 2):
        t[i] = str(i)
    assert 10 in t and 11 not in t
    assert t.get(11, "x") == "x" and t[10] == "10"
    with pytest.raises(KeyError):
        t[11]
    with pytest.raises(KeyError):
        del t[11]


def test_overwrite_keeps_size():
    t = BPlusTree(4)
    assert t.insert(1, "a") and not t.insert(1, "b")
    assert len(t) == 1 and t[1] == "b"


def test_range_scan_bounds():
    t = BPlusTree(4)
    for i in range(100):
        t.insert(i, i)
    assert [k for k, _ in t.range(10, 15)] == [10, 11, 12, 13, 14]
    assert [k for k, _ in t.range(10, 15, include_hi=True)][-1] == 15
    assert [k for k, _ in t.range(95)] == [95, 96, 97, 98, 99]
    assert [k for k, _ in t.range(hi=3)] == [0, 1, 2]
    assert list(t.range(200, 300)) == []
    assert t.count_range(20, 30) == 10


def test_reverse_iteration_and_extremes():
    t = BPlusTree(5)
    keys = random.Random(1).sample(range(1000), 200)
    for k in keys:
        t.insert(k, -k)
    assert [k for k, _ in t.reversed_items()] == sorted(keys, reverse=True)
    assert t.min_item() == (min(keys), -min(keys))
    assert t.max_item()[0] == max(keys)


def test_floor_ceiling():
    t = BPlusTree(4)
    for k in range(0, 100, 10):
        t.insert(k, k)
    assert t.floor(35) == (30, 30) and t.ceiling(35) == (40, 40)
    assert t.floor(-1) is None and t.ceiling(91) is None
    assert t.floor(40) == (40, 40) and t.ceiling(40) == (40, 40)


def test_delete_all_collapses_height():
    t = BPlusTree(4)
    for i in range(500):
        t.insert(i, i)
    assert t.height() > 2
    for i in random.Random(3).sample(range(500), 500):
        t.delete(i)
        t.check()
    assert len(t) == 0 and t.height() == 1
    with pytest.raises(KeyError):
        t.min_item()


def test_pop_min_gives_sorted_order():
    t = BPlusTree(6)
    keys = random.Random(9).sample(range(10000), 300)
    for k in keys:
        t.insert(k, None)
    assert [t.pop_min()[0] for _ in range(300)] == sorted(keys)


def test_invalid_order():
    with pytest.raises(ValueError):
        BPlusTree(2)


@pytest.mark.parametrize("n", [0, 1, 2, 5, 17, 100, 1000, 4097])
@pytest.mark.parametrize("m,fill", [(3, 0.5), (4, 1.0), (7, 0.7), (32, 0.9)])
def test_bulk_load_valid(n, m, fill):
    t = bulk_load(((i * 3, i) for i in range(n)), m, fill)
    t.check()
    assert len(t) == n
    assert [v for _, v in t.items()] == list(range(n))


def test_bulk_loaded_tree_supports_updates():
    t = bulk_load(((i * 2, i) for i in range(1000)), 8, 1.0)
    rng = random.Random(4)
    for _ in range(1500):
        k = rng.randrange(2200)
        if rng.random() < 0.5:
            t.insert(k, k)
        else:
            t.delete(k)
    t.check()


def test_bulk_load_rejects_unsorted():
    with pytest.raises(ValueError):
        bulk_load([(2, 0), (1, 0)])
    with pytest.raises(ValueError):
        bulk_load([(1, 0), (1, 0)])


def test_bulk_load_is_denser_than_inserts():
    keys = random.Random(5).sample(range(10**6), 5000)
    a = BPlusTree(32)
    for k in keys:
        a.insert(k, 0)
    b = bulk_load(((k, 0) for k in sorted(keys)), 32, 1.0)
    assert b.stats()["leaves"] < a.stats()["leaves"]
    assert b.stats()["leaf_fill"] > a.stats()["leaf_fill"]


def test_chunk_sizes_respect_bounds():
    for n in range(5, 400):
        s = chunk_sizes(n, 6, 3, 8)
        assert sum(s) == n and all(3 <= x <= 8 for x in s)


def test_serialization_roundtrip():
    t = BPlusTree(20)
    rng = random.Random(6)
    for k in rng.sample(range(-5000, 5000), 800):
        t.insert(k, bytes([k % 256]) * (k % 5))
    blob = dumps(t, 1024)
    assert len(blob) % 1024 == 0
    u = loads(blob)
    u.check()
    assert list(u.items()) == list(t.items())
    assert [k for k, _ in u.reversed_items()] == [k for k, _ in t.reversed_items()]


def test_serialization_errors():
    t = BPlusTree(50)
    for i in range(50):
        t.insert(i, b"x" * 100)
    with pytest.raises(PageOverflow):
        dumps(t, 512)
    with pytest.raises(ValueError):
        loads(b"JUNK" + bytes(100))
    assert max_keys_for_page(4096, 16) == 4096 // 28 - 0 or max_keys_for_page(4096, 16) > 100
    m = max_keys_for_page(2048, 8)
    u = BPlusTree(m)
    for i in range(m * 5):
        u.insert(i, b"12345678")
    loads(dumps(u, 2048)).check()


def test_buffer_pool_lru():
    p = BufferPool(2)
    assert [p.access(x) for x in [1, 2, 1, 3, 2]] == [False, False, True, False, False]
    assert p.hits == 1 and p.misses == 4


def test_buffer_pool_hit_rate_improves_with_size():
    t = bulk_load(((i, i) for i in range(20000)), 16, 0.9)
    rng = random.Random(7)
    trace = record_trace(t, lambda tr: [tr.get(rng.randrange(20000)) for _ in range(2000)])
    rates = []
    for cap in (4, 64, 1024):
        pool = BufferPool(cap)
        pool.replay(trace)
        rates.append(pool.hit_rate)
    assert rates[0] < rates[1] < rates[2]
    assert t.trace is None


def test_trace_length_equals_height_for_point_lookup():
    t = bulk_load(((i, i) for i in range(5000)), 8)
    trace = record_trace(t, lambda tr: tr.get(1234))
    assert len(trace) == t.height()


def test_kvstore_batch_is_atomic():
    s = KVStore(4)
    s.put(1, b"a")
    with pytest.raises(TypeError):
        s.batch([("put", 2, b"b"), ("del", 1, None), ("put", 3, "not-bytes")])
    assert s.get(1) == b"a" and s.get(2) is None and s.get(3) is None
    s.tree.check()
    s.batch([("put", 2, b"b"), ("del", 1, None)])
    assert s.get(1) is None and s.get(2) == b"b"


def test_kvstore_recover_and_snapshot():
    s = KVStore(8)
    rng = random.Random(8)
    for _ in range(500):
        k = rng.randrange(100)
        if rng.random() < 0.7:
            s.put(k, bytes([k]))
        else:
            s.delete(k)
    r = KVStore.recover(s.wal, 8)
    assert list(r.scan()) == list(s.scan())
    q = KVStore.restore(s.snapshot())
    assert list(q.scan()) == list(s.scan())


def test_prefix_scan():
    s = KVStore(4)
    words = ["apple", "apply", "ape", "banana", "band", "bandana", "cat"]
    for w in words:
        s.put(encode_key(w), w.encode())
    assert [v.decode() for _, v in s.scan_prefix("app")] == ["apple", "apply"]
    assert [v.decode() for _, v in s.scan_prefix("ban")] == ["banana", "band", "bandana"]
    assert list(s.scan_prefix("zzz")) == []
