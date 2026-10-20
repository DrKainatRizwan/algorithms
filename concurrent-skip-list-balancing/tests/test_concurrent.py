import random
import threading

from skiplist_lab import AtomicMarkableRef, GeometricLevels, LockFreeSkipList
from skiplist_lab.benchmark import threaded_stress


def test_atomic_markable_ref_cas_semantics():
    a, b = object(), object()
    ref = AtomicMarkableRef(a, False)
    assert not ref.compare_and_set(b, a, False, False)
    assert not ref.compare_and_set(a, b, True, False)
    assert ref.compare_and_set(a, b, False, True)
    assert ref.get() == (b, True)
    assert ref.attempt_mark(b, False) and ref.get() == (b, False)


def test_sequential_semantics_match_set():
    lst, model = LockFreeSkipList(GeometricLevels(seed=1)), set()
    rng = random.Random(11)
    for _ in range(2000):
        k = rng.randrange(100)
        op = rng.random()
        if op < 0.4:
            assert lst.add(k) == (k not in model)
            model.add(k)
        elif op < 0.8:
            assert lst.remove(k) == (k in model)
            model.discard(k)
        else:
            assert lst.contains(k) == (k in model)
    assert lst.keys() == sorted(model)
    lst.check_invariants()


def test_values_and_duplicate_add():
    lst = LockFreeSkipList()
    assert lst.add(3, "x") and not lst.add(3, "y")
    assert lst.get(3) == "x" and lst.get(4, "none") == "none"
    lst.remove(3)
    assert lst.get(3) is None and len(lst) == 0


def test_disjoint_writers_keep_all_keys():
    lst = LockFreeSkipList(GeometricLevels(seed=2))

    def work(t):
        for k in range(t, 800, 4):
            assert lst.add(k)

    pool = [threading.Thread(target=work, args=(t,)) for t in range(4)]
    [t.start() for t in pool]
    [t.join() for t in pool]
    assert lst.keys() == list(range(800))
    lst.check_invariants()


def test_concurrent_removers_exactly_one_wins():
    lst = LockFreeSkipList(GeometricLevels(seed=3))
    for k in range(200):
        lst.add(k)
    wins = [0] * 4

    def work(t):
        for k in range(200):
            if lst.remove(k):
                wins[t] += 1

    pool = [threading.Thread(target=work, args=(t,)) for t in range(4)]
    [t.start() for t in pool]
    [t.join() for t in pool]
    assert sum(wins) == 200 and len(lst) == 0


def test_stress_final_state_matches_operation_log():
    res = threaded_stress(threads=4, ops=1200, key_space=60, seed=5)
    lst, net = res["list"], res["net"]
    assert all(v in (0, 1) for v in net)
    assert lst.keys() == [k for k, v in enumerate(net) if v == 1]
    lst.check_invariants()
    assert all(lst.contains(k) == (net[k] == 1) for k in range(60))
