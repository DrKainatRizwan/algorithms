from crdt_toolkit import AFTER, BEFORE, CONCURRENT, EQUAL, LamportClock, VectorClock


def test_lamport_observe_advances_past_remote():
    c = LamportClock()
    c.tick()
    assert c.observe(10) == 11
    assert c.observe(3) == 12


def test_vector_clock_ordering():
    a = VectorClock().incremented("x")
    b = a.incremented("y")
    assert a.compare(b) == BEFORE
    assert b.compare(a) == AFTER
    assert a.compare(a.merged(VectorClock())) == EQUAL


def test_vector_clock_concurrent_and_merge():
    a = VectorClock().incremented("x")
    b = VectorClock().incremented("y")
    assert a.compare(b) == CONCURRENT
    m = a.merged(b)
    assert m.dominates(a) and m.dominates(b)
    assert m.get("x") == 1 and m.get("y") == 1 and m.get("z") == 0


def test_vector_clock_immutable_and_hashable():
    a = VectorClock({"x": 2})
    b = a.incremented("x")
    assert a.get("x") == 2 and b.get("x") == 3
    assert hash(VectorClock({"x": 2, "y": 0})) == hash(a)
