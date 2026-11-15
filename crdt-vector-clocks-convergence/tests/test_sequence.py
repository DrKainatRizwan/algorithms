import random

import pytest

from crdt_toolkit import RGA
from crdt_toolkit.workload import apply_random_ops, rga_op


def test_local_editing():
    d = RGA("a")
    d.insert_text(0, "hello")
    d.insert(5, "!")
    d.delete(0)
    assert d.text() == "ello!"
    with pytest.raises(IndexError):
        d.insert(99, "x")


def test_concurrent_inserts_at_same_position_converge():
    a, b = RGA("a"), RGA("b")
    a.insert_text(0, "ac")
    b.merge(a)
    a.insert(1, "X")
    b.insert(1, "Y")
    x, y = a.joined(b), b.joined(a)
    assert x.text() == y.text()
    assert sorted(x.text()) == sorted("aXYc") and x.text()[0] == "a" and x.text()[-1] == "c"


def test_delete_propagates_and_wins_over_nothing():
    a, b = RGA("a"), RGA("b")
    a.insert_text(0, "abc")
    b.merge(a)
    b.delete(1)
    a.insert(3, "d")
    a.merge(b)
    assert a.text() == "acd"


def test_random_editing_converges_after_full_sync():
    rng = random.Random(11)
    reps = {n: RGA(n) for n in ("a", "b", "c")}
    for _ in range(5):
        apply_random_ops(reps, rga_op, 15, rng)
        if rng.random() < 0.5:
            reps["a"].merge(reps["b"])
    for r in list(reps.values()):
        for s in list(reps.values()):
            r.merge(s)
    texts = {r.text() for r in reps.values()}
    assert len(texts) == 1
