import random

import pytest

from crdt_toolkit import (GCounter, GSet, LWWMap, LWWRegister, MVRegister, ORSet, PNCounter,
                          TwoPSet, check_lattice_laws)
from crdt_toolkit.workload import (apply_random_ops, counter_op, lwwmap_op, orset_op,
                                   pncounter_op, twopset_op)


def test_gcounter_sums_across_replicas():
    a, b = GCounter("a"), GCounter("b")
    a.increment(3)
    b.increment(4)
    a.merge(b)
    b.merge(a)
    assert a.value() == b.value() == 7
    with pytest.raises(ValueError):
        a.increment(-1)


def test_pncounter_decrement():
    a, b = PNCounter("a"), PNCounter("b")
    a.increment(10)
    b.decrement(4)
    assert a.joined(b).value() == 6


def test_orset_add_wins_over_concurrent_remove():
    a, b = ORSet("a"), ORSet("b")
    a.add("x")
    b.merge(a)
    b.remove("x")      # b removes the observed add
    a.add("x")         # a concurrently re-adds
    a.merge(b)
    b.merge(a)
    assert "x" in a.value() and a == b


def test_orset_remove_then_readd_works_but_2pset_does_not():
    s, t = ORSet("a"), TwoPSet("a")
    for c in (s, t):
        c.add(1)
        c.remove(1)
        c.add(1)
    assert 1 in s.value()
    assert 1 not in t.value()


def test_lww_register_tiebreak_by_node():
    a, b = LWWRegister("a"), LWWRegister("b")
    a.set("A", 5)
    b.set("B", 5)
    assert a.joined(b).value() == b.joined(a).value() == "B"


def test_mv_register_keeps_concurrent_siblings_and_resolves():
    a, b = MVRegister("a"), MVRegister("b")
    a.assign(1)
    b.assign(2)
    a.merge(b)
    assert a.value() == frozenset({1, 2})
    a.assign(3)            # causally after both
    b.merge(a)
    assert b.value() == frozenset({3})


def test_lwwmap_delete_wins_when_newer():
    a, b = LWWMap("a"), LWWMap("b")
    a.put("k", 1, 1)
    b.merge(a)
    b.delete("k", 2)
    a.put("j", 9, 3)
    a.merge(b)
    assert a.value() == {"j": 9}


@pytest.mark.parametrize("cls,op", [(GCounter, counter_op), (PNCounter, pncounter_op),
                                    (ORSet, orset_op), (TwoPSet, twopset_op),
                                    (LWWMap, lwwmap_op)])
def test_lattice_laws_on_random_states(cls, op):
    rng = random.Random(7)
    reps = {n: cls(n) for n in "abc"}
    samples = []
    for _ in range(4):
        apply_random_ops(reps, op, 6, rng)
        samples.extend(r.copy() for r in reps.values())
    laws = check_lattice_laws(samples[:8])
    assert all(laws.values()), laws


def test_gset_union():
    a, b = GSet("a"), GSet("b")
    a.add(1)
    b.add(2)
    assert a.joined(b).value() == frozenset({1, 2})
