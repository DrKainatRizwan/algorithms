import pytest

from graph_sat import TwoSat, random_2sat, planted_2sat, brute_force_2sat
from graph_sat.applications import schedule_two_slots, meeting_choices


def test_simple_sat_and_model_valid():
    s = TwoSat(3)
    s.add_clause(1, 2)
    s.add_clause(-1, 3)
    s.add_clause(-2, -3)
    r = s.solve()
    assert r.satisfiable and s.evaluate(r.assignment)


def test_simple_unsat_with_proof():
    s = TwoSat(2)
    s.add_clause(1, 2)
    s.add_clause(1, -2)
    s.add_clause(-1, 2)
    s.add_clause(-1, -2)
    r = s.solve()
    assert not r.satisfiable and r.assignment is None
    v = r.conflict_var
    fwd, back = r.proof
    assert fwd[0] == v and fwd[-1] == -v
    assert back[0] == -v and back[-1] == v


def test_matches_brute_force_random():
    for seed in range(60):
        n = 3 + seed % 7
        inst = random_2sat(n, int(n * (0.5 + (seed % 5) * 0.4)), seed)
        models = brute_force_2sat(inst)
        r = inst.solve()
        assert r.satisfiable == bool(models)
        if r.satisfiable:
            assert inst.evaluate(r.assignment)


def test_planted_always_sat_large():
    inst, _ = planted_2sat(3000, 9000, seed=1)
    r = inst.solve()
    assert r.satisfiable and inst.evaluate(r.assignment)


def test_unit_and_assumptions():
    s = TwoSat(3)
    s.add_implication(1, 2)
    s.add_implication(2, 3)
    s.add_unit(1)
    r = s.solve()
    assert r.assignment == {1: True, 2: True, 3: True}
    assert not s.solve([-3]).satisfiable


def test_backbone_matches_models():
    for seed in range(20):
        inst, _ = planted_2sat(6, 7, seed)
        models = brute_force_2sat(inst)
        expected = set()
        for v in range(1, 7):
            vals = {m[v] for m in models}
            if len(vals) == 1:
                expected.add(v if True in vals else -v)
        assert inst.backbone() == expected


def test_enumerate_counts_match():
    for seed in range(15):
        inst = random_2sat(6, 6, seed)
        got = list(inst.enumerate_solutions())
        assert len(got) == len(brute_force_2sat(inst))
        assert all(inst.evaluate(m) for m in got)


def test_enumerate_limit():
    inst = TwoSat(4)
    assert len(list(inst.enumerate_solutions(limit=5))) == 5


def test_equivalence_classes_and_xor():
    s = TwoSat(3)
    s.add_equiv(1, 2)
    s.add_xor(2, 3)
    classes = s.equivalence_classes()
    assert any({1, 2} <= set(c) for c in classes)
    assert any({-1, -2, 3} <= set(c) for c in classes)
    r = s.solve()
    assert r.assignment[1] == r.assignment[2] != r.assignment[3]


def test_at_most_one():
    s = TwoSat(4)
    s.add_at_most_one([1, 2, 3, 4])
    s.add_clause(2, 3)
    models = list(s.enumerate_solutions())
    assert all(sum(m.values()) == 1 for m in models) and len(models) == 2


def test_dimacs_roundtrip():
    inst = random_2sat(8, 12, 4)
    back = TwoSat.from_dimacs(inst.to_dimacs())
    assert back.clauses == inst.clauses and back.n_vars == 8
    with pytest.raises(ValueError):
        TwoSat.from_dimacs("p cnf 3 1\n1 2 3 0\n")


def test_invalid_literals():
    s = TwoSat(2)
    with pytest.raises(ValueError):
        s.add_clause(0, 1)
    with pytest.raises(ValueError):
        s.add_clause(1, 3)


def test_implied_literals():
    s = TwoSat(3)
    s.add_implication(1, 2)
    s.add_implication(2, 3)
    assert s.implied_literals(1) == {1, 2, 3}
    assert -1 in s.implied_literals(-3)


def test_scheduling_bipartite():
    even = [(i, (i + 1) % 6) for i in range(6)]
    slots = schedule_two_slots(6, even)
    assert slots is not None and all(slots[a] != slots[b] for a, b in even)
    assert schedule_two_slots(3, [(0, 1), (1, 2), (2, 0)]) is None
    assert schedule_two_slots(2, [(0, 1)], forced={0: True, 1: True}) is None


def test_meeting_choices():
    options = [(9, 10), (9, 11), (10, 11)]
    got = meeting_choices(options, [])
    assert got is not None and len(set(got)) == 3
    assert meeting_choices([(9, 9), (9, 9)], []) is None
