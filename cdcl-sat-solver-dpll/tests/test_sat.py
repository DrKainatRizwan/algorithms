import random

import pytest

from sat_solver import (CNF, CDCLSolver, DPLLSolver, check_rup_proof,
                        count_models, evaluate, generators as gen, luby, solve)


def test_luby_sequence():
    assert [luby(i) for i in range(1, 16)] == [1, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 4, 8]


def test_dimacs_roundtrip():
    cnf = gen.random_ksat(8, 20, 3, seed=1)
    again = CNF.from_dimacs(cnf.to_dimacs())
    assert again.clauses == cnf.clauses and again.num_vars == cnf.num_vars


def test_dimacs_parse_comments_and_multiline():
    cnf = CNF.from_dimacs("c hi\np cnf 3 2\n1 -2\n0 2 3 0\n")
    assert cnf.clauses == [(1, -2), (2, 3)]


def test_dimacs_errors():
    with pytest.raises(ValueError):
        CNF.from_dimacs("p cnf 2 2\n1 0\n")
    with pytest.raises(ValueError):
        CNF.from_dimacs("p cnf 2 1\n1 2\n")


def test_trivial_cases():
    assert solve(CNF(0)).status == "SAT"
    c = CNF(1); c.add_clause([1]); c.add_clause([-1])
    assert solve(c).status == "UNSAT"
    assert DPLLSolver(c).solve().status == "UNSAT"
    e = CNF(2); e.add_clause([])
    assert solve(e).status == "UNSAT"
    assert DPLLSolver(e).solve().status == "UNSAT"


def test_tautology_and_duplicates_ignored():
    c = CNF(2); c.add_clause([1, -1]); c.add_clause([2, 2])
    r = solve(c)
    assert r.status == "SAT" and r.model[2] is True


@pytest.mark.parametrize("seed", range(40))
def test_cdcl_and_dpll_agree_with_brute_force(seed):
    n = 10
    m = random.Random(seed).randint(25, 55)
    cnf = gen.random_ksat(n, m, 3, seed)
    truth = count_models(cnf) > 0
    r = solve(cnf, seed=seed)
    d = DPLLSolver(cnf).solve()
    assert (r.status == "SAT") == truth == (d.status == "SAT")
    if truth:
        assert evaluate(cnf, r.model) and evaluate(cnf, d.model)


@pytest.mark.parametrize("h", ["first", "moms", "jw"])
def test_dpll_heuristics_correct(h):
    cnf, _ = gen.planted_ksat(25, 90, 3, seed=3)
    r = DPLLSolver(cnf, heuristic=h).solve()
    assert r.status == "SAT" and evaluate(cnf, r.model)
    assert DPLLSolver(gen.pigeonhole(4), heuristic=h).solve().status == "UNSAT"


def test_dpll_bad_heuristic():
    with pytest.raises(ValueError):
        DPLLSolver(CNF(1), heuristic="nope")


def test_planted_instances_solved_with_valid_model():
    for seed in range(5):
        cnf, hidden = gen.planted_ksat(60, 240, 3, seed)
        assert evaluate(cnf, hidden)
        r = solve(cnf, seed=seed)
        assert r.status == "SAT" and evaluate(cnf, r.model)


@pytest.mark.parametrize("holes", [2, 3, 4, 5, 6])
def test_pigeonhole_unsat(holes):
    assert solve(gen.pigeonhole(holes)).status == "UNSAT"


def test_n_queens_sat_and_valid():
    cnf = gen.n_queens(8)
    r = solve(cnf)
    assert r.status == "SAT"
    placed = [v for v, b in r.model.items() if b]
    rows = {(v - 1) // 8 for v in placed}
    assert rows == set(range(8))
    assert len(placed) >= 8
    assert evaluate(cnf, r.model)


def test_n_queens_3_unsat():
    assert solve(gen.n_queens(3)).status == "UNSAT"


def test_graph_coloring():
    # K4 is 4-colourable, not 3-colourable
    edges = [(a, b) for a in range(4) for b in range(a + 1, 4)]
    assert solve(gen.graph_coloring(edges, 4, 3)).status == "UNSAT"
    r = solve(gen.graph_coloring(edges, 4, 4))
    assert r.status == "SAT"


def test_rup_proofs_verify_for_unsat():
    for cnf in (gen.pigeonhole(4), gen.parity_contradiction(6), gen.n_queens(3)):
        r = solve(cnf, log_proof=True)
        assert r.status == "UNSAT"
        assert check_rup_proof(cnf, r.proof)


def test_rup_rejects_bogus_proof():
    cnf = gen.random_ksat(12, 30, 3, seed=2)
    assert not check_rup_proof(cnf, [("a", (1,)), ("a", ())])
    assert not check_rup_proof(gen.pigeonhole(3), [])


def test_proof_with_clause_deletion_still_checks():
    cnf = gen.pigeonhole(6)
    r = solve(cnf, log_proof=True, reduce_first=20)
    assert r.status == "UNSAT" and r.stats.deleted > 0
    assert any(k == "d" for k, _ in r.proof)
    assert check_rup_proof(cnf, r.proof)


@pytest.mark.parametrize("policy", ["luby", "geometric", "none"])
def test_restart_policies_all_correct(policy):
    cnf, _ = gen.planted_ksat(50, 200, 3, seed=7)
    r = solve(cnf, restart=policy, restart_base=5)
    assert r.status == "SAT" and evaluate(cnf, r.model)
    assert solve(gen.pigeonhole(5), restart=policy, restart_base=5).status == "UNSAT"


def test_restarts_happen():
    r = solve(gen.pigeonhole(6), restart_base=4)
    assert r.stats.restarts > 0


def test_bad_restart_policy():
    with pytest.raises(ValueError):
        CDCLSolver(CNF(1), restart="x")


def test_determinism():
    cnf = gen.random_ksat(40, 170, 3, seed=11)
    a = solve(cnf, seed=5)
    b = solve(cnf, seed=5)
    assert a.status == b.status and a.model == b.model
    assert a.stats == b.stats


def test_conflict_budget_gives_unknown():
    r = solve(gen.pigeonhole(8), max_conflicts=10)
    assert r.status == "UNKNOWN" and r.stats.conflicts == 10


def test_minimisation_and_toggle_equivalent_answers():
    for seed in range(6):
        cnf = gen.random_ksat(30, 135, 3, seed)
        a = solve(cnf, minimise=True).status
        b = solve(cnf, minimise=False, phase_saving=False).status
        assert a == b


def test_cdcl_beats_dpll_on_pigeonhole_conflicts_not_worse():
    cnf = gen.pigeonhole(6)
    c = solve(cnf)
    d = DPLLSolver(cnf, heuristic="jw").solve()
    assert c.status == d.status == "UNSAT"
    assert c.stats.decisions < d.decisions


def test_parity_contradiction_unsat_and_parity_chain_sat():
    assert solve(gen.parity_contradiction(8)).status == "UNSAT"
    cnf = gen.parity_chain(8, 1)
    r = solve(cnf)
    assert r.status == "SAT"
    assert sum(r.model[v] for v in range(1, 9)) % 2 == 1
