# CDCL SAT Solver with a DPLL Baseline and Checkable Proofs

## Overview
A from-scratch propositional SAT toolkit in pure Python (standard library only): a classic DPLL solver used as a baseline, a modern conflict-driven clause learning (CDCL) solver, DIMACS I/O, benchmark generators (random and planted k-SAT, pigeonhole, graph colouring, N-queens, parity contradictions) and a RUP proof checker that independently validates UNSAT answers.

## How it works
**DPLL** (`dpll.py`) repeatedly applies unit propagation and pure-literal elimination, then branches on a literal chosen by `first`, MOMS or two-sided Jeroslow-Wang scoring, backtracking chronologically.

**CDCL** (`cdcl.py`) adds the ingredients that make modern solvers practical:
- *Two watched literals*: each clause is watched through two of its literals, so assignments only touch clauses that might become unit.
- *First-UIP learning*: on a conflict the implication graph is resolved back to the first unique implication point, giving an asserting learned clause and a non-chronological backjump level. Local clause minimisation removes literals implied by the rest.
- *VSIDS* branching with exponentially growing bump increments (lazy heap), and *phase saving*.
- *Restarts*: Luby (default), geometric, or none.
- *Clause database reduction*: learned clauses are ranked by literal block distance (LBD) and size; the worse half is deleted, keeping glue clauses and clauses that are currently reasons.

**Proofs** (`proof.py`): every learned clause (and deletion) can be logged as a DRUP trace. `check_rup_proof` re-derives each lemma by negating it and unit-propagating over the formula plus earlier lemmas, and accepts when the empty clause is reached.

Modules: `cnf.py` (formula, DIMACS, evaluation, brute-force counting), `generators.py`, `dpll.py`, `cdcl.py`, `proof.py`.

## Complexity
- DPLL: worst case O(2^n * m) time, O(n*m) space in this copy-on-branch implementation.
- CDCL: still exponential in the worst case (pigeonhole requires exponential resolution proofs), but propagation costs amortised O(watch-list length) per assignment; space O(m + learned clauses), bounded in practice by database reduction.
- RUP checking: polynomial per lemma (naive propagation here, O(m) per pass).

## Usage
```python
from sat_solver import solve, DPLLSolver, check_rup_proof, generators as gen

cnf = gen.pigeonhole(5)
res = solve(cnf, seed=0, log_proof=True)
print(res.status, res.stats.conflicts)       # UNSAT 156
print(check_rup_proof(cnf, res.proof))       # True

cnf, _ = gen.planted_ksat(150, 600, 3, seed=3)
res = solve(cnf)
print(res.status, len(res.model))            # SAT 150
```
Run `python examples/demo.py` for the benchmarks and `python -m pytest -q` for the 72 tests (brute-force cross-checks, proof checking, restart policies, determinism).

## Results
Output of `examples/demo.py` (single run, pure Python, wall-clock on a CI machine):

| Instance | Vars | Clauses | Result | CDCL | DPLL (JW) |
|---|---|---|---|---|---|
| pigeonhole 5 | 30 | 81 | UNSAT | 8 ms, 156 conflicts | 8 ms, 238 decisions |
| pigeonhole 7 | 56 | 204 | UNSAT | 1190 ms, 6741 conflicts | 300 ms, 10078 decisions |
| parity contradiction 12 | 34 | 90 | UNSAT | 2 ms, 60 conflicts | 81 ms, 4094 decisions |
| 10-queens | 100 | 1480 | SAT | 6 ms, 34 conflicts | 63 ms, 222 decisions |
| random 3-SAT n=60, ratio 4.26 | 60 | 256 | UNSAT | 6 ms, 112 conflicts | 85 ms, 342 decisions |
| planted 3-SAT n=150 | 150 | 600 | SAT | 3 ms, 14 conflicts | not run |
| random 3-SAT n=200, ratio 4.26 | 200 | 852 | SAT | 2157 ms, 10485 conflicts | not run |

Observations: clause learning wins clearly on structured instances (parity, queens), while pigeonhole is hard for both. DPLL's cheaper per-node work wins on php-7 in pure Python. A 60-node random graph is not 4-colourable (1994 conflicts) but is 5-colourable (33 conflicts). A pigeonhole-6 refutation produced 2010 lemmas and 1643 deletions and passed the RUP check. On this small ablation, restarts did not help (Luby 7035 total conflicts vs 3969 with none).

## References
- Davis, Putnam, Logemann, Loveland. *A machine program for theorem proving*, CACM 1962.
- Marques-Silva, Sakallah. *GRASP: a search algorithm for propositional satisfiability*, IEEE Trans. Computers 1999.
- Moskewicz et al. *Chaff: Engineering an efficient SAT solver*, DAC 2001.
- Eén, Sörensson. *An extensible SAT-solver (MiniSat)*, SAT 2003.
- Audemard, Simon. *Predicting learnt clauses quality in modern SAT solvers (LBD)*, IJCAI 2009.
- Biere et al. (eds). *Handbook of Satisfiability*, IOS Press, 2nd ed. 2021.
- Wetzler, Heule, Hunt. *DRAT-trim: efficient checking and trimming using expressive clausal proofs*, SAT 2014.

---

**Author:** Dr. Kainat Rizwan
