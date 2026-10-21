"""Instance generators and brute-force oracles for tests and benchmarks."""

from __future__ import annotations

import itertools
import random

from .digraph import Digraph
from .two_sat import TwoSat


def random_digraph(n: int, m: int, seed: int = 0) -> Digraph:
    rng = random.Random(seed)
    g = Digraph(n)
    for _ in range(m):
        g.add_edge(rng.randrange(n), rng.randrange(n))
    return g


def random_2sat(n_vars: int, n_clauses: int, seed: int = 0) -> TwoSat:
    """Uniform random 2-CNF (threshold of satisfiability at ratio 1)."""
    rng = random.Random(seed)
    inst = TwoSat(n_vars)
    for _ in range(n_clauses):
        a = rng.randrange(1, n_vars + 1) * rng.choice((-1, 1))
        b = rng.randrange(1, n_vars + 1) * rng.choice((-1, 1))
        inst.add_clause(a, b)
    return inst


def planted_2sat(n_vars: int, n_clauses: int, seed: int = 0) -> tuple[TwoSat, dict[int, bool]]:
    """Random 2-CNF guaranteed satisfiable by a hidden assignment."""
    rng = random.Random(seed)
    hidden = {v: rng.random() < 0.5 for v in range(1, n_vars + 1)}
    inst = TwoSat(n_vars)
    while len(inst.clauses) < n_clauses:
        a = rng.randrange(1, n_vars + 1) * rng.choice((-1, 1))
        b = rng.randrange(1, n_vars + 1) * rng.choice((-1, 1))
        if (hidden[abs(a)] == (a > 0)) or (hidden[abs(b)] == (b > 0)):
            inst.add_clause(a, b)
    return inst, hidden


def brute_force_2sat(inst: TwoSat) -> list[dict[int, bool]]:
    """All models by exhaustive search (n_vars <= ~20)."""
    models = []
    for bits in itertools.product((False, True), repeat=inst.n_vars):
        a = {i + 1: b for i, b in enumerate(bits)}
        if inst.evaluate(a):
            models.append(a)
    return models
