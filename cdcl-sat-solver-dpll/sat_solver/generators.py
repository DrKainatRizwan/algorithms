"""Benchmark formula generators (random k-SAT, pigeonhole, colouring, queens)."""
from __future__ import annotations

import itertools
import random

from .cnf import CNF


def random_ksat(n: int, m: int, k: int = 3, seed: int = 0) -> CNF:
    """Uniform random k-SAT with ``m`` clauses over ``n`` variables."""
    rng = random.Random(seed)
    cnf = CNF(n)
    for _ in range(m):
        vs = rng.sample(range(1, n + 1), k)
        cnf.add_clause([v if rng.random() < 0.5 else -v for v in vs])
    return cnf


def planted_ksat(n: int, m: int, k: int = 3, seed: int = 0) -> tuple[CNF, dict[int, bool]]:
    """Random k-SAT guaranteed satisfiable by a hidden assignment."""
    rng = random.Random(seed)
    hidden = {v: rng.random() < 0.5 for v in range(1, n + 1)}
    cnf = CNF(n)
    while cnf.num_clauses < m:
        vs = rng.sample(range(1, n + 1), k)
        c = [v if rng.random() < 0.5 else -v for v in vs]
        if any(hidden[abs(l)] == (l > 0) for l in c):
            cnf.add_clause(c)
    return cnf, hidden


def pigeonhole(holes: int) -> CNF:
    """PHP: ``holes+1`` pigeons into ``holes`` holes (always UNSAT)."""
    pigeons = holes + 1
    var = lambda p, h: p * holes + h + 1
    cnf = CNF(pigeons * holes)
    for p in range(pigeons):
        cnf.add_clause([var(p, h) for h in range(holes)])
    for h in range(holes):
        for p1, p2 in itertools.combinations(range(pigeons), 2):
            cnf.add_clause([-var(p1, h), -var(p2, h)])
    return cnf


def graph_coloring(edges: list[tuple[int, int]], n_nodes: int, colors: int) -> CNF:
    """Proper ``colors``-colouring of a graph; var(v,c) = v*colors + c + 1."""
    var = lambda v, c: v * colors + c + 1
    cnf = CNF(n_nodes * colors)
    for v in range(n_nodes):
        cnf.add_clause([var(v, c) for c in range(colors)])
        for c1, c2 in itertools.combinations(range(colors), 2):
            cnf.add_clause([-var(v, c1), -var(v, c2)])
    for u, v in edges:
        for c in range(colors):
            cnf.add_clause([-var(u, c), -var(v, c)])
    return cnf


def random_graph(n: int, p: float, seed: int = 0) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    return [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]


def n_queens(n: int) -> CNF:
    """N-queens: var(r,c) = r*n + c + 1 means a queen on that square."""
    var = lambda r, c: r * n + c + 1
    cnf = CNF(n * n)
    for r in range(n):
        cnf.add_clause([var(r, c) for c in range(n)])
    cells = [(r, c) for r in range(n) for c in range(n)]
    for (r1, c1), (r2, c2) in itertools.combinations(cells, 2):
        if r1 == r2 or c1 == c2 or abs(r1 - r2) == abs(c1 - c2):
            cnf.add_clause([-var(r1, c1), -var(r2, c2)])
    return cnf


def parity_chain(n: int, parity: int) -> CNF:
    """XOR of ``n`` variables encoded in CNF via a chain of auxiliary vars.

    Combined with its negation (``parity_contradiction``) this gives small
    but resolution-hard UNSAT instances.
    """
    cnf = CNF(n)
    prev = 1
    nxt = n + 1
    for i in range(2, n + 1):
        out = nxt
        nxt += 1
        # out = prev XOR i
        cnf.add_clause([-out, prev, i])
        cnf.add_clause([-out, -prev, -i])
        cnf.add_clause([out, -prev, i])
        cnf.add_clause([out, prev, -i])
        prev = out
    cnf.add_clause([prev] if parity else [-prev])
    return cnf


def parity_contradiction(n: int) -> CNF:
    """Two parity chains over the same inputs demanding opposite parity."""
    a = parity_chain(n, 1)
    shift = a.num_vars
    cnf = a.copy()
    b = parity_chain(n, 0)
    remap = lambda l: (l if abs(l) <= n else (abs(l) + shift - n) * (1 if l > 0 else -1))
    for c in b.clauses:
        cnf.add_clause([remap(l) for l in c])
    return cnf
