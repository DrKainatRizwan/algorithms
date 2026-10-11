"""Genetic algorithm with order crossover / edge recombination and 2-opt mutation."""

from __future__ import annotations

import numpy as np

from .annealing import Result
from .instances import Instance, tour_length
from .local_search import two_opt_delta


def order_crossover(p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """OX: copy a slice from p1, fill the rest in p2's order."""
    n = len(p1)
    a, b = sorted(rng.choice(n + 1, size=2, replace=False))
    child = -np.ones(n, dtype=int)
    child[a:b] = p1[a:b]
    taken = set(p1[a:b].tolist())
    fill = [c for c in np.roll(p2, -b) if c not in taken]
    positions = [(b + k) % n for k in range(n - (b - a))]
    for pos, c in zip(positions, fill):
        child[pos] = c
    return child


def edge_recombination(p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Edge recombination: preserve as many parental adjacencies as possible."""
    n = len(p1)
    adj: dict[int, set[int]] = {int(c): set() for c in p1}
    for p in (p1, p2):
        for k in range(n):
            adj[int(p[k])].update((int(p[k - 1]), int(p[(k + 1) % n])))
    cur = int(p1[0])
    child = [cur]
    remaining = set(int(c) for c in p1) - {cur}
    for nb in adj.values():
        nb.discard(cur)
    while remaining:
        nbrs = adj[cur]
        if nbrs:
            m = min(len(adj[x]) for x in nbrs)
            cands = sorted(x for x in nbrs if len(adj[x]) == m)
        else:
            cands = sorted(remaining)
        nxt = int(cands[int(rng.integers(len(cands)))])
        for nb in adj.values():
            nb.discard(nxt)
        remaining.discard(nxt)
        child.append(nxt)
        cur = nxt
    return np.array(child)


def _tournament(pop: np.ndarray, lens: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    idx = rng.choice(len(pop), size=k, replace=False)
    return pop[idx[np.argmin(lens[idx])]]


def genetic_algorithm(inst: Instance, pop_size: int = 60, generations: int = 200, seed: int = 0,
                      crossover: str = "ox", mutation_rate: float = 0.3, elite: int = 2,
                      tournament: int = 4, local_polish: bool = True) -> Result:
    """Generational GA with elitism.

    Mutation is a random segment reversal; with ``local_polish`` the best child of each
    generation also receives a greedy 2-opt sweep (a light memetic step).
    """
    rng = np.random.default_rng(seed)
    n = inst.n
    cx = {"ox": order_crossover, "erx": edge_recombination}[crossover]
    pop = np.array([rng.permutation(n) for _ in range(pop_size)])
    lens = np.array([tour_length(inst, t) for t in pop])
    evals = pop_size
    history = [float(lens.min())]
    for _ in range(generations):
        order = np.argsort(lens)
        pop, lens = pop[order], lens[order]
        nxt = [pop[k].copy() for k in range(elite)]
        while len(nxt) < pop_size:
            c = cx(_tournament(pop, lens, tournament, rng), _tournament(pop, lens, tournament, rng), rng)
            if n > 3 and rng.random() < mutation_rate:
                i = int(rng.integers(1, n - 1))
                j = int(rng.integers(i + 1, n))
                c[i:j + 1] = c[i:j + 1][::-1]
            nxt.append(c)
        pop = np.array(nxt)
        lens = np.array([tour_length(inst, t) for t in pop])
        evals += pop_size
        if local_polish and n > 3:
            b = int(np.argmin(lens))
            t = pop[b]
            for i in range(1, n - 1):
                for j in range(i + 1, n):
                    if two_opt_delta(inst, t, i, j) < -1e-12:
                        t[i:j + 1] = t[i:j + 1][::-1]
            lens[b] = tour_length(inst, t)
        history.append(float(lens.min()))
    b = int(np.argmin(lens))
    return Result(pop[b].copy(), float(lens[b]), history, evals)
