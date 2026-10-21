"""Problems that reduce to 2-SAT or SCC computations."""

from __future__ import annotations

from typing import Sequence

from .digraph import Digraph
from .scc import tarjan_scc, condensation
from .two_sat import TwoSat


def schedule_two_slots(
    n_jobs: int, conflicts: Sequence[tuple[int, int]], forced: dict[int, bool] | None = None
) -> list[bool] | None:
    """Assign each job to slot A (True) or B (False) so no conflicting pair
    shares a slot, i.e. 2-colouring expressed as XOR clauses.  ``forced``
    pins some jobs.  Returns ``None`` if impossible."""
    s = TwoSat(n_jobs)
    for a, b in conflicts:
        s.add_xor(a + 1, b + 1)
    for j, val in (forced or {}).items():
        s.add_unit((j + 1) if val else -(j + 1))
    res = s.solve()
    if not res.satisfiable:
        return None
    return [res.assignment[j + 1] for j in range(n_jobs)]


def meeting_choices(options: Sequence[tuple[int, int]], clashes: Sequence[tuple[tuple[int, int], tuple[int, int]]]):
    """Each meeting ``i`` has two candidate time slots ``options[i]``.

    ``clashes`` lists pairs ``((i, ci), (j, cj))`` of (meeting, choice) that
    cannot both be selected.  Variable ``i+1`` true selects choice 0.
    Returns the chosen slot per meeting or ``None``.
    """
    s = TwoSat(len(options))

    def lit(i: int, c: int) -> int:
        return (i + 1) if c == 0 else -(i + 1)

    for (i, ci), (j, cj) in clashes:
        s.add_clause(-lit(i, ci), -lit(j, cj))
    for i in range(len(options)):
        for j in range(i + 1, len(options)):
            for ci in (0, 1):
                for cj in (0, 1):
                    if options[i][ci] == options[j][cj]:
                        s.add_clause(-lit(i, ci), -lit(j, cj))
    res = s.solve()
    if not res.satisfiable:
        return None
    return [options[i][0 if res.assignment[i + 1] else 1] for i in range(len(options))]


def orient_edges_for_strong_connectivity(n: int, edges: Sequence[tuple[int, int]]) -> list[tuple[int, int]] | None:
    """Robbins' theorem: orient a connected undirected graph strongly iff it
    is 2-edge-connected.  Uses a DFS tree and orients tree edges downward,
    back edges upward.  Returns ``None`` if a bridge exists or graph is
    disconnected."""
    adj: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for idx, (u, v) in enumerate(edges):
        adj[u].append((v, idx))
        adj[v].append((u, idx))
    disc = [-1] * n
    low = [0] * n
    used = [False] * len(edges)
    oriented: list[tuple[int, int]] = [(-1, -1)] * len(edges)
    t = 0
    disc[0] = low[0] = t
    t += 1
    work = [(0, -1, 0)]
    while work:
        u, pe, i = work.pop()
        if i < len(adj[u]):
            work.append((u, pe, i + 1))
            v, idx = adj[u][i]
            if idx == pe or used[idx]:
                continue
            used[idx] = True
            if disc[v] == -1:
                oriented[idx] = (u, v)
                disc[v] = low[v] = t
                t += 1
                work.append((v, idx, 0))
            else:
                oriented[idx] = (u, v)
                low[u] = min(low[u], disc[v])
        else:
            if work:
                p = work[-1][0]
                if low[u] > disc[p]:
                    return None  # bridge
                low[p] = min(low[p], low[u])
    if any(d == -1 for d in disc):
        return None
    return oriented


def is_strongly_connected(g: Digraph) -> bool:
    return g.n == 0 or max(tarjan_scc(g)) == 0


def kernel_graph_summary(g: Digraph) -> dict[str, int]:
    """Counts describing the SCC structure of ``g``."""
    dag, comp = condensation(g)
    sizes = [0] * dag.n
    for c in comp:
        sizes[c] += 1
    return {
        "vertices": g.n,
        "edges": g.m,
        "components": dag.n,
        "largest": max(sizes, default=0),
        "singletons": sum(1 for s in sizes if s == 1),
        "dag_edges": dag.m,
    }
