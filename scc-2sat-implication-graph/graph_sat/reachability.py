"""Reachability on general digraphs via condensation and bitsets."""

from __future__ import annotations

from .digraph import Digraph
from .scc import condensation


def transitive_closure(g: Digraph) -> list[int]:
    """Reachability bitsets: bit ``v`` of result[u] is set iff u reaches v.

    Computed on the condensation DAG in reverse topological order with Python
    integers as bitsets: O(m * n / w).  Every vertex reaches itself.
    """
    dag, comp = condensation(g)
    order = dag.topological_order()
    assert order is not None
    members = [0] * dag.n
    for v, c in enumerate(comp):
        members[c] |= 1 << v
    reach = [0] * dag.n
    for c in reversed(order):
        r = members[c]
        for d in dag.adj[c]:
            r |= reach[d]
        reach[c] = r
    return [reach[comp[v]] for v in range(g.n)]


def transitive_reduction_dag(dag: Digraph) -> Digraph:
    """Minimal edge set with the same reachability, for a DAG."""
    order = dag.topological_order()
    if order is None:
        raise ValueError("graph is not a DAG")
    pos = {v: i for i, v in enumerate(order)}
    reach = [0] * dag.n
    out = Digraph(dag.n)
    for u in reversed(order):
        reach[u] = 1 << u
        # Visit successors nearest-first in topological order; a successor
        # already covered by an earlier one is redundant.
        for v in sorted(set(dag.adj[u]), key=pos.__getitem__):
            if not (reach[u] >> v) & 1:
                out.add_edge(u, v)
                reach[u] |= reach[v]
    return out


class ReachabilityIndex:
    """O(1) reachability queries after O(n*m/w) preprocessing."""

    def __init__(self, g: Digraph) -> None:
        self._closure = transitive_closure(g)
        self.n = g.n

    def reaches(self, u: int, v: int) -> bool:
        return bool((self._closure[u] >> v) & 1)

    def reachable_count(self, u: int) -> int:
        return bin(self._closure[u]).count("1")

    def mutually_reachable(self, u: int, v: int) -> bool:
        return self.reaches(u, v) and self.reaches(v, u)
