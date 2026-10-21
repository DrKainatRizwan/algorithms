"""Three linear-time strongly-connected-component algorithms.

* Tarjan (1972): one DFS with low-link values.
* Kosaraju-Sharir (1978): two DFS passes, one on the transpose.
* Gabow's path-based algorithm (2000): two stacks, no low-links.

All are iterative.  Each returns ``comp`` where ``comp[v]`` is the component
id of ``v``.  Tarjan and Gabow number components in reverse topological order
of the condensation (sinks first); Kosaraju numbers them in topological order
(sources first).
"""

from __future__ import annotations

from .digraph import Digraph


def tarjan_scc(g: Digraph) -> list[int]:
    n = g.n
    index = [-1] * n
    low = [0] * n
    on_stack = [False] * n
    comp = [-1] * n
    stack: list[int] = []
    counter = 0
    ncomp = 0
    for root in range(n):
        if index[root] != -1:
            continue
        work = [(root, 0)]
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack[root] = True
        while work:
            u, i = work.pop()
            if i < len(g.adj[u]):
                work.append((u, i + 1))
                v = g.adj[u][i]
                if index[v] == -1:
                    index[v] = low[v] = counter
                    counter += 1
                    stack.append(v)
                    on_stack[v] = True
                    work.append((v, 0))
                elif on_stack[v]:
                    low[u] = min(low[u], index[v])
            else:
                if work:
                    p = work[-1][0]
                    low[p] = min(low[p], low[u])
                if low[u] == index[u]:
                    while True:
                        w = stack.pop()
                        on_stack[w] = False
                        comp[w] = ncomp
                        if w == u:
                            break
                    ncomp += 1
    return comp


def kosaraju_scc(g: Digraph) -> list[int]:
    n = g.n
    visited = [False] * n
    finish: list[int] = []
    for root in range(n):
        if visited[root]:
            continue
        visited[root] = True
        work = [(root, 0)]
        while work:
            u, i = work.pop()
            if i < len(g.adj[u]):
                work.append((u, i + 1))
                v = g.adj[u][i]
                if not visited[v]:
                    visited[v] = True
                    work.append((v, 0))
            else:
                finish.append(u)
    rev = g.reverse()
    comp = [-1] * n
    ncomp = 0
    for root in reversed(finish):
        if comp[root] != -1:
            continue
        comp[root] = ncomp
        work = [root]
        while work:
            u = work.pop()
            for v in rev.adj[u]:
                if comp[v] == -1:
                    comp[v] = ncomp
                    work.append(v)
        ncomp += 1
    return comp


def gabow_scc(g: Digraph) -> list[int]:
    n = g.n
    pre = [-1] * n
    comp = [-1] * n
    S: list[int] = []  # all vertices not yet assigned
    B: list[int] = []  # boundaries of tentative components (preorder numbers)
    counter = 0
    ncomp = 0
    for root in range(n):
        if pre[root] != -1:
            continue
        pre[root] = counter
        counter += 1
        S.append(root)
        B.append(pre[root])
        work = [(root, 0)]
        while work:
            u, i = work.pop()
            if i < len(g.adj[u]):
                work.append((u, i + 1))
                v = g.adj[u][i]
                if pre[v] == -1:
                    pre[v] = counter
                    counter += 1
                    S.append(v)
                    B.append(pre[v])
                    work.append((v, 0))
                elif comp[v] == -1:
                    while B and B[-1] > pre[v]:
                        B.pop()
            else:
                if B and B[-1] == pre[u]:
                    B.pop()
                    while True:
                        w = S.pop()
                        comp[w] = ncomp
                        if w == u:
                            break
                    ncomp += 1
    return comp


def components(comp: list[int]) -> list[list[int]]:
    """Group vertices by component id."""
    k = max(comp) + 1 if comp else 0
    groups: list[list[int]] = [[] for _ in range(k)]
    for v, c in enumerate(comp):
        groups[c].append(v)
    return groups


def same_partition(a: list[int], b: list[int]) -> bool:
    """True if two labelings induce the same partition (ids may differ)."""
    if len(a) != len(b):
        return False
    ab: dict[int, int] = {}
    ba: dict[int, int] = {}
    for x, y in zip(a, b):
        if ab.setdefault(x, y) != y or ba.setdefault(y, x) != x:
            return False
    return True


def condensation(g: Digraph, comp: list[int] | None = None) -> tuple[Digraph, list[int]]:
    """Return the (deduplicated) condensation DAG and the component labels."""
    if comp is None:
        comp = tarjan_scc(g)
    k = max(comp) + 1 if comp else 0
    dag = Digraph(k)
    seen: set[tuple[int, int]] = set()
    for u, v in g.edges():
        cu, cv = comp[u], comp[v]
        if cu != cv and (cu, cv) not in seen:
            seen.add((cu, cv))
            dag.add_edge(cu, cv)
    return dag, comp


def source_and_sink_components(g: Digraph) -> tuple[list[int], list[int]]:
    """Component ids with no incoming / no outgoing condensation edges."""
    dag, comp = condensation(g)
    indeg = [0] * dag.n
    for _, v in dag.edges():
        indeg[v] += 1
    sources = [c for c in range(dag.n) if indeg[c] == 0]
    sinks = [c for c in range(dag.n) if not dag.adj[c]]
    return sources, sinks


def min_edges_to_strongly_connect(g: Digraph) -> int:
    """Fewest edges to add so that ``g`` becomes strongly connected.

    Equals ``max(#source components, #sink components)`` when more than one
    component exists (Eswaran-Tarjan), else 0.
    """
    dag, _ = condensation(g)
    if dag.n <= 1:
        return 0
    srcs, snks = source_and_sink_components(g)
    return max(len(srcs), len(snks))
