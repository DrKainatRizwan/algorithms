"""Critical Path Method (activity-on-node) with floats and path enumeration."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from .graph import DiGraph, kahn_order

EPS = 1e-9


@dataclass(frozen=True)
class Activity:
    name: str
    duration: float
    predecessors: tuple[str, ...] = ()
    demands: Dict[str, int] = field(default_factory=dict, compare=False, hash=False)


@dataclass
class CPMResult:
    es: Dict[str, float]
    ef: Dict[str, float]
    ls: Dict[str, float]
    lf: Dict[str, float]
    total_float: Dict[str, float]
    free_float: Dict[str, float]
    duration: float
    order: List[str]

    @property
    def critical_activities(self) -> List[str]:
        return [a for a in self.order if self.total_float[a] <= EPS]


def build_graph(activities: Sequence[Activity]) -> DiGraph:
    names = {a.name for a in activities}
    if len(names) != len(activities):
        raise ValueError("duplicate activity names")
    g = DiGraph(nodes=[a.name for a in activities])
    for a in activities:
        if a.duration < 0:
            raise ValueError(f"negative duration for {a.name}")
        for p in a.predecessors:
            if p not in names:
                raise KeyError(f"unknown predecessor {p!r} of {a.name!r}")
            g.add_edge(p, a.name)
    return g


def critical_path_method(activities: Sequence[Activity]) -> CPMResult:
    """Forward/backward pass. Raises CycleError on cyclic dependencies."""
    g = build_graph(activities)
    dur = {a.name: float(a.duration) for a in activities}
    order = kahn_order(g)
    es: Dict[str, float] = {}
    ef: Dict[str, float] = {}
    for u in order:
        es[u] = max((ef[p] for p in g.pred[u]), default=0.0)
        ef[u] = es[u] + dur[u]
    total = max(ef.values(), default=0.0)
    ls: Dict[str, float] = {}
    lf: Dict[str, float] = {}
    for u in reversed(order):
        lf[u] = min((ls[s] for s in g.succ[u]), default=total)
        ls[u] = lf[u] - dur[u]
    tf = {u: ls[u] - es[u] for u in order}
    ff = {u: min((es[s] for s in g.succ[u]), default=total) - ef[u] for u in order}
    return CPMResult(es, ef, ls, lf, tf, ff, total, order)


def critical_paths(activities: Sequence[Activity], limit: int = 1000) -> List[List[str]]:
    """Enumerate all critical paths (start to finish through zero-float activities)."""
    res = critical_path_method(activities)
    g = build_graph(activities)
    crit = set(res.critical_activities)
    starts = [a for a in res.order if a in crit and abs(res.es[a]) <= EPS]
    paths: List[List[str]] = []

    def walk(u: str, path: List[str]) -> None:
        if len(paths) >= limit:
            return
        nxt = [v for v in g.succ[u] if v in crit and abs(res.es[v] - res.ef[u]) <= EPS]
        if not nxt:
            if abs(res.ef[u] - res.duration) <= EPS:
                paths.append(list(path))
            return
        for v in nxt:
            path.append(v)
            walk(v, path)
            path.pop()

    for s in starts:
        walk(s, [s])
    return paths


def crash_sensitivity(activities: Sequence[Activity], name: str, delta: float = 1.0) -> float:
    """Project-duration change when ``name`` is lengthened by ``delta`` (>=0)."""
    base = critical_path_method(activities).duration
    changed = [Activity(a.name, a.duration + (delta if a.name == name else 0.0), a.predecessors, a.demands)
               for a in activities]
    return critical_path_method(changed).duration - base
