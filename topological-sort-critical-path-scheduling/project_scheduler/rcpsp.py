"""Resource-constrained project scheduling: serial and parallel schedule generation schemes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

from .cpm import Activity, CPMResult, build_graph, critical_path_method, EPS
from .graph import DiGraph


@dataclass(frozen=True)
class Resource:
    name: str
    capacity: int


def _priority(rule: str, acts: Dict[str, Activity], cpm: CPMResult, g: DiGraph) -> Callable[[str], tuple]:
    if rule == "lft":      # latest finish time first
        return lambda a: (cpm.lf[a], a)
    if rule == "lst":
        return lambda a: (cpm.ls[a], a)
    if rule == "mslk":     # minimum slack
        return lambda a: (cpm.total_float[a], a)
    if rule == "spt":
        return lambda a: (acts[a].duration, a)
    if rule == "grd":      # greatest resource demand
        return lambda a: (-sum(acts[a].demands.values()), a)
    if rule == "mts":      # most total successors
        def count(a: str) -> int:
            seen, st = set(), [a]
            while st:
                for s in g.succ[st.pop()]:
                    if s not in seen:
                        seen.add(s)
                        st.append(s)
            return len(seen)
        counts = {a: count(a) for a in acts}
        return lambda a: (-counts[a], a)
    raise ValueError(f"unknown priority rule {rule!r}")


def _check(acts: Sequence[Activity], resources: Sequence[Resource]) -> Dict[str, int]:
    cap = {r.name: r.capacity for r in resources}
    for a in acts:
        for r, d in a.demands.items():
            if r not in cap:
                raise KeyError(f"{a.name} uses unknown resource {r!r}")
            if d > cap[r]:
                raise ValueError(f"{a.name} demands {d} of {r} but capacity is {cap[r]}")
    return cap


class _Profile:
    """Piecewise-constant resource usage over time."""

    def __init__(self, cap: Dict[str, int]):
        self.cap = cap
        self.intervals: List[tuple[float, float, Dict[str, int]]] = []

    def usage_at(self, t: float, r: str) -> int:
        return sum(d.get(r, 0) for s, e, d in self.intervals if s <= t + EPS < e)

    def fits(self, start: float, dur: float, demands: Dict[str, int]) -> bool:
        if dur <= 0 or not demands:
            return True
        pts = [start] + [s for s, e, _ in self.intervals if start < s < start + dur - EPS]
        return all(self.usage_at(t, r) + d <= self.cap[r] for t in pts for r, d in demands.items())

    def add(self, start: float, dur: float, demands: Dict[str, int]) -> None:
        if dur > 0 and demands:
            self.intervals.append((start, start + dur, dict(demands)))

    def release_times(self, after: float) -> List[float]:
        return sorted({e for _, e, _ in self.intervals if e > after + EPS})


def schedule_serial(acts: Sequence[Activity], resources: Sequence[Resource], rule: str = "lft") -> Dict[str, float]:
    """Serial SGS: repeatedly pick the eligible activity with best priority and place it
    at the earliest precedence- and resource-feasible start. Returns start times."""
    cap = _check(acts, resources)
    by = {a.name: a for a in acts}
    g = build_graph(acts)
    cpm = critical_path_method(acts)
    key = _priority(rule, by, cpm, g)
    prof = _Profile(cap)
    start: Dict[str, float] = {}
    while len(start) < len(acts):
        elig = [a for a in by if a not in start and all(p in start for p in g.pred[a])]
        a = min(elig, key=key)
        t0 = max((start[p] + by[p].duration for p in g.pred[a]), default=0.0)
        cands = [t0] + [t for t in prof.release_times(t0)]
        for t in cands:
            if prof.fits(t, by[a].duration, by[a].demands):
                start[a] = t
                prof.add(t, by[a].duration, by[a].demands)
                break
    return start


def schedule_parallel(acts: Sequence[Activity], resources: Sequence[Resource], rule: str = "lft") -> Dict[str, float]:
    """Parallel SGS: time-incremental; at each decision time start every eligible
    activity (in priority order) that fits the remaining resources."""
    cap = _check(acts, resources)
    by = {a.name: a for a in acts}
    g = build_graph(acts)
    cpm = critical_path_method(acts)
    key = _priority(rule, by, cpm, g)
    prof = _Profile(cap)
    start: Dict[str, float] = {}
    t = 0.0
    while len(start) < len(acts):
        elig = sorted((a for a in by if a not in start and all(
            p in start and start[p] + by[p].duration <= t + EPS for p in g.pred[a])), key=key)
        for a in elig:
            if prof.fits(t, by[a].duration, by[a].demands):
                start[a] = t
                prof.add(t, by[a].duration, by[a].demands)
        finishes = [start[a] + by[a].duration for a in start if start[a] + by[a].duration > t + EPS]
        if len(start) < len(acts):
            t = min(finishes)
    return start


def makespan(acts: Sequence[Activity], start: Dict[str, float]) -> float:
    return max((start[a.name] + a.duration for a in acts), default=0.0)


def validate_schedule(acts: Sequence[Activity], resources: Sequence[Resource], start: Dict[str, float]) -> List[str]:
    """Return a list of violations (empty if the schedule is feasible)."""
    errs: List[str] = []
    by = {a.name: a for a in acts}
    for a in acts:
        if a.name not in start:
            errs.append(f"{a.name} unscheduled")
            continue
        for p in a.predecessors:
            if start[a.name] + EPS < start[p] + by[p].duration:
                errs.append(f"precedence {p}->{a.name} violated")
    events = sorted({start[a.name] for a in acts if a.name in start})
    for t in events:
        for r in resources:
            use = sum(a.demands.get(r.name, 0) for a in acts
                      if start[a.name] <= t + EPS < start[a.name] + a.duration)
            if use > r.capacity:
                errs.append(f"resource {r.name} over capacity at t={t}: {use}>{r.capacity}")
    return errs


def resource_lower_bound(acts: Sequence[Activity], resources: Sequence[Resource]) -> float:
    """max(critical path length, max_r total work_r / capacity_r)."""
    lb = critical_path_method(acts).duration
    for r in resources:
        work = sum(a.duration * a.demands.get(r.name, 0) for a in acts)
        lb = max(lb, work / r.capacity)
    return lb
