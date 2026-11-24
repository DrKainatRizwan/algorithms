"""Branch-and-bound / branch-and-cut driver on top of the dense simplex."""

from __future__ import annotations

import heapq
import itertools
import time
from dataclasses import dataclass, field

import numpy as np

from .cuts import cover_cuts, gomory_cuts
from .lp import LPResult, solve_lp
from .model import MIP


@dataclass
class SolverOptions:
    branching: str = "most_fractional"   # first_fractional | most_fractional | pseudocost | strong
    selection: str = "best_first"        # best_first | depth_first
    gomory_rounds: int = 0
    cover_rounds: int = 0
    rounding_heuristic: bool = True
    node_limit: int = 50_000
    time_limit: float = 60.0
    int_tol: float = 1e-6
    strong_candidates: int = 6


@dataclass
class MIPResult:
    status: str                          # optimal | infeasible | unbounded | node_limit
    x: np.ndarray | None = None
    obj: float = np.inf
    bound: float = -np.inf
    nodes: int = 0
    lp_solves: int = 0
    cuts_added: int = 0
    root_bound_before: float = -np.inf
    root_bound_after: float = -np.inf
    incumbent_updates: int = 0
    seconds: float = 0.0
    stats: dict = field(default_factory=dict)

    @property
    def gap(self) -> float:
        if self.x is None:
            return np.inf
        return abs(self.obj - self.bound) / max(1.0, abs(self.obj))


class _Solver:
    def __init__(self, mip: MIP, opt: SolverOptions):
        self.mip, self.opt = mip, opt
        self.A_ub, self.b_ub = mip.A_ub.copy(), mip.b_ub.copy()
        self.best = np.inf
        self.best_x: np.ndarray | None = None
        self.lp_solves = 0
        self.updates = 0
        self.pc_sum = np.zeros((mip.n, 2))
        self.pc_cnt = np.zeros((mip.n, 2))
        # integral objective -> bounds can be rounded up
        c, integ = mip.c, mip.integer
        self.int_obj = bool(np.all(np.abs(c[~integ]) < 1e-12) and np.all(np.abs(c[integ] - np.round(c[integ])) < 1e-9)
                            and np.all(np.abs(mip.lb[integ] - np.round(mip.lb[integ])) < 1e-9))

    def lp(self, lb, ub, tableau=False) -> LPResult:
        self.lp_solves += 1
        m = self.mip
        return solve_lp(m.c, self.A_ub, self.b_ub, m.A_eq, m.b_eq, lb, ub, want_tableau=tableau)

    def fractional(self, x: np.ndarray) -> np.ndarray:
        idx = np.flatnonzero(self.mip.integer)
        d = np.abs(x[idx] - np.round(x[idx]))
        return idx[d > self.opt.int_tol]

    def tighten(self, v: float) -> float:
        return float(np.ceil(v - 1e-6)) if self.int_obj else v

    def try_incumbent(self, x: np.ndarray) -> None:
        m = self.mip
        xr = x.copy()
        xr[m.integer] = np.round(xr[m.integer])
        if m.is_feasible(xr, 1e-6) and m.objective(xr) < self.best - 1e-9:
            self.best, self.best_x = m.objective(xr), xr
            self.updates += 1

    def round_heuristic(self, x: np.ndarray) -> None:
        m = self.mip
        for fn in (np.round, np.floor, np.ceil):
            xr = x.copy()
            xr[m.integer] = fn(xr[m.integer])
            xr = np.clip(xr, m.lb, m.ub)
            if m.is_feasible(xr, 1e-6) and m.objective(xr) < self.best - 1e-9:
                self.best, self.best_x = m.objective(xr), xr
                self.updates += 1

    def choose(self, x, frac, lb, ub, obj) -> int:
        rule = self.opt.branching
        if rule == "first_fractional":
            return int(frac[0])
        if rule == "most_fractional":
            d = np.abs(x[frac] - np.floor(x[frac]) - 0.5)
            return int(frac[np.argmin(d)])
        if rule == "pseudocost":
            avg = np.where(self.pc_cnt > 0, self.pc_sum / np.maximum(self.pc_cnt, 1), np.nan)
            fill = np.nanmean(avg) if np.isfinite(avg).any() else 1.0
            avg = np.where(np.isnan(avg), fill, avg)
            f = x[frac] - np.floor(x[frac])
            score = np.maximum(avg[frac, 0] * f, 1e-6) * np.maximum(avg[frac, 1] * (1 - f), 1e-6)
            return int(frac[np.argmax(score)])
        if rule == "strong":
            order = frac[np.argsort(np.abs(x[frac] - np.floor(x[frac]) - 0.5))][: self.opt.strong_candidates]
            best_j, best_score = int(order[0]), -1.0
            for j in order:
                gains = []
                for side in (0, 1):
                    l2, u2 = lb.copy(), ub.copy()
                    if side == 0:
                        u2[j] = np.floor(x[j])
                    else:
                        l2[j] = np.ceil(x[j])
                    r = self.lp(l2, u2)
                    gains.append(1e6 if r.status != "optimal" else max(r.obj - obj, 1e-6))
                score = gains[0] * gains[1]
                if score > best_score:
                    best_j, best_score = int(j), score
            return best_j
        raise ValueError(f"unknown branching rule {rule!r}")

    def add_cuts(self, cuts) -> int:
        added = 0
        for a, b in cuts:
            self.A_ub = np.vstack([self.A_ub, a])
            self.b_ub = np.append(self.b_ub, b)
            added += 1
        return added

    def root(self):
        m = self.mip
        res = self.lp(m.lb.copy(), m.ub.copy(), tableau=self.opt.gomory_rounds > 0)
        first = res.obj if res.status == "optimal" else np.inf
        cuts = 0
        rounds = max(self.opt.gomory_rounds, self.opt.cover_rounds)
        for r in range(rounds):
            if res.status != "optimal" or not len(self.fractional(res.x)):
                break
            new = []
            if r < self.opt.cover_rounds:
                new += cover_cuts(m, res.x)
            if r < self.opt.gomory_rounds:
                new += gomory_cuts(m, res)
            if not new:
                break
            cuts += self.add_cuts(new)
            nxt = self.lp(m.lb.copy(), m.ub.copy(), tableau=self.opt.gomory_rounds > 0)
            if nxt.status != "optimal":
                res = nxt
                break
            res = nxt
        return res, first, cuts


def solve_mip(mip: MIP, options: SolverOptions | None = None) -> MIPResult:
    """Solve a MIP by branch-and-bound, optionally strengthened with root cutting planes."""
    opt = options or SolverOptions()
    t0 = time.perf_counter()
    s = _Solver(mip, opt)
    root, first, cuts = s.root()
    out = MIPResult("infeasible", cuts_added=cuts)
    if root.status == "unbounded":
        out.status = "unbounded"
        out.bound = -np.inf
        out.lp_solves = s.lp_solves
        return out
    if root.status != "optimal":
        out.lp_solves = s.lp_solves
        out.seconds = time.perf_counter() - t0
        return out
    out.root_bound_before, out.root_bound_after = s.tighten(first), s.tighten(root.obj)

    counter = itertools.count()
    heap: list = []

    def push(bound, depth, lb, ub, parent_x, parent_obj, var, side):
        key = bound if opt.selection == "best_first" else -depth
        heapq.heappush(heap, (key, next(counter), bound, depth, lb, ub, parent_x, parent_obj, var, side))

    push(s.tighten(root.obj), 0, mip.lb.copy(), mip.ub.copy(), None, 0.0, -1, 0)
    nodes = 0
    status = "optimal"
    global_bound = np.inf
    while heap:
        if nodes >= opt.node_limit or time.perf_counter() - t0 > opt.time_limit:
            status = "node_limit"
            break
        _, _, pbound, depth, lb, ub, parent_x, parent_obj, var, side = heapq.heappop(heap)
        if pbound >= s.best - 1e-9:
            continue
        nodes += 1
        res = root if depth == 0 else s.lp(lb, ub)
        if res.status != "optimal":
            continue
        obj = s.tighten(res.obj)
        if var >= 0 and parent_x is not None:
            d = abs(parent_x[var] - (np.floor(parent_x[var]) if side == 0 else np.ceil(parent_x[var])))
            if d > 1e-9:
                s.pc_sum[var, side] += max(res.obj - parent_obj, 0.0) / d
                s.pc_cnt[var, side] += 1
        if obj >= s.best - 1e-9:
            continue
        frac = s.fractional(res.x)
        if frac.size == 0:
            s.try_incumbent(res.x)
            continue
        if opt.rounding_heuristic:
            s.round_heuristic(res.x)
            if obj >= s.best - 1e-9:
                continue
        j = s.choose(res.x, frac, lb, ub, res.obj)
        v = res.x[j]
        lo_ub, hi_lb = ub.copy(), lb.copy()
        lo_ub[j] = np.floor(v)
        hi_lb[j] = np.ceil(v)
        push(obj, depth + 1, lb.copy(), lo_ub, res.x, res.obj, j, 0)
        push(obj, depth + 1, hi_lb, ub.copy(), res.x, res.obj, j, 1)

    out.nodes, out.lp_solves, out.incumbent_updates = nodes, s.lp_solves, s.updates
    out.seconds = time.perf_counter() - t0
    if s.best_x is not None:
        out.x, out.obj = s.best_x, s.best
    if status == "optimal":
        out.status = "optimal" if s.best_x is not None else "infeasible"
        out.bound = s.best if s.best_x is not None else np.inf
    else:
        out.status = "node_limit"
        out.bound = min([h[2] for h in heap] + [s.best])
    return out
