"""Run all optimisers on a problem and tabulate cost."""
from __future__ import annotations

import time
from typing import Callable, Dict, List

from .functions import Problem
from .nonlinear_cg import nonlinear_cg
from .quasi_newton import bfgs, dfp, lbfgs, sr1
from .trust_region import trust_region


def compare(make_problem: Callable[[], Problem], tol: float = 1e-6) -> List[Dict]:
    """Solve a fresh copy of the problem with each method; return rows of statistics."""
    methods = {
        "bfgs": lambda p: bfgs(p, tol=tol),
        "dfp": lambda p: dfp(p, tol=tol),
        "sr1": lambda p: sr1(p, tol=tol),
        "lbfgs(m=5)": lambda p: lbfgs(p, m=5, tol=tol),
        "cg-pr+": lambda p: nonlinear_cg(p, kind="pr+", tol=tol),
        "cg-fr": lambda p: nonlinear_cg(p, kind="fr", tol=tol),
        "tr-steihaug": lambda p: trust_region(p, subproblem="steihaug", tol=tol),
        "tr-dogleg": lambda p: trust_region(p, subproblem="dogleg", tol=tol),
    }
    rows = []
    for name, run in methods.items():
        p = make_problem()
        t0 = time.perf_counter()
        r = run(p)
        rows.append({"method": name, "converged": r.converged, "iters": r.iterations,
                     "f_evals": p.n_f, "g_evals": p.n_g, "h_evals": p.n_h,
                     "f": r.f, "grad": r.grad_norm, "time_ms": 1000 * (time.perf_counter() - t0)})
    return rows
