"""Compare evolution strategies on standard benchmarks."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from evolution_strategies import (SNES, XNES, CrossEntropyMethod, OnePlusOneES,
                                  cmaes_minimize, get_problem, ipop_cmaes)

TARGET = 1e-8


def run(method: str, problem: str, dim: int, budget: int, seed: int):
    p = get_problem(problem)
    x0 = np.full(dim, 2.0)
    if method == "CMA-ES":
        return cmaes_minimize(p.func, x0, 1.0, max_evals=budget, ftarget=TARGET, seed=seed)
    if method == "xNES":
        return XNES(x0, 1.0, seed=seed).minimize(p.func, budget, TARGET)
    if method == "SNES":
        return SNES(x0, 1.0, seed=seed).minimize(p.func, budget, TARGET)
    if method == "CEM":
        return CrossEntropyMethod(x0, 1.0, popsize=40, seed=seed).minimize(p.func, budget, TARGET)
    return OnePlusOneES(x0, 1.0, seed=seed).minimize(p.func, budget, TARGET)


def main() -> None:
    methods = ["CMA-ES", "xNES", "SNES", "CEM", "(1+1)-ES"]
    problems = [("sphere", 10), ("rotated_ellipsoid", 8), ("rosenbrock", 6), ("cigar", 10)]
    budget, seeds = 20000, range(3)
    print(f"Median evaluations to reach f<={TARGET:g} (3 seeds, budget {budget}); "
          "(successes/3)")
    print(f"{'problem':<22}" + "".join(f"{m:>16}" for m in methods))
    for name, dim in problems:
        row = f"{name + ' d=' + str(dim):<22}"
        for m in methods:
            rs = [run(m, name, dim, budget, s) for s in seeds]
            ok = [r.evals for r in rs if r.success]
            cell = f"{int(np.median(ok))} ({len(ok)}/3)" if ok else "- (0/3)"
            row += f"{cell:>16}"
        print(row)

    p = get_problem("rastrigin")
    print("\nRastrigin d=5: plain CMA-ES vs IPOP-CMA-ES (3 seeds)")
    plain = [cmaes_minimize(p.func, np.full(5, 3.0), 1.0, max_evals=120000, ftarget=1e-6,
                            seed=s).fun for s in range(3)]
    ipop = [ipop_cmaes(p.func, 5, (-5.12, 5.12), 2.0, max_evals=120000, ftarget=1e-6,
                       seed=s).fun for s in range(3)]
    print(f"  plain CMA-ES  median best f = {np.median(plain):.4f}")
    print(f"  IPOP-CMA-ES   median best f = {np.median(ipop):.4f}")


if __name__ == "__main__":
    main()
