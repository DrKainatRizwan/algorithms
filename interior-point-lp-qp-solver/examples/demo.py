"""Demo: convergence, LP/QP scaling, barrier comparison and applications."""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from interior_point import (chebyshev_regression, l1_regression, min_variance_portfolio,
                            random_convex_qp, random_feasible_lp, solve, solve_barrier,
                            solve_lp, svm_dual)

print("== Textbook LP: max 3x+5y ==")
r = solve_lp([-3, -5], A_ub=[[1, 0], [0, 2], [3, 2]], b_ub=[4, 12, 18])
print(f"x = {np.round(r.x, 6)}, objective = {-r.objective:.6f}, iterations = {r.iterations}")

print("\n== Convergence on random LP (m=40, n=100) ==")
p = random_feasible_lp(40, 100, seed=0)
res = solve(p).raw
for h in res.history:
    print(f"iter {h['iter']:2d}  mu={h['mu']:.2e}  pres={h['pres']:.1e}  dres={h['dres']:.1e}  gap={h['gap']:.1e}")

print("\n== Scaling (iterations stay nearly constant) ==")
for (m, n) in [(10, 30), (50, 150), (100, 300), (200, 600)]:
    for kind, mk in (("LP", random_feasible_lp), ("QP", random_convex_qp)):
        q = mk(m, n, seed=1)
        t = time.perf_counter()
        s = solve(q)
        dt = time.perf_counter() - t
        print(f"{kind} m={m:3d} n={n:3d}: {s.iterations:2d} iters, {dt*1000:7.1f} ms, status={s.status}")

print("\n== Barrier vs primal-dual on a box-constrained LP ==")
rng = np.random.default_rng(4)
n = 20
G = np.vstack([rng.standard_normal((30, n)), np.eye(n), -np.eye(n)])
h = np.concatenate([rng.uniform(1, 2, 30), np.full(2 * n, 3.0)])
c = rng.standard_normal(n)
t = time.perf_counter(); b = solve_barrier(None, c, G, h); tb = time.perf_counter() - t
t = time.perf_counter(); d = solve_lp(c, A_ub=G, b_ub=h, lb=np.full(n, -np.inf)); td = time.perf_counter() - t
print(f"barrier: obj={b.objective:.8f}, newton steps={b.iterations}, {tb*1000:.1f} ms")
print(f"primal-dual: obj={d.objective:.8f}, iterations={d.iterations}, {td*1000:.1f} ms")

print("\n== Applications ==")
rng = np.random.default_rng(0)
x = np.linspace(0, 1, 60)
X = np.column_stack([np.ones(60), x])
y = 1 + 2 * x + 0.02 * rng.standard_normal(60)
y[::10] += 15
print("L1 fit :", np.round(l1_regression(X, y), 3), "(truth [1, 2])")
print("L2 fit :", np.round(np.linalg.lstsq(X, y, rcond=None)[0], 3))
w, rmax = chebyshev_regression(np.column_stack([np.ones(30), np.linspace(-1, 1, 30)]), np.linspace(-1, 1, 30) ** 2)
print(f"Chebyshev fit of x^2: w={np.round(w, 3)}, max error={rmax:.4f}")
F = rng.standard_normal((6, 6)); cov = F @ F.T / 6 + 0.05 * np.eye(6)
mu = rng.uniform(0.02, 0.12, 6)
wp = min_variance_portfolio(cov, mu, 0.08, 0.5)
print("portfolio:", np.round(wp, 4), f"return={mu @ wp:.4f} risk={np.sqrt(wp @ cov @ wp):.4f}")
Xs = np.vstack([rng.normal(-1.5, 0.8, (50, 2)), rng.normal(1.5, 0.8, (50, 2))])
ys = np.array([-1] * 50 + [1] * 50)
ws, bs = svm_dual(Xs, ys, C=1.0)
print(f"SVM: w={np.round(ws, 3)}, b={bs:.3f}, train accuracy={np.mean(np.sign(Xs @ ws + bs) == ys):.2%}")
