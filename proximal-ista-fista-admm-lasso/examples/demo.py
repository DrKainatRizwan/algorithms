"""Demo: compare ISTA, FISTA, coordinate descent and ADMM on a lasso problem."""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sparse_solvers import (make_sparse_regression, lambda_max, kkt_violation, ista, fista,
                            coordinate_descent, lasso_admm, lasso_path, cross_validate,
                            support_metrics, generalized_lasso_admm, difference_matrix)

X, y, w_true = make_sparse_regression(200, 100, 8, noise=0.2, corr=0.3, seed=0)
lam = 0.1 * lambda_max(X, y)
print(f"lasso n=200 p=100 k=8, lambda = {lam:.3f}")
print(f"{'solver':<14}{'iters':>7}{'time ms':>10}{'objective':>14}{'KKT viol':>12}{'nnz':>5}")
ref = None
for name, fn in [("ISTA", lambda: ista(X, y, lam, max_iter=20000, tol=1e-9)),
                 ("FISTA", lambda: fista(X, y, lam, restart=False, tol=1e-9)),
                 ("FISTA+restart", lambda: fista(X, y, lam, tol=1e-9)),
                 ("coord. desc.", lambda: coordinate_descent(X, y, lam, tol=1e-9)),
                 ("ADMM", lambda: lasso_admm(X, y, lam))]:
    t0 = time.perf_counter()
    r = fn()
    dt = (time.perf_counter() - t0) * 1e3
    print(f"{name:<14}{r.iterations:>7}{dt:>10.1f}{r.objective:>14.6f}"
          f"{kkt_violation(X, y, r.w, lam):>12.1e}{int((r.w != 0).sum()):>5}")

lams, coefs = lasso_path(X, y, 30)
best, errs = cross_validate(X, y, lams, folds=5)
w_hat = coordinate_descent(X, y, best).w
m = support_metrics(w_hat, w_true)
print(f"\nCV-selected lambda = {best:.3f}; precision {m['precision']:.2f}, "
      f"recall {m['recall']:.2f}, nnz {int((w_hat != 0).sum())}")

rng = np.random.default_rng(1)
truth = np.repeat([0.0, 3.0, -1.0, 2.0], 25)
obs = truth + 0.4 * rng.standard_normal(100)
fit = generalized_lasso_admm(np.eye(100), obs, 2.0, difference_matrix(100)).w
print(f"\nfused lasso denoising: RMSE noisy {np.sqrt(np.mean((obs-truth)**2)):.3f} "
      f"-> fitted {np.sqrt(np.mean((fit-truth)**2)):.3f}; jumps found "
      f"{int((np.abs(np.diff(fit)) > 1e-2).sum())} (true 3)")
