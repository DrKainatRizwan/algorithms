"""Demo: randomized SVD, sketching, CUR, preconditioned least squares and trace estimation."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from randomized_linalg import *


def timeit(f):
    t = time.perf_counter(); out = f(); return out, time.perf_counter() - t


rng = np.random.default_rng(42)
print("== Randomized SVD (1500x1000, polynomial spectrum, rank 20) ==")
A = decaying_spectrum(1500, 1000, "poly", rng, rate=0.5)
s_true = np.linalg.svd(A, compute_uv=False)
(Ue, se, Vte), t_full = timeit(lambda: np.linalg.svd(A, full_matrices=False))
print(f"optimal spectral error {s_true[20]:.3e}  (full SVD {t_full*1e3:.0f} ms)")
for q in (0, 1, 2, 4):
    (U, s, Vt), t = timeit(lambda: randomized_svd(A, 20, 10, q, np.random.default_rng(1)))
    print(f"  q={q}: error {np.linalg.norm(A-(U*s)@Vt,2):.3e}  time {t*1e3:.0f} ms")
U, s, Vt = single_pass_svd(A, 20, np.random.default_rng(1))
print(f"  single-pass: error {np.linalg.norm(A-(U*s)@Vt,2):.3e}")
B = low_rank_plus_noise(600, 400, 12, 1e-3, rng)
Q = adaptive_range_finder(B, 5e-2, 4, rng=rng)
print(f"adaptive range finder found rank {Q.shape[1]} for tol 5e-2 (true rank 12)")

print("\n== Sketch distortion of a 10-dim subspace in R^4096 (m=400) ==")
Qs = np.linalg.qr(rng.standard_normal((4096, 10)))[0]
for kind in ("gaussian", "srht", "countsketch", "sparse_sign"):
    S, t = timeit(lambda: make_sketch(kind, 400, 4096, np.random.default_rng(3)).apply(Qs))
    sv = np.linalg.svd(S, compute_uv=False)
    print(f"  {kind:12s} singular values in [{sv.min():.3f}, {sv.max():.3f}]  apply {t*1e3:.1f} ms")
print(f"JL dimension for 1000 points, eps=0.2: {jl_min_dim(1000, 0.2)}")

print("\n== CUR on noisy rank-8 matrix (400x300), 8 rows/cols ==")
M = low_rank_plus_noise(400, 300, 8, 0.05, rng)
best = np.linalg.svd(M, compute_uv=False)
print(f"best rank-8 Frobenius error {np.sqrt(np.sum(best[8:]**2)):.3f}")
for m in ("uniform", "leverage", "pivot", "deim"):
    C, U_, R, *_ = cur_decomposition(M, 8, m, c=12, r=12, rng=np.random.default_rng(5))
    print(f"  {m:9s} error {np.linalg.norm(M-C@U_@R):.3f}")

print("\n== Least squares 20000x30, cond=1e8 ==")
A2, b2, _ = ill_conditioned_ls(20000, 30, 1e8, rng)
xo = np.linalg.lstsq(A2, b2, rcond=None)[0]
ro = np.linalg.norm(A2 @ xo - b2)
xs = sketch_and_solve(A2, b2, 300, "countsketch", rng)
print(f"  sketch-and-solve residual ratio {np.linalg.norm(A2@xs-b2)/ro:.4f}")
(xp, it, cond), t = timeit(lambda: sketch_precondition_cgls(A2, b2, "srht", 4.0, 1e-10, rng))
print(f"  preconditioned CGLS: {it} iters, cond(AR^-1)={cond:.2f}, residual ratio {np.linalg.norm(A2@xp-b2)/ro:.6f}")
_, it0 = cgls(lambda v: A2 @ v, lambda u: A2.T @ u, b2, 30, 1e-10, 5000)
print(f"  plain CGLS iterations: {it0}")

print("\n== Trace estimation (PSD, eigenvalues i^-2, n=500, 30 queries, 100 trials) ==")
Qm = np.linalg.qr(rng.standard_normal((500, 500)))[0]
K = (Qm * (np.arange(1, 501.0) ** -2)) @ Qm.T
tr = np.trace(K)
for name, f in (("hutchinson", hutchinson), ("hutch++", hutchpp)):
    est = [f(lambda X: K @ X, 500, 30, np.random.default_rng(i)) for i in range(100)]
    print(f"  {name:10s} mean rel. error {np.mean(np.abs(np.array(est)-tr))/tr:.2e}")
