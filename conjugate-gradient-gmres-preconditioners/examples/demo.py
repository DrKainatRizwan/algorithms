"""Compare solvers and preconditioners on model PDE problems."""
import pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from krylov_solvers import (IC0, ILU0, SSOR, Jacobi, bicgstab, conjugate_gradient,
                            convection_diffusion_2d, gmres, poisson_2d)

rng = np.random.default_rng(0)
print("== Poisson 2D (SPD), CG iterations to 1e-8 ==")
print(f"{'n':>6} {'none':>6} {'jacobi':>7} {'ssor':>6} {'ic0':>5}")
for g in (16, 24, 32):
    a = poisson_2d(g); b = rng.standard_normal(g * g)
    its = [conjugate_gradient(a, b, m=m).iterations for m in (None, Jacobi(a), SSOR(a, 1.5), IC0(a))]
    print(f"{g*g:>6} " + " ".join(f"{i:>6}" for i in its))

print("\n== Convection-diffusion (nonsymmetric), n=484, Peclet=20 ==")
a = convection_diffusion_2d(22, 20.0); b = rng.standard_normal(484)
for name, f in [
    ("GMRES(20)", lambda: gmres(a, b, restart=20, maxiter=5000)),
    ("GMRES(20)+ILU0", lambda: gmres(a, b, m=ILU0(a), restart=20, maxiter=5000)),
    ("BiCGSTAB", lambda: bicgstab(a, b)),
    ("BiCGSTAB+ILU0", lambda: bicgstab(a, b, m=ILU0(a))),
]:
    t = time.perf_counter(); r = f(); dt = time.perf_counter() - t
    err = np.linalg.norm(a.matvec(r.x) - b) / np.linalg.norm(b)
    print(f"{name:<16} iters={r.iterations:>4} conv={r.converged} relres={err:.1e} time={dt:.2f}s")
