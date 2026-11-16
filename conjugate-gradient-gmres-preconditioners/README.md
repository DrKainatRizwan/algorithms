# Krylov Solvers: Conjugate Gradient, GMRES and BiCGSTAB with Sparse Preconditioners

## Overview
From-scratch implementations of the main Krylov subspace methods for sparse linear systems `Ax = b`, built on a custom CSR matrix type (numpy only). Includes Jacobi, SSOR, incomplete Cholesky IC(0) and ILU(0) preconditioners, plus 2D Poisson and convection-diffusion test problems.

## How it works
- **CG / PCG**: for SPD `A`, minimises the A-norm of the error over the Krylov space `K_k(A, r0)` with short recurrences.
- **GMRES(m)**: builds an orthonormal Arnoldi basis (modified Gram-Schmidt), reduces the Hessenberg least-squares problem incrementally with Givens rotations, and restarts every `m` steps. Left preconditioned.
- **BiCGSTAB**: a transpose-free bi-Lanczos variant with a local residual-minimising stabilisation step; right preconditioned.
- **Preconditioners**: Jacobi (diagonal), SSOR (symmetric sweep), IC(0) and ILU(0) (factorisations restricted to the sparsity pattern of `A`).

## Complexity
| Method | Time per iteration | Memory |
|---|---|---|
| CG | O(nnz + n) | O(n) |
| GMRES(m) | O(nnz + m n) | O(m n) |
| BiCGSTAB | 2 matvecs, O(nnz + n) | O(n) |
| IC(0)/ILU(0) setup | O(nnz · row length) | O(nnz) |

CG on SPD systems needs O(√κ) iterations; good preconditioners reduce the effective condition number.

## Usage
```python
import numpy as np
from krylov_solvers import poisson_2d, conjugate_gradient, IC0

a = poisson_2d(32)
b = np.ones(32 * 32)
res = conjugate_gradient(a, b, m=IC0(a), tol=1e-8)
print(res.converged, res.iterations)
```

## Results
`python examples/demo.py` (iterations to relative residual 1e-8):

| Unknowns | None | Jacobi | SSOR(1.5) | IC(0) |
|---|---|---|---|---|
| 256 | 52 | 52 | 16 | 20 |
| 576 | 77 | 77 | 21 | 27 |
| 1024 | 101 | 101 | 27 | 35 |

Convection-diffusion, 484 unknowns, Péclet 20:

| Solver | Iterations |
|---|---|
| GMRES(20) | 96 |
| GMRES(20) + ILU(0) | 20 |
| BiCGSTAB | 44 |
| BiCGSTAB + ILU(0) | 12 |

Run tests with `python -m pytest -q` (20 tests).

## References
- Hestenes & Stiefel, "Methods of Conjugate Gradients for Solving Linear Systems", 1952.
- Saad & Schultz, "GMRES: A Generalized Minimal Residual Algorithm", SIAM J. Sci. Stat. Comput., 1986.
- van der Vorst, "Bi-CGSTAB", SIAM J. Sci. Stat. Comput., 1992.
- Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed., SIAM, 2003.
- Meijerink & van der Vorst, "An iterative solution method for linear systems of which the coefficient matrix is a symmetric M-matrix", 1977.

---

**Author:** Dr. Kainat Rizwan
