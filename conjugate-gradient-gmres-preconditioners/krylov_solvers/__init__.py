"""Krylov subspace solvers and preconditioners for sparse linear systems."""
from .sparse import CSRMatrix
from .preconditioners import (Identity, Jacobi, SSOR, IC0, ILU0, Preconditioner)
from .cg import conjugate_gradient, SolveResult
from .gmres import gmres
from .bicgstab import bicgstab
from .problems import poisson_2d, convection_diffusion_2d, random_spd

__all__ = ["CSRMatrix", "Identity", "Jacobi", "SSOR", "IC0", "ILU0", "Preconditioner",
           "conjugate_gradient", "SolveResult", "gmres", "bicgstab",
           "poisson_2d", "convection_diffusion_2d", "random_spd"]
