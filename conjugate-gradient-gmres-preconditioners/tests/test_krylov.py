import numpy as np
import pytest

from krylov_solvers import (CSRMatrix, IC0, ILU0, SSOR, Jacobi, Identity, bicgstab,
                            conjugate_gradient, convection_diffusion_2d, gmres,
                            poisson_2d, random_spd)


def rhs(n, seed=0):
    return np.random.default_rng(seed).standard_normal(n)


def test_csr_matvec_matches_dense():
    a = random_spd(30, 0.2, seed=1)
    x = rhs(30)
    assert np.allclose(a.matvec(x), a.to_dense() @ x)


def test_duplicates_summed_and_transpose():
    a = CSRMatrix.from_triplets([0, 0, 1], [1, 1, 0], [1.0, 2.0, 5.0], (2, 2))
    assert a.to_dense().tolist() == [[0, 3], [5, 0]]
    assert a.transpose().to_dense().tolist() == [[0, 5], [3, 0]]


def test_diagonal_and_symmetry():
    a = poisson_2d(4)
    assert np.all(a.diagonal() == 4) and a.is_symmetric()
    assert not convection_diffusion_2d(4).is_symmetric()


def test_cg_solves_poisson():
    a = poisson_2d(10); b = rhs(100)
    r = conjugate_gradient(a, b, tol=1e-10)
    assert r.converged and np.linalg.norm(a.matvec(r.x) - b) < 1e-8 * np.linalg.norm(b)


def test_cg_finite_termination_small_spectrum():
    d = np.diag([1.0, 1.0, 2.0, 2.0, 5.0, 5.0])
    r = conjugate_gradient(CSRMatrix.from_dense(d), rhs(6), tol=1e-12)
    assert r.converged and r.iterations <= 3


@pytest.mark.parametrize("cls", [Jacobi, IC0, ILU0, SSOR])
def test_preconditioners_reduce_cg_iterations(cls):
    a = poisson_2d(16); b = rhs(256)
    base = conjugate_gradient(a, b).iterations
    pre = conjugate_gradient(a, b, m=cls(a)).iterations
    assert pre <= base
    if cls in (IC0, ILU0):
        assert pre < 0.6 * base


def test_ic0_exact_on_tridiagonal():
    n = 12
    d = 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
    a = CSRMatrix.from_dense(d)
    ic = IC0(a)
    assert np.allclose(ic.L.to_dense() @ ic.L.to_dense().T, d)
    b = rhs(n)
    assert np.allclose(ic.apply(b), np.linalg.solve(d, b))


def test_ilu0_exact_on_tridiagonal():
    n = 10
    d = 3 * np.eye(n) - np.eye(n, k=1) - 2 * np.eye(n, k=-1)
    a = CSRMatrix.from_dense(d)
    b = rhs(n)
    assert np.allclose(ILU0(a).apply(b), np.linalg.solve(d, b))


def test_ic0_breakdown_on_indefinite():
    a = CSRMatrix.from_dense(np.array([[1.0, 2.0], [2.0, 1.0]]))
    with pytest.raises(np.linalg.LinAlgError):
        IC0(a)


def test_ssor_is_symmetric_operator():
    a = poisson_2d(5)
    m = SSOR(a, 1.3)
    M = np.column_stack([m.apply(e) for e in np.eye(25)])
    assert np.allclose(M, M.T)


def test_gmres_matches_direct_solve():
    a = convection_diffusion_2d(8, 20.0); b = rhs(64)
    r = gmres(a, b, restart=64, tol=1e-12)
    assert r.converged
    assert np.allclose(r.x, np.linalg.solve(a.to_dense(), b), atol=1e-8)


def test_gmres_full_converges_within_n_iterations():
    a = convection_diffusion_2d(5, 5.0); b = rhs(25)
    r = gmres(a, b, restart=25, tol=1e-10)
    assert r.converged and r.iterations <= 25


def test_gmres_residual_monotone_within_cycle():
    a = convection_diffusion_2d(8, 10.0); b = rhs(64)
    r = gmres(a, b, restart=64, tol=1e-10)
    assert all(y <= x + 1e-12 for x, y in zip(r.residuals, r.residuals[1:]))


def test_gmres_ilu_beats_unpreconditioned():
    a = convection_diffusion_2d(14, 10.0); b = rhs(196)
    plain = gmres(a, b, restart=20, tol=1e-8, maxiter=5000)
    pre = gmres(a, b, m=ILU0(a), restart=20, tol=1e-8, maxiter=5000)
    assert pre.converged and pre.iterations < plain.iterations


def test_bicgstab_solves_nonsymmetric():
    a = convection_diffusion_2d(12, 10.0); b = rhs(144)
    r = bicgstab(a, b, m=ILU0(a), tol=1e-10)
    assert r.converged and np.linalg.norm(a.matvec(r.x) - b) < 1e-8 * np.linalg.norm(b)


def test_zero_rhs_and_warm_start():
    a = poisson_2d(4)
    assert conjugate_gradient(a, np.zeros(16)).iterations == 0
    b = rhs(16); xs = np.linalg.solve(a.to_dense(), b)
    assert conjugate_gradient(a, b, x0=xs, tol=1e-8).iterations == 0


def test_maxiter_reports_nonconvergence():
    a = poisson_2d(10)
    assert not conjugate_gradient(a, rhs(100), maxiter=3).converged
