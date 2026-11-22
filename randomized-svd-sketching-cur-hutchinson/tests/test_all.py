import numpy as np
import pytest

from randomized_linalg import *


def rng(s=0):
    return np.random.default_rng(s)


def test_fwht_matches_hadamard_matrix():
    n = 8
    H = np.array([[(-1) ** bin(i & j).count("1") for j in range(n)] for i in range(n)], float)
    x = rng().standard_normal((n, 3))
    assert np.allclose(fwht(x), H @ x)
    with pytest.raises(ValueError):
        fwht(np.ones(6))


@pytest.mark.parametrize("kind", ["gaussian", "srht", "countsketch", "sparse_sign"])
def test_subspace_embedding_preserves_norms(kind):
    r = rng(1)
    A = r.standard_normal((512, 5))
    S = make_sketch(kind, 160, 512, r)
    Q = np.linalg.qr(A)[0]
    sv = np.linalg.svd(S.apply(Q), compute_uv=False)
    assert sv.min() > 0.5 and sv.max() < 1.5


def test_sketch_unbiased_gram():
    r = rng(2)
    A = r.standard_normal((64, 3))
    acc = np.zeros((3, 3))
    for _ in range(400):
        SA = make_sketch("countsketch", 32, 64, r).apply(A)
        acc += SA.T @ SA
    assert np.allclose(acc / 400, A.T @ A, atol=3.0)


def test_jl_dim_and_distances():
    assert jl_min_dim(100, 0.5) < jl_min_dim(100, 0.25)
    r = rng(3)
    X = r.standard_normal((40, 2000))
    m = jl_min_dim(40, 0.5)
    Y = make_sketch("gaussian", m, 2000, r).apply(X.T).T
    d0 = np.linalg.norm(X[0] - X[1:], axis=1)
    d1 = np.linalg.norm(Y[0] - Y[1:], axis=1)
    assert np.all(np.abs(d1 / d0 - 1) < 0.5)


def test_rsvd_near_optimal_with_power_iterations():
    A = decaying_spectrum(200, 150, "poly", rng(4), rate=0.6)
    s_true = np.linalg.svd(A, compute_uv=False)
    U, s, Vt = randomized_svd(A, 10, 10, 3, rng(5))
    err = np.linalg.norm(A - (U * s) @ Vt, 2)
    assert err < 1.1 * s_true[10]
    assert np.allclose(U.T @ U, np.eye(10), atol=1e-10)


def test_power_iterations_improve_accuracy():
    A = decaying_spectrum(300, 300, "poly", rng(6), rate=0.3)
    errs = []
    for q in (0, 2):
        U, s, Vt = randomized_svd(A, 10, 5, q, rng(7))
        errs.append(np.linalg.norm(A - (U * s) @ Vt, 2))
    assert errs[1] < errs[0]


def test_exact_recovery_of_low_rank():
    A = low_rank_plus_noise(80, 60, 5, 0.0, rng(8))
    U, s, Vt = randomized_svd(A, 5, 5, 1, rng(9))
    assert np.allclose((U * s) @ Vt, A, atol=1e-9)


def test_adaptive_range_finder_meets_tolerance():
    A = low_rank_plus_noise(120, 100, 8, 1e-3, rng(10))
    Q = adaptive_range_finder(A, 1e-2, block=4, rng=rng(11))
    rel = np.linalg.norm(A - Q @ (Q.T @ A)) / np.linalg.norm(A)
    assert rel <= 1e-2 and Q.shape[1] < 40


def test_single_pass_svd_reasonable():
    A = decaying_spectrum(150, 120, "exp", rng(12), rate=0.7)
    U, s, Vt = single_pass_svd(A, 8, rng(13))
    assert np.linalg.norm(A - (U * s) @ Vt, 2) < 1e-2


def test_spectral_error_estimate():
    A = decaying_spectrum(100, 100, "poly", rng(14), rate=0.5)
    Q = randomized_range_finder(A, 15, 2, rng(15))
    est = spectral_error_estimate(A, Q, rng(16))
    true = np.linalg.norm(A - Q @ (Q.T @ A), 2)
    assert est <= true * (1 + 1e-9) and est > 0.8 * true


def test_pivoted_qr_picks_independent_columns():
    r = rng(17)
    B = r.standard_normal((30, 4))
    A = B @ r.standard_normal((4, 20))
    J = pivoted_qr_columns(A, 4)
    assert len(set(J)) == 4 and np.linalg.matrix_rank(A[:, J]) == 4


def test_leverage_scores_sum_to_k():
    A = low_rank_plus_noise(50, 40, 6, 0.1, rng(18))
    assert np.isclose(leverage_scores(A, 6, 0).sum(), 6)
    assert np.isclose(leverage_scores(A, 6, 1).sum(), 6)


@pytest.mark.parametrize("method", ["leverage", "pivot", "deim", "uniform"])
def test_cur_exact_on_low_rank(method):
    A = low_rank_plus_noise(60, 50, 4, 0.0, rng(19))
    C, U, R, cols, rows = cur_decomposition(A, 4, method, c=8, r=8, rng=rng(20))
    assert np.allclose(C @ U @ R, A, atol=1e-7)


def test_cur_pivot_beats_uniform_on_noisy():
    A = low_rank_plus_noise(100, 80, 5, 0.01, rng(21))
    e = {}
    for m in ("pivot", "uniform"):
        C, U, R, *_ = cur_decomposition(A, 5, m, rng=rng(22))
        e[m] = np.linalg.norm(A - C @ U @ R)
    assert e["pivot"] <= e["uniform"]


def test_interpolative_decomposition():
    A = low_rank_plus_noise(40, 30, 3, 0.0, rng(23))
    J, T = interpolative_decomposition(A, 3)
    assert np.allclose(A[:, J] @ T, A, atol=1e-8)
    assert np.allclose(T[:, J], np.eye(3), atol=1e-8)


def test_sketch_and_solve_residual_guarantee():
    A, b, _ = ill_conditioned_ls(2000, 10, 10.0, rng(24))
    x = sketch_and_solve(A, b, 200, "srht", rng(25))
    xo = np.linalg.lstsq(A, b, rcond=None)[0]
    assert np.linalg.norm(A @ x - b) <= 1.3 * np.linalg.norm(A @ xo - b)


def test_sketch_precondition_converges_fast_on_ill_conditioned():
    A, b, _ = ill_conditioned_ls(3000, 20, 1e6, rng(26))
    x, it, cond = sketch_precondition_cgls(A, b, "srht", 4.0, 1e-12, rng(27))
    xo = np.linalg.lstsq(A, b, rcond=None)[0]
    assert cond < 10 and it < 60
    assert np.linalg.norm(A @ x - b) <= np.linalg.norm(A @ xo - b) * (1 + 1e-8)
    _, plain_it = cgls(lambda v: A @ v, lambda u: A.T @ u, b, 20, 1e-12, 2000)
    assert plain_it > 3 * it


def test_hutchinson_unbiased_and_hutchpp_lower_error():
    r = rng(28)
    Qm = np.linalg.qr(r.standard_normal((200, 200)))[0]
    A = (Qm * (np.arange(1, 201, dtype=float) ** -2)) @ Qm.T
    tr = np.trace(A)
    mv = lambda X: A @ X
    h = [hutchinson(mv, 200, 30, rng(i)) for i in range(400)]
    hp = [hutchpp(mv, 200, 30, rng(i)) for i in range(400)]
    assert abs(np.mean(h) - tr) < 3 * np.std(h) / np.sqrt(len(h))
    assert np.mean(np.abs(np.array(hp) - tr)) < 0.2 * np.mean(np.abs(np.array(h) - tr))
    with pytest.raises(ValueError):
        hutchinson(mv, 200, 5, r, "bad")
