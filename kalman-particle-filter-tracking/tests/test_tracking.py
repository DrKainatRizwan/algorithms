import numpy as np
import pytest
from tracking_filters import *


def linear_setup(seed=0, n=300):
    truth, zs, F, Q, H, R = simulate_cv(n, 1.0, 0.05, 2.0, seed)
    kf = KalmanFilter(F, Q, H, R, np.zeros(4), np.eye(4) * 10)
    return truth, zs, F, Q, H, R, kf


def test_cv_matrices_shapes_and_symmetry():
    F, Q = cv_matrices(0.5, 1.0)
    assert F.shape == (4, 4) and np.allclose(Q, Q.T)
    assert np.all(np.linalg.eigvalsh(Q) >= -1e-12)
    assert F[0, 2] == 0.5


def test_wrap_angle():
    assert np.isclose(wrap_angle(3 * np.pi / 2), -np.pi / 2)
    assert np.isclose(wrap_angle(-3 * np.pi), -np.pi)


def test_kf_beats_raw_measurements():
    truth, zs, *_, kf = linear_setup()
    xs = kf.run(zs)[0]
    assert rmse(xs[:, :2], truth[:, :2]) < 0.7 * rmse(zs, truth[:, :2])


def test_kf_matches_scalar_closed_form():
    kf = KalmanFilter([[1.0]], [[0.0]], [[1.0]], [[1.0]], [0.0], [[1.0]])
    kf.predict(); kf.update([2.0])
    assert np.isclose(kf.x[0], 1.0) and np.isclose(kf.P[0, 0], 0.5)


def test_covariance_stays_symmetric_psd():
    truth, zs, *_, kf = linear_setup()
    Ps = kf.run(zs)[1]
    for P in Ps:
        assert np.allclose(P, P.T, atol=1e-9)
        assert np.linalg.eigvalsh(P).min() > 0


def test_kf_consistency_nees_nis():
    truth, zs, *_, kf = linear_setup(seed=2, n=500)
    xs, Ps, xp, Pp, ys, Ss = kf.run(zs)
    assert 0.8 < mean_consistency(nees(xs[50:], truth[50:], Ps[50:]), 4) < 1.25
    assert 0.8 < mean_consistency(nis(ys, Ss), 2) < 1.25


def test_rts_smoother_improves():
    truth, zs, F, *_, kf = linear_setup(seed=4)
    xs, Ps, xp, Pp, *_ = kf.run(zs)
    sm, Psm = rts_smoother(F, xs, Ps, xp, Pp)
    assert rmse(sm[:, :2], truth[:, :2]) < rmse(xs[:, :2], truth[:, :2])
    assert np.all(np.trace(Psm, axis1=1, axis2=2) <= np.trace(Ps, axis1=1, axis2=2) + 1e-9)
    assert np.allclose(sm[-1], xs[-1])


def test_log_likelihood_finite_and_higher_for_good_measurement():
    kf = KalmanFilter([[1.0]], [[0.1]], [[1.0]], [[1.0]], [0.0], [[1.0]])
    kf.predict(); kf.update([0.0]); good = kf.log_likelihood()
    kf2 = KalmanFilter([[1.0]], [[0.1]], [[1.0]], [[1.0]], [0.0], [[1.0]])
    kf2.predict(); kf2.update([8.0]); bad = kf2.log_likelihood()
    assert np.isfinite(good) and good > bad


def test_numeric_jacobian_matches_analytic():
    h = range_bearing()
    x = np.array([3.0, 4.0, 0.0, 0.0])
    J = numeric_jacobian(h, x)
    assert np.allclose(J[0, :2], [0.6, 0.8], atol=1e-6)
    assert np.allclose(J[1, :2], [-4 / 25, 3 / 25], atol=1e-6)


def test_ct_transition_preserves_speed_and_reduces_to_cv():
    f = ct_transition(1.0)
    s = f(np.array([0, 0, 3.0, 4.0, 0.2]))
    assert np.isclose(np.hypot(s[2], s[3]), 5.0)
    s0 = f(np.array([0, 0, 3.0, 4.0, 0.0]))
    assert np.allclose(s0[:2], [3, 4])


def test_ekf_equals_kf_on_linear_model():
    truth, zs, F, Q, H, R, kf = linear_setup(seed=5, n=60)
    ekf = ExtendedKalmanFilter(lambda x: F @ x, lambda x: H @ x, Q, R, np.zeros(4), np.eye(4) * 10,
                               F_jac=lambda x: F, H_jac=lambda x: H)
    assert np.allclose(ekf.run(zs)[0], kf.run(zs)[0], atol=1e-8)


def test_ukf_matches_kf_on_linear_model():
    truth, zs, F, Q, H, R, kf = linear_setup(seed=6, n=60)
    ukf = UnscentedKalmanFilter(lambda x: F @ x, lambda x: H @ x, Q, R, np.zeros(4), np.eye(4) * 10)
    assert np.allclose(ukf.run(zs)[0], kf.run(zs)[0], atol=1e-5)


def test_ukf_weights_sum_to_one():
    ukf = UnscentedKalmanFilter(lambda x: x, lambda x: x, np.eye(3), np.eye(3), np.zeros(3), np.eye(3), alpha=0.5)
    assert np.isclose(ukf.Wm.sum(), 1.0)
    pts = ukf.sigma_points(ukf.x, ukf.P)
    assert np.allclose(ukf.Wm @ pts, ukf.x)


def test_resamplers_valid_and_proportional():
    rng = np.random.default_rng(0)
    w = np.array([0.7, 0.1, 0.1, 0.1])
    for fn in (systematic_resample, multinomial_resample):
        idx = fn(np.tile(w, 250) / 250, rng)
        assert idx.min() >= 0 and idx.max() < 1000
        assert abs(np.mean(idx % 4 == 0) - 0.7) < 0.06
    assert np.isclose(effective_sample_size(np.full(10, 0.1)), 10.0)


def _radar_problem(seed=3, n=80):
    truth, zs = simulate_ct(n, 1.0, 0.3, 2.0, 0.01, seed, omega_switch={40: 0.08})
    return truth, zs


def _pf(zs, seed=7, N=800):
    f = ct_transition(1.0); h = range_bearing()
    x0 = np.array([100.0, 50.0, 5.0, 2.0, 0.0])
    sd = np.array([10, 10, 3, 3, 0.1])
    init = x0 + np.random.default_rng(1).standard_normal((N, 5)) * sd
    qs = np.sqrt([0.01, 0.01, 0.3, 0.3, 0.003**2 * 10])
    tr = lambda p, rng: np.array([f(s) for s in p]) + qs * rng.standard_normal(p.shape)
    sig = np.array([2.0, 0.01])
    def ll(z, p):
        r = bearing_residual(np.broadcast_to(z, (len(p), 2)).copy(), h(p))
        return -0.5 * np.sum((r / sig) ** 2, axis=1)
    return ParticleFilter(tr, ll, init, seed=seed), x0, sd


def test_particle_filter_tracks_radar_target():
    truth, zs = _radar_problem()
    pf, *_ = _pf(zs)
    est = pf.run(zs)
    raw = np.stack([zs[:, 0] * np.cos(zs[:, 1]), zs[:, 0] * np.sin(zs[:, 1])], 1)
    assert rmse(est[:, :2], truth[:, :2]) < rmse(raw, truth[:, :2])
    assert np.isclose(pf.w.sum(), 1.0)


def test_particle_filter_deterministic_given_seed():
    truth, zs = _radar_problem(n=20)
    a = _pf(zs, seed=11)[0].run(zs)
    b = _pf(zs, seed=11)[0].run(zs)
    assert np.array_equal(a, b)


def test_ekf_ukf_track_radar_target():
    truth, zs = _radar_problem()
    f = ct_transition(1.0); h = range_bearing()
    Q = np.diag([0.01, 0.01, 0.3, 0.3, 0.003**2 * 10]); R = np.diag([4.0, 1e-4])
    x0 = np.array([100.0, 50.0, 5.0, 2.0, 0.0]); P0 = np.diag([10, 10, 3, 3, 0.1]) ** 2
    raw = np.stack([zs[:, 0] * np.cos(zs[:, 1]), zs[:, 0] * np.sin(zs[:, 1])], 1)
    for cls in (ExtendedKalmanFilter, UnscentedKalmanFilter):
        est = cls(f, h, Q, R, x0, P0, residual=bearing_residual).run(zs)[0]
        assert np.all(np.isfinite(est))
        assert rmse(est[:, :2], truth[:, :2]) < 1.5 * rmse(raw, truth[:, :2])


def test_bearing_residual_wraps():
    r = bearing_residual(np.array([10.0, 3.1]), np.array([9.0, -3.1]))
    assert abs(r[1]) < 0.1 and np.isclose(r[0], 1.0)
