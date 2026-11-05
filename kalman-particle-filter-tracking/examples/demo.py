"""Compare KF, EKF, UKF and particle filter on linear and radar tracking problems."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from tracking_filters import *


def linear_demo():
    truth, zs, F, Q, H, R = simulate_cv(200, 1.0, 0.05, 2.0, seed=1)
    kf = KalmanFilter(F, Q, H, R, np.zeros(4), np.diag([10, 10, 5, 5.0]))
    xs, Ps, xp, Pp, ys, Ss = kf.run(zs)
    sm, _ = rts_smoother(F, xs, Ps, xp, Pp)
    print("== Linear CV tracking (200 steps, sigma_z=2) ==")
    print(f"raw measurement pos RMSE : {rmse(zs, truth[:, :2]):.3f}")
    print(f"Kalman pos RMSE          : {rmse(xs[:, :2], truth[:, :2]):.3f}")
    print(f"RTS smoother pos RMSE    : {rmse(sm[:, :2], truth[:, :2]):.3f}")
    print(f"NEES/dim {mean_consistency(nees(xs, truth, Ps), 4):.2f}  NIS/dim {mean_consistency(nis(ys, Ss), 2):.2f}")


def radar_demo():
    dt, n = 1.0, 150
    sensor = (0.0, 0.0)
    truth, zs = simulate_ct(n, dt, 0.3, 2.0, 0.01, seed=3, omega_switch={50: 0.08, 100: -0.06})
    h = range_bearing(sensor)
    f = ct_transition(dt)
    Rm = np.diag([2.0**2, 0.01**2])
    x0 = np.array([100.0, 50.0, 5.0, 2.0, 0.0])
    P0 = np.diag([10, 10, 3, 3, 0.1])**2
    Q = np.diag([0.01, 0.01, 0.3, 0.3, 0.003**2 * 10])
    ekf = ExtendedKalmanFilter(f, h, Q, Rm, x0, P0, residual=bearing_residual)

    def zmean(Z, W):
        s = W @ (Z[:, 0] * 1.0)
        a = np.arctan2(W @ np.sin(Z[:, 1]), W @ np.cos(Z[:, 1]))
        return np.array([s, a])
    ukf = UnscentedKalmanFilter(f, h, Q, Rm, x0, P0, residual=bearing_residual, mean_fn_z=zmean)
    rows = {}
    t = time.time(); rows["EKF"] = ekf.run(zs)[0]; te = time.time() - t
    t = time.time(); rows["UKF"] = ukf.run(zs)[0]; tu = time.time() - t

    rng0 = np.random.default_rng(5)
    N = 1000
    init = x0 + rng0.standard_normal((N, 5)) * np.sqrt(np.diag(P0))
    qs = np.sqrt(np.diag(Q))
    trans = lambda p, rng: np.array([f(s) for s in p]) + qs * rng.standard_normal(p.shape)
    sig = np.array([2.0, 0.01])
    def loglik(z, p):
        r = bearing_residual(np.broadcast_to(z, (len(p), 2)).copy(), h(p))
        return -0.5 * np.sum((r / sig) ** 2, axis=1)
    pf = ParticleFilter(trans, loglik, init, seed=7)
    t = time.time(); rows["PF"] = pf.run(zs); tp = time.time() - t
    print("\n== Maneuvering target, radar range/bearing (150 steps) ==")
    raw = np.stack([zs[:, 0] * np.cos(zs[:, 1]), zs[:, 0] * np.sin(zs[:, 1])], 1)
    print(f"raw polar->xy pos RMSE: {rmse(raw, truth[:, :2]):.3f}")
    for name, tt in (("EKF", te), ("UKF", tu), ("PF", tp)):
        e = rows[name]
        print(f"{name:3s} pos RMSE {rmse(e[:, :2], truth[:, :2]):.3f}  vel RMSE {rmse(e[:, 2:4], truth[:, 2:4]):.3f}  time {tt:.2f}s")
    print(f"PF resampled {pf.resample_count}/{n} steps")


if __name__ == "__main__":
    linear_demo()
    radar_demo()
