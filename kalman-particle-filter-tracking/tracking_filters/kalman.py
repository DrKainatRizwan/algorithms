"""Kalman filter family: linear KF, EKF, UKF and the RTS smoother."""
from __future__ import annotations
from typing import Callable
import numpy as np
from .models import numeric_jacobian

Residual = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _sub(z, zhat):
    return z - zhat


class KalmanFilter:
    """Linear-Gaussian Kalman filter (Joseph-form covariance update)."""

    def __init__(self, F, Q, H, R, x0, P0):
        self.F, self.Q, self.H, self.R = map(np.asarray, (F, Q, H, R))
        self.x = np.asarray(x0, float).copy()
        self.P = np.asarray(P0, float).copy()
        self.last_innovation = None
        self.last_S = None

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x, self.P

    def update(self, z):
        S = self.H @ self.P @ self.H.T + self.R
        K = np.linalg.solve(S, self.H @ self.P).T
        y = np.asarray(z, float) - self.H @ self.x
        self.x = self.x + K @ y
        I_KH = np.eye(len(self.x)) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        self.last_innovation, self.last_S = y, S
        return self.x, self.P

    def log_likelihood(self) -> float:
        """Gaussian log-likelihood of the most recent innovation."""
        y, S = self.last_innovation, self.last_S
        _, logdet = np.linalg.slogdet(S)
        return -0.5 * (y @ np.linalg.solve(S, y) + logdet + len(y) * np.log(2 * np.pi))

    def run(self, zs):
        """Filter a measurement sequence; returns (xs, Ps, xs_pred, Ps_pred, innovations, Ss)."""
        xs, Ps, xp, Pp, ys, Ss = [], [], [], [], [], []
        for z in zs:
            a, B = self.predict()
            xp.append(a.copy()); Pp.append(B.copy())
            self.update(z)
            xs.append(self.x.copy()); Ps.append(self.P.copy())
            ys.append(self.last_innovation); Ss.append(self.last_S)
        return tuple(np.array(v) for v in (xs, Ps, xp, Pp, ys, Ss))


class ExtendedKalmanFilter:
    """EKF with arbitrary nonlinear f and h; Jacobians are numeric unless supplied."""

    def __init__(self, f, h, Q, R, x0, P0, F_jac=None, H_jac=None, residual: Residual = _sub):
        self.f, self.h = f, h
        self.Q, self.R = np.asarray(Q), np.asarray(R)
        self.x = np.asarray(x0, float).copy()
        self.P = np.asarray(P0, float).copy()
        self.F_jac = F_jac or (lambda x: numeric_jacobian(f, x))
        self.H_jac = H_jac or (lambda x: numeric_jacobian(h, x))
        self.residual = residual
        self.last_innovation = None
        self.last_S = None

    def predict(self):
        F = self.F_jac(self.x)
        self.x = self.f(self.x)
        self.P = F @ self.P @ F.T + self.Q
        return self.x, self.P

    def update(self, z):
        H = self.H_jac(self.x)
        S = H @ self.P @ H.T + self.R
        K = np.linalg.solve(S, H @ self.P).T
        y = self.residual(np.asarray(z, float), self.h(self.x))
        self.x = self.x + K @ y
        I_KH = np.eye(len(self.x)) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        self.last_innovation, self.last_S = y, S
        return self.x, self.P

    def run(self, zs):
        xs, Ps, ys, Ss = [], [], [], []
        for z in zs:
            self.predict(); self.update(z)
            xs.append(self.x.copy()); Ps.append(self.P.copy())
            ys.append(self.last_innovation); Ss.append(self.last_S)
        return tuple(np.array(v) for v in (xs, Ps, ys, Ss))


class UnscentedKalmanFilter:
    """UKF using the scaled unscented transform (Julier / Wan-van der Merwe)."""

    def __init__(self, f, h, Q, R, x0, P0, alpha=1.0, beta=2.0, kappa=None,
                 residual: Residual = _sub, mean_fn_z=None):
        self.f, self.h = f, h
        self.Q, self.R = np.asarray(Q), np.asarray(R)
        self.x = np.asarray(x0, float).copy()
        self.P = np.asarray(P0, float).copy()
        n = self.x.size
        self.n = n
        kappa = 3.0 - n if kappa is None else kappa
        self.lam = alpha**2 * (n + kappa) - n
        self.Wm = np.full(2 * n + 1, 0.5 / (n + self.lam))
        self.Wc = self.Wm.copy()
        self.Wm[0] = self.lam / (n + self.lam)
        self.Wc[0] = self.Wm[0] + (1 - alpha**2 + beta)
        self.residual = residual
        self.mean_fn_z = mean_fn_z
        self.last_innovation = None
        self.last_S = None

    def sigma_points(self, x, P) -> np.ndarray:
        """2n+1 sigma points around (x, P)."""
        n = self.n
        L = np.linalg.cholesky((n + self.lam) * (P + 1e-12 * np.eye(n)))
        pts = np.empty((2 * n + 1, n))
        pts[0] = x
        pts[1:n + 1] = x + L.T
        pts[n + 1:] = x - L.T
        return pts

    def predict(self):
        X = np.array([self.f(p) for p in self.sigma_points(self.x, self.P)])
        self.x = self.Wm @ X
        d = X - self.x
        self.P = (self.Wc[:, None] * d).T @ d + self.Q
        return self.x, self.P

    def _zmean(self, Z):
        if self.mean_fn_z is not None:
            return self.mean_fn_z(Z, self.Wm)
        return self.Wm @ Z

    def update(self, z):
        X = self.sigma_points(self.x, self.P)
        Z = np.array([self.h(p) for p in X])
        zm = self._zmean(Z)
        dz = np.array([self.residual(zi, zm) for zi in Z])
        dx = X - self.x
        S = (self.Wc[:, None] * dz).T @ dz + self.R
        Pxz = (self.Wc[:, None] * dx).T @ dz
        K = np.linalg.solve(S, Pxz.T).T
        y = self.residual(np.asarray(z, float), zm)
        self.x = self.x + K @ y
        self.P = self.P - K @ S @ K.T
        self.last_innovation, self.last_S = y, S
        return self.x, self.P

    def run(self, zs):
        xs, Ps, ys, Ss = [], [], [], []
        for z in zs:
            self.predict(); self.update(z)
            xs.append(self.x.copy()); Ps.append(self.P.copy())
            ys.append(self.last_innovation); Ss.append(self.last_S)
        return tuple(np.array(v) for v in (xs, Ps, ys, Ss))


def rts_smoother(F, xs, Ps, xs_pred, Ps_pred):
    """Rauch-Tung-Striebel fixed-interval smoother for a linear model.

    Takes the outputs of ``KalmanFilter.run`` and returns smoothed (xs, Ps).
    """
    n = len(xs)
    xs_s, Ps_s = xs.copy(), Ps.copy()
    for k in range(n - 2, -1, -1):
        C = Ps[k] @ F.T @ np.linalg.inv(Ps_pred[k + 1])
        xs_s[k] = xs[k] + C @ (xs_s[k + 1] - xs_pred[k + 1])
        Ps_s[k] = Ps[k] + C @ (Ps_s[k + 1] - Ps_pred[k + 1]) @ C.T
    return xs_s, Ps_s
