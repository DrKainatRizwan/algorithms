"""Motion and measurement models plus trajectory simulators."""
from __future__ import annotations
from typing import Callable
import numpy as np


def wrap_angle(a):
    """Wrap angle(s) to [-pi, pi)."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def cv_matrices(dt: float, q: float, dim: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Constant-velocity transition F and discrete white-noise-acceleration Q.

    State layout is [x, y, vx, vy] (positions first, then velocities).
    """
    F1 = np.array([[1.0, dt], [0.0, 1.0]])
    Q1 = q * np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]])
    F = np.zeros((2 * dim, 2 * dim))
    Q = np.zeros_like(F)
    for d in range(dim):
        idx = [d, dim + d]
        F[np.ix_(idx, idx)] = F1
        Q[np.ix_(idx, idx)] = Q1
    return F, Q


def ct_transition(dt: float) -> Callable[[np.ndarray], np.ndarray]:
    """Coordinated-turn transition for state [x, y, vx, vy, omega]."""
    def f(s: np.ndarray) -> np.ndarray:
        x, y, vx, vy, w = s
        if abs(w) < 1e-9:
            return np.array([x + vx * dt, y + vy * dt, vx, vy, w])
        sn, cs = np.sin(w * dt), np.cos(w * dt)
        return np.array([
            x + (sn * vx - (1 - cs) * vy) / w,
            y + ((1 - cs) * vx + sn * vy) / w,
            cs * vx - sn * vy,
            sn * vx + cs * vy,
            w])
    return f


def linear_position(dim: int = 2) -> np.ndarray:
    """Measurement matrix observing positions only from a [pos, vel] state."""
    H = np.zeros((dim, 2 * dim))
    H[:, :dim] = np.eye(dim)
    return H


def range_bearing(sensor: tuple[float, float] = (0.0, 0.0)) -> Callable[[np.ndarray], np.ndarray]:
    """Nonlinear radar measurement h(s) = [range, bearing] (works on 1-D or (N, n) input)."""
    sx, sy = sensor
    def h(s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=float)
        dx, dy = s[..., 0] - sx, s[..., 1] - sy
        return np.stack([np.hypot(dx, dy), np.arctan2(dy, dx)], axis=-1)
    return h


def bearing_residual(z: np.ndarray, zhat: np.ndarray) -> np.ndarray:
    """Innovation for [range, bearing] measurements with angle wrapping."""
    r = np.asarray(z, dtype=float) - np.asarray(zhat, dtype=float)
    r[..., 1] = wrap_angle(r[..., 1])
    return r


def numeric_jacobian(f: Callable[[np.ndarray], np.ndarray], x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Central-difference Jacobian of f at x."""
    x = np.asarray(x, dtype=float)
    cols = []
    for i in range(x.size):
        d = np.zeros_like(x)
        d[i] = eps
        cols.append((f(x + d) - f(x - d)) / (2 * eps))
    return np.stack(cols, axis=-1)


def simulate_cv(n: int, dt: float, q: float, r_std: float, seed: int,
                x0: np.ndarray | None = None):
    """Simulate a 2-D CV trajectory with position measurements.

    Returns (truth (n,4), measurements (n,2), F, Q, H, R).
    """
    rng = np.random.default_rng(seed)
    F, Q = cv_matrices(dt, q)
    H = linear_position()
    R = r_std**2 * np.eye(2)
    x = np.array([0.0, 0.0, 1.0, 0.5]) if x0 is None else np.asarray(x0, float)
    L = np.linalg.cholesky(Q + 1e-12 * np.eye(4))
    truth, zs = [], []
    for _ in range(n):
        x = F @ x + L @ rng.standard_normal(4)
        truth.append(x)
        zs.append(H @ x + r_std * rng.standard_normal(2))
    return np.array(truth), np.array(zs), F, Q, H, R


def simulate_ct(n: int, dt: float, q_acc: float, r_range: float, r_bearing: float,
                seed: int, sensor: tuple[float, float] = (0.0, 0.0),
                x0: np.ndarray | None = None, omega_switch: dict[int, float] | None = None):
    """Simulate a maneuvering target (coordinated turn) seen by a radar.

    ``omega_switch`` maps step index -> new turn rate. Returns (truth (n,5), meas (n,2)).
    """
    rng = np.random.default_rng(seed)
    f = ct_transition(dt)
    h = range_bearing(sensor)
    x = np.array([100.0, 50.0, 5.0, 2.0, 0.0]) if x0 is None else np.asarray(x0, float)
    truth, zs = [], []
    for k in range(n):
        if omega_switch and k in omega_switch:
            x[4] = omega_switch[k]
        x = f(x)
        x[2:4] += q_acc * dt * rng.standard_normal(2)
        truth.append(x.copy())
        z = h(x) + np.array([r_range, r_bearing]) * rng.standard_normal(2)
        zs.append(z)
    return np.array(truth), np.array(zs)
