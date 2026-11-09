"""No-U-Turn Sampler (Hoffman & Gelman 2014, Algorithm 6, slice-sampling form)."""
from __future__ import annotations

import numpy as np

from .hmc import DualAveraging, find_reasonable_step_size, leapfrog, hamiltonian
from .targets import Target

_DELTA_MAX = 1000.0


class NUTS:
    """NUTS with dual-averaging step-size adaptation.

    Trajectories double in a random direction until the path makes a U-turn
    (``dtheta . r < 0`` at either end) or a divergence is detected.
    """

    def __init__(self, target: Target, target_accept: float = 0.8,
                 max_depth: int = 10, step: float | None = None, seed: int = 0) -> None:
        self.target = target
        self.target_accept = target_accept
        self.max_depth = max_depth
        self.step = step
        self.rng = np.random.default_rng(seed)
        self.divergences = 0
        self.grad_evals = 0
        self.depths: list[int] = []

    def _joint(self, q: np.ndarray, p: np.ndarray) -> float:
        return -hamiltonian(self.target, q, p)

    def _build_tree(self, q, p, log_u, v, j, eps, joint0):
        """Return (q-, p-, q+, p+, q', n, s, alpha_sum, n_alpha)."""
        if j == 0:
            q1, p1 = leapfrog(self.target, q, p, v * eps)
            self.grad_evals += 1
            joint = self._joint(q1, p1)
            if not np.isfinite(joint):
                joint = -np.inf
            n = int(log_u <= joint)
            ok = log_u < _DELTA_MAX + joint
            if not ok:
                self.divergences += 1
            alpha = min(1.0, float(np.exp(min(joint - joint0, 0.0)))) if np.isfinite(joint) else 0.0
            return q1, p1, q1, p1, q1, n, int(ok), alpha, 1
        qm, pm, qp, pp, qs, n, s, a, na = self._build_tree(q, p, log_u, v, j - 1, eps, joint0)
        if s == 1:
            if v == -1:
                qm, pm, _, _, q2, n2, s2, a2, na2 = self._build_tree(qm, pm, log_u, v, j - 1, eps, joint0)
            else:
                _, _, qp, pp, q2, n2, s2, a2, na2 = self._build_tree(qp, pp, log_u, v, j - 1, eps, joint0)
            if n + n2 > 0 and self.rng.random() < n2 / (n + n2):
                qs = q2
            a += a2
            na += na2
            dq = qp - qm
            s = s2 * int(dq @ pm >= 0) * int(dq @ pp >= 0)
            n += n2
        return qm, pm, qp, pp, qs, n, s, a, na

    def _transition(self, q0: np.ndarray, eps: float):
        p0 = self.rng.normal(size=q0.size)
        joint0 = self._joint(q0, p0)
        log_u = joint0 - self.rng.exponential()
        qm = qp = q0
        pm = pp = p0
        q_new = q0
        n, s, j = 1, 1, 0
        alpha, n_alpha = 0.0, 1
        while s == 1 and j < self.max_depth:
            v = 1 if self.rng.random() < 0.5 else -1
            if v == -1:
                qm, pm, _, _, q1, n1, s1, alpha, n_alpha = self._build_tree(qm, pm, log_u, v, j, eps, joint0)
            else:
                _, _, qp, pp, q1, n1, s1, alpha, n_alpha = self._build_tree(qp, pp, log_u, v, j, eps, joint0)
            if s1 == 1 and self.rng.random() < min(1.0, n1 / n):
                q_new = q1
            n += n1
            dq = qp - qm
            s = s1 * int(dq @ pm >= 0) * int(dq @ pp >= 0)
            j += 1
        self.depths.append(j)
        return q_new, alpha / n_alpha

    def sample(self, n: int, x0: np.ndarray, warmup: int = 200) -> np.ndarray:
        q = np.asarray(x0, float).copy()
        eps = self.step if self.step is not None else find_reasonable_step_size(self.target, q, self.rng)
        adapt = DualAveraging(eps, self.target_accept) if warmup > 0 else None
        out = np.empty((n, q.size))
        self.divergences = self.grad_evals = 0
        self.depths = []
        for i in range(warmup + n):
            q, a = self._transition(q, eps)
            if adapt is not None and i < warmup:
                eps = adapt.update(a)
                if i == warmup - 1:
                    eps = adapt.final
            if i == warmup - 1:
                self.divergences = 0  # report post-warm-up divergences only
            if i >= warmup:
                out[i - warmup] = q
        self.step_size_ = eps
        return out
