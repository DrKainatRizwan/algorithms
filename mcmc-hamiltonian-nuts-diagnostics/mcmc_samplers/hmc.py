"""Hamiltonian Monte Carlo with leapfrog integration and dual-averaging step size."""
from __future__ import annotations

import numpy as np

from .targets import Target


def leapfrog(target: Target, q: np.ndarray, p: np.ndarray, eps: float,
             n_steps: int = 1, inv_mass: np.ndarray | None = None):
    """Volume-preserving, reversible leapfrog integrator.

    ``inv_mass`` is the diagonal of the inverse mass matrix (default identity).
    Returns the new ``(q, p)``.
    """
    m_inv = 1.0 if inv_mass is None else inv_mass
    q = q.copy()
    p = p + 0.5 * eps * target.grad_log_prob(q)
    for i in range(n_steps):
        q = q + eps * m_inv * p
        if i != n_steps - 1:
            p = p + eps * target.grad_log_prob(q)
    p = p + 0.5 * eps * target.grad_log_prob(q)
    return q, p


def hamiltonian(target: Target, q: np.ndarray, p: np.ndarray,
                inv_mass: np.ndarray | None = None) -> float:
    """Negative log of the joint density (up to a constant)."""
    m_inv = 1.0 if inv_mass is None else inv_mass
    return -target.log_prob(q) + 0.5 * float(np.sum(m_inv * p * p))


def find_reasonable_step_size(target: Target, q: np.ndarray, rng: np.random.Generator,
                              eps: float = 1.0) -> float:
    """Heuristic from Hoffman & Gelman (2014), Algorithm 4."""
    p = rng.normal(size=q.size)
    h0 = hamiltonian(target, q, p)
    q1, p1 = leapfrog(target, q, p, eps)
    dh = h0 - hamiltonian(target, q1, p1)
    if not np.isfinite(dh):
        dh = -np.inf
    a = 1.0 if dh > np.log(0.5) else -1.0
    for _ in range(60):
        if not a * dh > -a * np.log(2.0):
            break
        eps *= 2.0 ** a
        q1, p1 = leapfrog(target, q, p, eps)
        dh = h0 - hamiltonian(target, q1, p1)
        if not np.isfinite(dh):
            dh = -np.inf
    return eps


class DualAveraging:
    """Nesterov dual averaging of log step size (Hoffman & Gelman 2014, Alg. 5)."""

    def __init__(self, eps0: float, target_accept: float = 0.8, gamma: float = 0.05,
                 t0: float = 10.0, kappa: float = 0.75) -> None:
        self.mu = np.log(10.0 * eps0)
        self.delta, self.gamma, self.t0, self.kappa = target_accept, gamma, t0, kappa
        self.m = 0
        self.h_bar = 0.0
        self.log_eps_bar = 0.0
        self.eps = eps0

    def update(self, accept_stat: float) -> float:
        self.m += 1
        m = self.m
        w = 1.0 / (m + self.t0)
        self.h_bar = (1 - w) * self.h_bar + w * (self.delta - accept_stat)
        log_eps = self.mu - np.sqrt(m) / self.gamma * self.h_bar
        eta = m ** (-self.kappa)
        self.log_eps_bar = eta * log_eps + (1 - eta) * self.log_eps_bar
        self.eps = float(np.exp(log_eps))
        return self.eps

    @property
    def final(self) -> float:
        return float(np.exp(self.log_eps_bar))


class HMC:
    """Static-trajectory HMC with optional step-size adaptation during warm-up."""

    def __init__(self, target: Target, step: float | None = None, n_leapfrog: int = 20,
                 target_accept: float = 0.8, jitter: float = 0.0, seed: int = 0) -> None:
        self.target = target
        self.step = step
        self.n_leapfrog = n_leapfrog
        self.target_accept = target_accept
        self.jitter = jitter
        self.rng = np.random.default_rng(seed)
        self.accepted = 0
        self.proposed = 0
        self.divergences = 0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / max(self.proposed, 1)

    def _step(self, q: np.ndarray, eps: float) -> tuple[np.ndarray, float]:
        p = self.rng.normal(size=q.size)
        n = self.n_leapfrog
        if self.jitter:
            n = max(1, int(round(n * (1 + self.jitter * (2 * self.rng.random() - 1)))))
        h0 = hamiltonian(self.target, q, p)
        q1, p1 = leapfrog(self.target, q, p, eps, n)
        h1 = hamiltonian(self.target, q1, p1)
        dh = h0 - h1
        if not np.isfinite(dh):
            dh = -np.inf
        if dh < -1000:
            self.divergences += 1
        a = float(min(1.0, np.exp(min(dh, 0.0))))
        self.proposed += 1
        if self.rng.random() < a:
            self.accepted += 1
            return q1, a
        return q, a

    def sample(self, n: int, x0: np.ndarray, warmup: int = 0) -> np.ndarray:
        q = np.asarray(x0, float).copy()
        if self.step is None:
            eps = find_reasonable_step_size(self.target, q, self.rng)
        else:
            eps = self.step
        adapt = DualAveraging(eps, self.target_accept) if warmup > 0 else None
        out = np.empty((n, q.size))
        self.accepted = self.proposed = self.divergences = 0
        for i in range(warmup + n):
            q, a = self._step(q, eps)
            if adapt is not None and i < warmup:
                eps = adapt.update(a)
                if i == warmup - 1:
                    eps = adapt.final
            if i >= warmup:
                out[i - warmup] = q
        self.step_size_ = eps
        return out
