"""QAOA ansatz for Max-Cut with exact expectation values and adjoint gradients."""
from __future__ import annotations

import numpy as np

from .graphs import Graph, cut_values
from .statevector import (apply_mixer, apply_phase, apply_sum_x, expectation, plus_state,
                          probabilities, sample_bitstrings)


class QAOA:
    r"""Depth-``p`` QAOA, :math:`|\psi\rangle=\prod_k e^{-i\beta_k B}e^{-i\gamma_k C}|+\rangle^n`.

    ``C`` is the (diagonal) cut-value operator and ``B = sum_j X_j``. The objective
    is the expected cut value, which is *maximised*. Parameters are stored as one
    vector ``[gamma_1..gamma_p, beta_1..beta_p]``.
    """

    def __init__(self, graph: Graph, p: int = 1) -> None:
        if p < 1:
            raise ValueError("depth must be >= 1")
        self.graph = graph
        self.p = p
        self.n = graph.n
        self.diag = cut_values(graph)
        self.optimum = float(self.diag.max())
        self.evaluations = 0

    def _split(self, params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        params = np.asarray(params, dtype=float)
        if params.shape != (2 * self.p,):
            raise ValueError(f"expected {2 * self.p} parameters")
        return params[: self.p], params[self.p:]

    def state(self, params: np.ndarray) -> np.ndarray:
        gammas, betas = self._split(params)
        psi = plus_state(self.n)
        for g, b in zip(gammas, betas):
            psi = apply_mixer(apply_phase(psi, self.diag, g), self.n, b)
        return psi

    def energy(self, params: np.ndarray) -> float:
        """Expected cut value."""
        self.evaluations += 1
        return expectation(self.state(params), self.diag)

    def gradient(self, params: np.ndarray) -> np.ndarray:
        """Exact gradient of the expected cut via adjoint differentiation (O(p) passes)."""
        gammas, betas = self._split(params)
        phi = self.state(params)
        lam = self.diag * phi
        grad_g = np.zeros(self.p)
        grad_b = np.zeros(self.p)
        for k in reversed(range(self.p)):
            # mixer layer: phi is the state after e^{-i b B}
            grad_b[k] = 2.0 * np.real(np.vdot(lam, -1j * apply_sum_x(phi, self.n)))
            phi = apply_mixer(phi, self.n, -betas[k])
            lam = apply_mixer(lam, self.n, -betas[k])
            # phase layer: phi is the state after e^{-i g C}
            grad_g[k] = 2.0 * np.real(np.vdot(lam, -1j * self.diag * phi))
            phi = apply_phase(phi, self.diag, -gammas[k])
            lam = apply_phase(lam, self.diag, -gammas[k])
        return np.concatenate([grad_g, grad_b])

    def numeric_gradient(self, params: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Central finite differences (reference for tests)."""
        params = np.asarray(params, dtype=float)
        out = np.zeros_like(params)
        for i in range(params.size):
            d = np.zeros_like(params)
            d[i] = eps
            out[i] = (self.energy(params + d) - self.energy(params - d)) / (2 * eps)
        return out

    def approximation_ratio(self, params: np.ndarray) -> float:
        return self.energy(params) / self.optimum

    def success_probability(self, params: np.ndarray) -> float:
        """Probability of measuring an optimal cut."""
        pr = probabilities(self.state(params))
        return float(pr[self.diag >= self.optimum - 1e-9].sum())

    def sample_energy(self, params: np.ndarray, shots: int, seed: int = 0) -> tuple[float, float]:
        """Shot-based mean cut and best cut observed."""
        idx = sample_bitstrings(self.state(params), shots, seed)
        vals = self.diag[idx]
        return float(vals.mean()), float(vals.max())

    def noisy_energy(self, params: np.ndarray, shots: int, seed: int = 0) -> float:
        """Shot-noise estimate of the energy."""
        return self.sample_energy(params, shots, seed)[0]
