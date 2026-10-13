"""Gradients of the VQE energy: parameter shift and finite differences."""
from __future__ import annotations

import numpy as np

from .ansatz import Ansatz
from .pauli import PauliHamiltonian


def parameter_shift_gradient(ansatz: Ansatz, h: PauliHamiltonian,
                             params: np.ndarray) -> np.ndarray:
    """Exact gradient via the parameter-shift rule.

    Each gate is ``exp(-i theta P/2)`` so
    ``dE/dtheta_gate = [E(theta + pi/2) - E(theta - pi/2)] / 2``.
    Parameters shared by several gates sum their per-gate contributions
    (chain rule).
    """
    params = np.asarray(params, dtype=float)
    grad = np.zeros(ansatz.num_params)
    s = np.pi / 2
    for pos, op in enumerate(ansatz.ops):
        if op.kind != "rot" or op.param < 0:
            continue
        ep = h.expectation(ansatz.run(params, {pos: +s}))
        em = h.expectation(ansatz.run(params, {pos: -s}))
        grad[op.param] += op.sign * 0.5 * (ep - em)
    return grad


def finite_difference_gradient(f, x: np.ndarray, eps: float = 1e-6
                               ) -> np.ndarray:
    """Central finite differences of a scalar function."""
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x)
    for i in range(x.size):
        d = np.zeros_like(x)
        d[i] = eps
        g[i] = (f(x + d) - f(x - d)) / (2 * eps)
    return g
