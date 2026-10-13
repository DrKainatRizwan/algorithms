"""VQE driver, shot-noise energy estimation and ADAPT-VQE."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .ansatz import Ansatz
from .gradients import parameter_shift_gradient
from .optimizers import OptResult, adam, nelder_mead, rotosolve, spsa
from .pauli import PauliHamiltonian, apply_pauli_string


def shot_energy(h: PauliHamiltonian, state: np.ndarray, shots: int,
                rng: np.random.Generator) -> float:
    """Energy estimate from ``shots`` measurements per Pauli term.

    Each Pauli term has eigenvalues +-1, so a measurement outcome is a
    Bernoulli variable with P(+1) = (1 + <P>)/2.
    """
    total = 0.0
    for label, c in h.terms.items():
        if set(label) == {"I"}:
            total += c
            continue
        ev = float(np.real(np.vdot(state, apply_pauli_string(label, state))))
        p = min(1.0, max(0.0, (1 + ev) / 2))
        k = rng.binomial(shots, p)
        total += c * (2 * k / shots - 1)
    return total


@dataclass
class VQEResult:
    energy: float
    params: np.ndarray
    history: list[float]
    nfev: int
    exact: float
    optimizer: str

    @property
    def error(self) -> float:
        return self.energy - self.exact


@dataclass
class VQE:
    """Minimise ``<psi(theta)|H|psi(theta)>`` over ansatz parameters."""

    hamiltonian: PauliHamiltonian
    ansatz: Ansatz
    shots: int | None = None
    seed: int = 0
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.hamiltonian.num_qubits != self.ansatz.num_qubits:
            raise ValueError("qubit count mismatch")
        self._rng = np.random.default_rng(self.seed)

    def energy(self, params: np.ndarray) -> float:
        state = self.ansatz.run(params)
        if self.shots is None:
            return self.hamiltonian.expectation(state)
        return shot_energy(self.hamiltonian, state, self.shots, self._rng)

    def exact_energy(self, params: np.ndarray) -> float:
        return self.hamiltonian.expectation(self.ansatz.run(params))

    def gradient(self, params: np.ndarray) -> np.ndarray:
        return parameter_shift_gradient(self.ansatz, self.hamiltonian, params)

    def initial_params(self, scale: float = 0.1, seed: int | None = None
                       ) -> np.ndarray:
        rng = np.random.default_rng(self.seed if seed is None else seed)
        return rng.normal(0.0, scale, self.ansatz.num_params)

    def run(self, optimizer: str = "adam", x0: np.ndarray | None = None,
            iters: int = 150, **kw) -> VQEResult:
        x0 = self.initial_params() if x0 is None else np.asarray(x0, float)
        res: OptResult
        if optimizer == "adam":
            res = adam(self.energy, self.gradient, x0, iters=iters,
                       lr=kw.get("lr", 0.1))
        elif optimizer == "spsa":
            res = spsa(self.energy, x0, iters=iters, seed=self.seed,
                       a=kw.get("a", 0.3), c=kw.get("c", 0.1))
        elif optimizer == "nelder-mead":
            res = nelder_mead(self.energy, x0, iters=iters)
        elif optimizer == "rotosolve":
            res = rotosolve(self.energy, x0, sweeps=iters)
        else:
            raise ValueError(f"unknown optimizer {optimizer!r}")
        return VQEResult(self.exact_energy(res.x), res.x, res.history,
                         res.nfev, self.hamiltonian.ground_energy(), optimizer)


def adapt_vqe(h: PauliHamiltonian, pool: list[str], reference: Ansatz,
              max_ops: int = 10, grad_tol: float = 1e-4, inner_iters: int = 100
              ) -> tuple[Ansatz, np.ndarray, list[float]]:
    """ADAPT-VQE: greedily append the pool rotation with the largest energy
    gradient, then re-optimise all parameters.

    ``reference`` supplies the (parameter-free) initial state circuit.
    Returns the final ansatz, its parameters and the energy after each step.
    """
    ans = Ansatz(reference.num_qubits, list(reference.ops), reference.num_params)
    params = np.zeros(ans.num_params)
    energies: list[float] = []
    s = np.pi / 2
    for _ in range(max_ops):
        best, best_g = None, 0.0
        for label in pool:
            trial = Ansatz(ans.num_qubits, list(ans.ops), ans.num_params)
            trial.rot(label, ans.num_params)
            x = np.append(params, 0.0)
            pos = len(trial.ops) - 1
            g = 0.5 * (h.expectation(trial.run(x, {pos: s}))
                       - h.expectation(trial.run(x, {pos: -s})))
            if abs(g) > abs(best_g):
                best, best_g = label, g
        if best is None or abs(best_g) < grad_tol:
            break
        ans.rot(best, ans.num_params)
        x0 = np.append(params, 0.0)
        res = adam(lambda p: ans.energy(p, h),
                   lambda p: parameter_shift_gradient(ans, h, p),
                   x0, iters=inner_iters, lr=0.2)
        params = res.x
        energies.append(ans.energy(params, h))
    return ans, params, energies
