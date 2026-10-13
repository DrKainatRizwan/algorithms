"""Variational quantum eigensolver simulator (state-vector, NumPy only)."""

from .pauli import PauliHamiltonian, apply_pauli_string, pauli_matrix
from .hamiltonians import h2_hamiltonian, heisenberg_chain, transverse_field_ising
from .ansatz import Ansatz, hardware_efficient, h2_uccsd, hamiltonian_variational
from .gradients import parameter_shift_gradient, finite_difference_gradient
from .optimizers import (OptResult, gradient_descent, adam, spsa, nelder_mead,
                         rotosolve)
from .vqe import VQE, VQEResult, adapt_vqe, shot_energy

__all__ = [
    "PauliHamiltonian", "apply_pauli_string", "pauli_matrix",
    "h2_hamiltonian", "heisenberg_chain", "transverse_field_ising",
    "Ansatz", "hardware_efficient", "h2_uccsd", "hamiltonian_variational",
    "parameter_shift_gradient", "finite_difference_gradient",
    "OptResult", "gradient_descent", "adam", "spsa", "nelder_mead", "rotosolve",
    "VQE", "VQEResult", "adapt_vqe", "shot_energy",
]
