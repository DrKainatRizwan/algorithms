"""Model Hamiltonians: a reduced H2 molecule and spin chains."""
from __future__ import annotations

from .pauli import PauliHamiltonian


def _label(n: int, ops: dict[int, str]) -> str:
    return "".join(ops.get(k, "I") for k in range(n))


def h2_hamiltonian() -> PauliHamiltonian:
    """Two-qubit H2 Hamiltonian (STO-3G, parity mapping with two-qubit
    reduction) at bond length 0.7414 Angstrom.

    Energies are electronic only; add :data:`H2_NUCLEAR_REPULSION` for the
    total energy (about -1.1373 Hartree).
    """
    return PauliHamiltonian({
        "II": -1.052373245772859,
        "ZI": 0.39793742484318045,
        "IZ": -0.39793742484318045,
        "ZZ": -0.01128010425623538,
        "XX": 0.18093119978423156,
    })


H2_NUCLEAR_REPULSION = 0.7199689944489797


def transverse_field_ising(n: int, j: float = 1.0, h: float = 1.0,
                           periodic: bool = False) -> PauliHamiltonian:
    """H = -J sum Z_i Z_{i+1} - h sum X_i."""
    terms: dict[str, float] = {}
    bonds = n if periodic and n > 2 else n - 1
    for i in range(bonds):
        terms[_label(n, {i: "Z", (i + 1) % n: "Z"})] = -j
    for i in range(n):
        terms[_label(n, {i: "X"})] = -h
    return PauliHamiltonian(terms)


def heisenberg_chain(n: int, j: float = 1.0, periodic: bool = False
                     ) -> PauliHamiltonian:
    """Antiferromagnetic Heisenberg chain H = J sum (XX + YY + ZZ)."""
    terms: dict[str, float] = {}
    bonds = n if periodic and n > 2 else n - 1
    for i in range(bonds):
        for p in "XYZ":
            terms[_label(n, {i: p, (i + 1) % n: p})] = j
    return PauliHamiltonian(terms)
