# Variational Quantum Eigensolver Simulator for Small Molecular Hamiltonians

A from-scratch, NumPy-only state-vector implementation of the variational quantum eigensolver (VQE). It includes Pauli-sum Hamiltonians, a reduced two-qubit H2 molecule, spin-chain models, parameterised circuits, exact parameter-shift gradients, four classical optimisers, a shot-noise energy estimator and ADAPT-VQE.

## How it works

VQE minimises the Rayleigh quotient `E(θ) = <ψ(θ)|H|ψ(θ)>` over a parameterised circuit `|ψ(θ)> = U(θ)|0…0>`. By the variational principle `E(θ) ≥ E₀`, so any optimiser that lowers `E` approaches the ground-state energy.

* **Pauli algebra** (`pauli.py`): a Pauli string acts on a basis state as `P|j> = i^{n_Y} (-1)^{popcount(j & z)} |j ⊕ x>`, so `P|ψ>` is a signed permutation computed in O(2ⁿ) without building matrices. `H = Σ cₖ Pₖ` is applied term by term.
* **Hamiltonians** (`hamiltonians.py`): the parity-mapped, two-qubit-reduced H2 Hamiltonian (STO-3G, 0.7414 Å), the transverse-field Ising chain and the Heisenberg chain.
* **Ansätze** (`ansatz.py`): every parametrised gate is `exp(-iθP/2)`, covering RX/RY/RZ and multi-qubit Pauli rotations. Provided circuits are a hardware-efficient RY+CNOT ansatz, a one-parameter UCC-style ansatz for H2 and a Hamiltonian-variational ansatz with shared angles.
* **Gradients** (`gradients.py`): the parameter-shift rule `∂E/∂θ = [E(θ+π/2) − E(θ−π/2)]/2` is exact for these gates. Shared parameters sum per-gate contributions. A central finite-difference routine is included as a cross-check.
* **Optimisers** (`optimizers.py`): gradient descent, Adam, SPSA (two evaluations per step, tolerant of shot noise), Nelder–Mead and Rotosolve. Rotosolve exploits that the energy is a sinusoid in each angle and jumps to its analytic minimum from three evaluations.
* **Driver** (`vqe.py`): the `VQE` class, `shot_energy` (binomial sampling of each Pauli term with P(+1) = (1+<P>)/2) and `adapt_vqe`. ADAPT-VQE grows the circuit by repeatedly appending the pool rotation with the largest gradient and re-optimising.

## Complexity

| Operation | Time | Space |
|-----------|------|-------|
| Apply one Pauli string | O(2ⁿ) | O(2ⁿ) |
| Energy, `T` Pauli terms | O(T·2ⁿ) | O(2ⁿ) |
| Circuit with `G` gates | O(G·2ⁿ) | O(2ⁿ) |
| Parameter-shift gradient, `P` gate parameters | O(2P·(G + T)·2ⁿ) | O(2ⁿ) |
| Dense reference diagonalisation | O(8ⁿ) | O(4ⁿ) |
| ADAPT step with pool size `M` | O(2M·(G + T)·2ⁿ) plus inner optimisation | O(2ⁿ) |

## Usage

```python
import numpy as np
from vqe_simulator import VQE, h2_hamiltonian, h2_uccsd

h = h2_hamiltonian()
result = VQE(h, h2_uccsd()).run("adam", x0=np.array([0.1]), iters=100)
print(result.energy, result.exact, result.error)
```

Run the tests with `python -m pytest -q` and the demo with `python examples/demo.py`.

## Results

Output of `examples/demo.py` (all numbers in Hartree for H2, in units of J for the spin chains):

| System | Method | Final error |
|--------|--------|-------------|
| H2 (exact −1.857275 electronic, −1.137306 total) | UCC ansatz, Adam, 1 parameter | 9.3e-08 |
| H2 | Hardware-efficient (2 layers), Adam | 8.9e-08 |
| H2 | Hardware-efficient, Nelder–Mead (229 evaluations) | 4.9e-11 |
| H2 | Hardware-efficient, Rotosolve (114 evaluations) | ~1e-15 |
| H2 | Hardware-efficient, SPSA with 4000 shots per term | 2.5e-02 |
| 4-site TFIM, h = 1 | Hardware-efficient (3 layers), Adam | 1.3e-03 |
| 4-site Heisenberg | Hardware-efficient (3 layers), Adam | 5.5e-08 |
| 4-site TFIM, h = 0.8 | ADAPT-VQE, 5 operators | 4.1e-04 |

The SPSA error is dominated by shot noise: the evaluations it sees are noisy, and the demo reports the final noisy-optimised parameters.
The 22 tests in `tests/` check the Pauli algebra against dense matrices, parameter-shift gradients against finite differences, the variational bound and convergence of each optimiser.

## References

* A. Peruzzo et al., "A variational eigenvalue solver on a photonic quantum processor", *Nature Communications* 5, 4213 (2014).
* P. J. J. O'Malley et al., "Scalable quantum simulation of molecular energies", *Physical Review X* 6, 031007 (2016).
* K. Mitarai, M. Negoro, M. Kitagawa, K. Fujii, "Quantum circuit learning", *Physical Review A* 98, 032309 (2018).
* M. Ostaszewski, E. Grant, M. Benedetti, "Structure optimization for parameterized quantum circuits", *Quantum* 5, 391 (2021).
* H. R. Grimsley et al., "An adaptive variational algorithm for exact molecular simulations on a quantum computer", *Nature Communications* 10, 3007 (2019).
* J. C. Spall, "Multivariate stochastic approximation using a simultaneous perturbation gradient approximation", *IEEE Transactions on Automatic Control* 37, 332 (1992).

---

**Author:** Dr. Kainat Rizwan
