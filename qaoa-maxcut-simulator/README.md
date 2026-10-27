# QAOA Simulator for Max-Cut

## Overview
A from-scratch NumPy state-vector simulator of the Quantum Approximate Optimization Algorithm (QAOA) applied to Max-Cut. It provides exact expectation values, an adjoint-differentiation gradient (cost of O(p) state passes regardless of parameter count), Adam and Nelder-Mead optimisers, INTERP warm starts across depths, shot sampling, the closed-form p=1 expectation for unweighted graphs, and classical baselines (greedy, 1-flip local search, brute force).

## How it works
For a graph with cut operator `C = sum_{(u,v)} w_uv (1 - Z_u Z_v)/2` (diagonal in the computational basis) and mixer `B = sum_j X_j`, the depth-p ansatz is

    |gamma, beta> = prod_{k=1..p} exp(-i beta_k B) exp(-i gamma_k C) |+>^n

and the objective is `F(gamma, beta) = <gamma, beta| C |gamma, beta>`, maximised classically.

* **Phase layer:** elementwise multiplication by `exp(-i gamma C(x))`, with the 2^n cut values precomputed once.
* **Mixer layer:** a single-qubit RX rotation on each qubit, applied axis by axis on a `(2,)*n` tensor view.
* **Gradient:** adjoint differentiation. Keep the final state `phi` and `lambda = C phi`; step backwards through the layers, reading `2 Re <lambda| -iG |phi>` for each generator `G` and un-applying the layer from both vectors.
* **p=1 closed form:** for edge (u,v) with d = deg(u)-1, e = deg(v)-1 and f common neighbours,
  `<C_uv> = 1/2 + 1/4 sin4b sin g (cos^d g + cos^e g) - 1/4 sin^2 2b cos^(d+e-2f) g (1 - cos^f 2g)`. This is used to cross-check the simulator.
* **INTERP:** the depth-p optimum is linearly interpolated to a depth-(p+1) initial guess, which avoids most poor local optima.

## Complexity
| Operation | Time | Space |
|-----------|------|-------|
| One energy evaluation | O(p (2^n n + 2^n)) | O(2^n) |
| Adjoint gradient | O(p n 2^n) (about 3x one evaluation per layer) | O(2^n) |
| Finite-difference gradient | O(p^2 n 2^n) | O(2^n) |
| Cut table | O(m 2^n) | O(2^n) |

## Usage
```python
import numpy as np
from qaoa_maxcut import *

g = random_regular_graph(10, 3, seed=11)
q = QAOA(g, p=2)
res = multistart(q.energy, q.gradient, dim=4, starts=6, seed=0)
print(q.approximation_ratio(res.x), q.success_probability(res.x))
```
Run the demo with `python examples/demo.py` and the tests with `python -m pytest -q`.

## Results
Demo output on a random 3-regular graph (n=10, m=15, optimum cut 13), Adam with INTERP warm starts:

| p | Expected cut | Approx. ratio | P(optimal cut) |
|---|-------------|---------------|----------------|
| 1 | 10.3868 | 0.7990 | 0.1320 |
| 2 | 11.4792 | 0.8830 | 0.3021 |
| 3 | 12.1143 | 0.9319 | 0.5127 |
| 4 | 12.4929 | 0.9610 | 0.6926 |
| 5 | 12.7094 | 0.9776 | 0.8136 |

On the 12-cycle the optimised p=1 ratio is 0.7500, matching the known 3/4 limit for rings. The p=1 grid maximum (10.3853) agrees with the closed form to four decimals. Greedy and local search both find the optimum (13) on this small instance, so QAOA's value here is in the scaling behaviour with depth rather than beating classical heuristics.

## References
* E. Farhi, J. Goldstone, S. Gutmann, "A Quantum Approximate Optimization Algorithm", arXiv:1411.4028 (2014).
* Z. Wang, S. Hadfield, Z. Jiang, E. Rieffel, "Quantum approximate optimization algorithm for MaxCut: A fermionic view", Phys. Rev. A 97, 022304 (2018).
* L. Zhou, S.-T. Wang, S. Choi, H. Pichler, M. Lukin, "Quantum Approximate Optimization Algorithm: Performance, Mechanism, and Implementation on Near-Term Devices", Phys. Rev. X 10, 021067 (2020).
* T. Jones, J. Gacon, "Efficient calculation of gradients in classical simulations of variational quantum algorithms", arXiv:2009.02823 (2020).

---

**Author:** Dr. Kainat Rizwan
