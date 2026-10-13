"""Demo: H2 ground state, optimizer comparison, spin chains and ADAPT-VQE."""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vqe_simulator import (VQE, adapt_vqe, h2_uccsd, h2_hamiltonian,
                           hardware_efficient, heisenberg_chain,
                           transverse_field_ising)
from vqe_simulator.hamiltonians import H2_NUCLEAR_REPULSION


def main() -> None:
    t0 = time.time()
    h = h2_hamiltonian()
    exact = h.ground_energy()
    print(f"H2 exact electronic energy : {exact:.6f} Ha "
          f"(total {exact + H2_NUCLEAR_REPULSION:.6f} Ha)")

    print("\n[1] H2 with one-parameter UCC ansatz")
    r = VQE(h, h2_uccsd()).run("adam", x0=np.array([0.1]), iters=100)
    print(f"  adam      E={r.energy:.6f}  error={r.error:.2e}  theta={r.params[0]:.4f}")

    print("\n[2] Optimizer comparison, H2 with hardware-efficient ansatz (2 layers)")
    v = VQE(h, hardware_efficient(2, 2))
    x0 = v.initial_params(0.5)
    for name, it in [("adam", 150), ("nelder-mead", 400), ("rotosolve", 6)]:
        r = v.run(name, x0=x0, iters=it)
        print(f"  {name:12s} error={r.error:.2e}  evaluations={r.nfev}")
    vs = VQE(h, hardware_efficient(2, 2), shots=4000, seed=1)
    r = vs.run("spsa", x0=x0, iters=250)
    print(f"  spsa (4000 shots/term) error={r.error:.2e}  evaluations={r.nfev}")

    print("\n[3] Spin chains, 4 qubits, hardware-efficient ansatz (3 layers)")
    for name, ham in [("TFIM h=1.0", transverse_field_ising(4, 1.0, 1.0)),
                      ("Heisenberg", heisenberg_chain(4))]:
        v = VQE(ham, hardware_efficient(4, 3))
        r = v.run("adam", iters=300)
        print(f"  {name:11s} exact={r.exact:.5f}  vqe={r.energy:.5f}  "
              f"error={r.error:.2e}")

    print("\n[4] ADAPT-VQE on 4-site TFIM with a Pauli-rotation pool")
    ham = transverse_field_ising(4, 1.0, 0.8)
    ref = hardware_efficient(4, 0)
    ref.ops, ref.num_params = [], 0
    for q in range(4):
        ref.fixed_rot("".join("Y" if k == q else "I" for k in range(4)), np.pi / 2)
    pool = ["YIII", "IYII", "IIYI", "IIIY", "YZII", "IYZI", "IIYZ", "ZYII",
            "IZYI", "IIZY", "XYII", "IXYI", "IIXY", "YXII", "IYXI", "IIYX"]
    ans, x, es = adapt_vqe(ham, pool, ref, max_ops=6, inner_iters=80)
    for i, e in enumerate(es, 1):
        print(f"  step {i:2d}: E={e:.5f}  error={e - ham.ground_energy():.2e}")
    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
