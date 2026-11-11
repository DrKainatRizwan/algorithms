"""Demo: order finding, phase estimation and factoring with simulated Shor."""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shor_sim import (order_recovery_probability, outcome_distribution, peak_mass,
                      resource_estimate, shor_factor, split_probability)
from shor_sim.modular import multiplicative_order
from shor_sim.qft import qft_gates, qft_matrix


def main() -> None:
    print("== QFT: gate-level circuit vs dense matrix (8 qubits) ==")
    rng = np.random.default_rng(0)
    v = rng.normal(size=256) + 1j * rng.normal(size=256)
    v /= np.linalg.norm(v)
    print("max deviation:", f"{np.abs(qft_gates(v, 8) - qft_matrix(8) @ v).max():.2e}")

    print("\n== Order finding for a=7, N=15 (t=8, Q=256) ==")
    d = outcome_distribution(7, 15, 8)
    for y in np.nonzero(d > 1e-9)[0]:
        print(f"  outcome {y:3d}  p={d[y]:.3f}  y/Q={y/256:.3f}")
    print("order:", multiplicative_order(7, 15),
          " P(recover r)=", f"{order_recovery_probability(7, 15):.3f}")

    print("\n== Success statistics (t = ceil(2 log2 N)) ==")
    print(f"{'a':>3} {'N':>4} {'r':>4} {'P(recover r)':>13} {'peak mass':>10}")
    for a, n in [(2, 21), (2, 35), (3, 55), (2, 77), (5, 91)]:
        t = int(np.ceil(2 * np.log2(n)))
        print(f"{a:>3} {n:>4} {multiplicative_order(a, n):>4} "
              f"{order_recovery_probability(a, n):>13.3f} {peak_mass(a, n, t):>10.3f}")

    print("\n== Factoring semiprimes (seeded, compressed back-end) ==")
    print(f"{'N':>6} {'factors':>12} {'quantum runs':>13} {'P(good base)':>13} {'time s':>8}")
    for n in [15, 21, 35, 77, 143, 221, 391, 667]:
        t0 = time.perf_counter()
        res = shor_factor(n, np.random.default_rng(n))
        dt = time.perf_counter() - t0
        print(f"{n:>6} {str(res.factors):>12} {res.quantum_runs:>13} "
              f"{split_probability(n):>13.3f} {dt:>8.3f}")

    print("\n== Textbook resource estimates ==")
    for n in (15, 221, 2**64 + 13):
        print(n if n < 10**6 else "2^64+13", resource_estimate(n))


if __name__ == "__main__":
    main()
