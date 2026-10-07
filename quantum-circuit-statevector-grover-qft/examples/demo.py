"""Demo: Bell/GHZ states, QFT check, Grover scaling and phase estimation."""
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from qsim_core import (Circuit, dft_matrix, estimate_phase, ghz_circuit, grover_circuit,
                       grover_success_curve, optimal_grover_iterations, qft_circuit)

rng = np.random.default_rng(7)
print("GHZ(5) samples:", ghz_circuit(5).run().sample(1000, rng))

print("\nQFT vs DFT matrix (max abs error):")
for n in (2, 4, 6, 8):
    err = np.abs(qft_circuit(n).unitary() - dft_matrix(n)).max()
    c = qft_circuit(n)
    print(f"  n={n}: err={err:.2e} depth={c.depth()} gates={len(c)}")

print("\nGrover search (single marked item):")
print(f"{'n':>3} {'N':>6} {'iters':>6} {'P(success)':>11} {'time(s)':>8}")
for n in (4, 6, 8, 10, 12):
    target = (5 * n + 3) % 2**n
    t0 = time.perf_counter()
    c = grover_circuit(n, [target])
    p = c.run().probabilities()[target]
    print(f"{n:>3} {2**n:>6} {optimal_grover_iterations(n, 1):>6} {p:>11.4f} {time.perf_counter()-t0:>8.3f}")

curve = grover_success_curve(6, [42], 12)
print("\nSuccess probability vs iterations (n=6):")
print("  " + " ".join(f"{x:.2f}" for x in curve))

print("\nPhase estimation (true phase 1/3):")
for t in (3, 5, 7, 9):
    est, p = estimate_phase(t, 1 / 3)
    print(f"  t={t}: estimate={est:.5f} error={abs(est-1/3):.5f} P={p:.3f}")
