"""Demo: break a 4-round SPN with differential and linear cryptanalysis."""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from spn_cryptanalysis import (SBOX_AFFINE, SBOX_HEYS, SBOX_PRESENT, SPN,
                               best_characteristic, best_trail,
                               differential_attack, linear_attack, sbox_report)
from spn_cryptanalysis import analysis, differential, linear
from spn_cryptanalysis.spn import nibble

t0 = time.time()
print("== S-box quality ==")
print(f"{'sbox':8s} {'DU':>3s} {'Lin':>4s} {'NL':>3s} {'best3r diff':>12s} {'best3r bias':>12s}")
for name, sb in (("heys", SBOX_HEYS), ("present", SBOX_PRESENT), ("affine", SBOX_AFFINE)):
    r = sbox_report(name, sb)
    print(f"{r.name:8s} {r.diff_uniformity:3d} {r.linearity:4d} {r.nonlinearity:3d} "
          f"{r.best_diff_char:12.5f} {r.best_linear_bias:12.5f}")

rng = np.random.default_rng(2024)
cipher = SPN.random(4, rng)
print("\nround keys:", [f"{k:04x}" for k in cipher.keys])

print("\n== Differential attack ==")
ch = best_characteristic(SBOX_HEYS, 3)
print(f"best 3-round characteristic: {ch.input_diff:04x} -> {ch.output_diff:04x}, "
      f"p = {ch.probability:.5f} ({ch.probability * 1024:.0f}/1024), "
      f"{ch.active_sboxes} active S-boxes")
print(f"empirical p under random keys: "
      f"{differential.empirical_probability(cipher, ch, 100000, rng):.5f}")
res = differential_attack(cipher, ch, 400, rng)
truth = differential.true_subkey(cipher, res.nibble_positions)
print(f"nibbles {res.nibble_positions}: guess {res.best_guess}, truth {truth}, "
      f"{res.pairs_filtered}/{res.pairs_used} pairs survive filtering")
for n in (25, 50, 100, 200, 400):
    sr = differential.success_rate(SBOX_HEYS, 4, ch, n, 40, seed=1)
    print(f"  pairs={n:4d}  success rate {sr:.2f}")

print("\n== Linear attack ==")
tr = best_trail(SBOX_HEYS, 3)
print(f"best 3-round trail: alpha={tr.input_mask:04x}, beta={tr.output_mask:04x}, "
      f"bias = {tr.bias:.5f}, rule-of-thumb N ~ {linear.required_samples(tr.bias)}")
print(f"empirical bias: {linear.empirical_bias(cipher, tr, 200000, rng):.5f}")
res = linear_attack(cipher, tr, 20000, rng)
truth = linear.true_subkey(cipher, res.nibble_positions)
print(f"nibbles {res.nibble_positions}: guess {res.best_guess}, truth {truth}")
for n in (500, 1000, 2000, 4000, 8000, 16000):
    sr = linear.success_rate(SBOX_HEYS, 4, tr, n, 40, seed=1)
    print(f"  known plaintexts={n:6d}  success rate {sr:.2f}")

print("\n== Round count vs best characteristic (PRESENT S-box) ==")
for r in range(1, 5):
    c = best_characteristic(SBOX_PRESENT, r)
    print(f"  {r} rounds: p = 2^{np.log2(c.probability):.1f}")
print(f"\ntotal time {time.time() - t0:.1f}s")
