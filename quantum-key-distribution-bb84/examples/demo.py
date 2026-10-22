"""Demo: BB84 sessions with and without an eavesdropper."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qkd_bb84 import (BB84Config, Channel, InterceptResend, run_bb84,
                      qber_threshold, sweep_eavesdropping,
                      detection_probability, expected_qber_intercept)


def show(name, r):
    print(f"{name:34s} sifted={r.n_sifted:5d} QBER={r.true_qber:6.3f} "
          f"aborted={str(r.aborted):5s} key={r.final_key_alice.size:4d} "
          f"match={r.keys_match if not r.aborted else '-'} leaked={r.leaked_bits}")


print(f"Asymptotic QBER threshold: {qber_threshold():.4f}")
cfg = BB84Config(n_photons=6000, seed=1)
show("ideal channel", run_bb84(cfg, Channel()))
show("10% loss, 0.10 rad misalignment", run_bb84(cfg, Channel(loss=0.1, misalignment=0.10)))
show("full intercept-resend", run_bb84(cfg, Channel(eve=InterceptResend(1.0))))
show("20% intercept-resend", run_bb84(cfg, Channel(eve=InterceptResend(0.2))))

print("\nEavesdropping sweep (4000 photons, 8 trials, 10% loss)")
print("frac  QBER  theory  abort  key_bits")
for row in sweep_eavesdropping([0.0, 0.1, 0.2, 0.3, 0.5, 1.0], 4000, 8, 3):
    print(f"{row['fraction']:.1f}  {row['qber']:.3f}  {row['expected']:.3f}  "
          f"{row['abort_rate']:.2f}   {row['key_bits']:.0f}")

print("\nDetection probability of full attack vs bits compared:")
for k in (1, 5, 10, 20, 50):
    print(f"  n={k:3d}: {detection_probability(1.0, k):.4f}")
