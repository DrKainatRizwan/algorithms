"""Demo: DPA / CPA / masking / second-order attacks on simulated AES traces."""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from power_analysis import (cpa_attack, dpa_attack_multibit, encrypt_block,
                            expand_key, invert_key_schedule, key_rank, recover_key,
                            second_order_cpa, simulate_masked_traces,
                            simulate_traces, success_curve, traces_to_success)

KEY = bytes.fromhex("2b7e151628aed2a6abf7158809cf4f3c")
t0 = time.time()
print("AES-128 check:", encrypt_block(KEY, bytes.fromhex("3243f6a8885a308d313198a2e0370734")).hex())
print("Round-10 key inverts to master key:", invert_key_schedule(bytes(expand_key(KEY)[10])) == KEY)

ts = simulate_traces(KEY, 300, noise_sigma=1.0, seed=1)
rec = recover_key(ts, cpa_attack)
print("\nCPA, 300 traces, sigma=1: recovered", rec.hex(), "correct:", rec == KEY)

print("\nSuccess rate vs traces (byte 0, 30 experiments)")
for name, atk in [("CPA", cpa_attack),
                  ("DPA(8-bit sum)", dpa_attack_multibit)]:
    for sigma in (1.0, 3.0):
        c = success_curve(lambda n, s: simulate_traces(KEY, n, sigma, s), atk, 0,
                          [10, 25, 50, 100, 200, 400, 800, 1600], 30)
        print(f"  {name:15s} sigma={sigma}: SR=", [round(float(x), 2) for x in c["success_rate"]],
              "-> 90% at", traces_to_success(c))

ms = simulate_masked_traces(KEY, 5000, noise_sigma=0.5, seed=2)
print("\nMasked implementation, 5000 traces, sigma=0.5")
print("  1st-order CPA rank of true key byte:", key_rank(cpa_attack(ms, 0), KEY[0]))
print("  2nd-order CPA rank of true key byte:", key_rank(second_order_cpa(ms, 0), KEY[0]))
rec2 = recover_key(ms, second_order_cpa)
print("  2nd-order full key correct bytes:", sum(a == b for a, b in zip(rec2, KEY)), "/ 16")
c = success_curve(lambda n, s: simulate_masked_traces(KEY, n, 0.5, s), second_order_cpa, 0,
                  [100, 250, 500, 1000, 2000, 4000], 20)
print("  2nd-order SR:", [round(float(x), 2) for x in c["success_rate"]], "-> 90% at", traces_to_success(c))
print(f"\nElapsed {time.time() - t0:.1f}s")
