"""Demo: Yao's protocol with three garbling schemes."""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from garbled_circuits import SCHEMES, garble, run_two_party
from garbled_circuits.circuit import adder_circuit, max_circuit, millionaires, multiplier_circuit
from garbled_circuits.primitives import bits_to_int, int_to_bits

alice, bob = 48_213, 37_990
c = millionaires(16)
r = run_two_party(c, int_to_bits(alice, 16), int_to_bits(bob, 16), seed=1)
print(f"Millionaires: Alice={alice} Bob={bob} -> Alice richer? {bool(r.outputs[0])}")

m = max_circuit(16)
r = run_two_party(m, int_to_bits(alice, 16), int_to_bits(bob, 16), seed=2)
print(f"Secure max: {bits_to_int(r.outputs)}")

print("\nCommunication per scheme (32-bit adder / 16x16 multiplier):")
print(f"{'circuit':<12}{'scheme':<12}{'ANDs':>6}{'XORs':>7}{'ciphertexts':>13}{'garbled B':>11}{'total B':>9}{'ms':>8}")
for name, circ, w in (("add32", adder_circuit(32), 32), ("mul16", multiplier_circuit(16), 16)):
    for s in SCHEMES:
        t = time.perf_counter()
        res = run_two_party(circ, int_to_bits(123456 % (1 << w), w), int_to_bits(98765 % (1 << w), w), s, seed=3)
        ms = (time.perf_counter() - t) * 1000
        gc = garble(circ, s, 3)
        print(f"{name:<12}{s:<12}{circ.count('AND'):>6}{circ.count('XOR'):>7}{gc.n_ciphertexts:>13}{res.garbled_bytes:>11}{res.total_bytes:>9}{ms:>8.0f}")
        expected = ((123456 % (1 << w)) + (98765 % (1 << w))) if name == "add32" else (123456 % (1 << w)) * (98765 % (1 << w))
        assert bits_to_int(res.outputs) == expected
print("\nAll protocol outputs match plaintext evaluation.")
