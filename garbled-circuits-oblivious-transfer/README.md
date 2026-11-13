# Garbled Circuits and Oblivious Transfer: Yao's Two-Party Computation from Scratch

## Overview
A from-scratch implementation of Yao's protocol for semi-honest secure two-party computation. Two parties, Alice and Bob, compute a function of their private inputs and learn only the output. The package contains a Boolean circuit builder with arithmetic gadgets, three garbling schemes (classic point-and-permute, free-XOR and half-gates), a Chou–Orlandi 1-out-of-2 oblivious transfer, and an end-to-end protocol with communication accounting. Only the Python standard library is used.

## How it works
1. **Circuit.** The function is expressed over XOR, AND and NOT gates. Builders are provided for ripple-carry addition, comparison (the millionaires' problem), equality, multiplication and max.
2. **Garbling.** Alice assigns two random 128-bit labels to every wire (one per bit value) and encrypts each gate's output labels under the input labels via a SHA-256 based random oracle. Colour (lsb) bits let Bob pick the single correct table row.
   - *classic*: 4 ciphertexts for every XOR/AND gate.
   - *free-XOR*: a global offset `R` makes XOR gates free (`W0 = A0 ^ B0`); AND still costs 4 rows.
   - *half-gates*: an AND is split into a garbler-half and an evaluator-half costing 2 ciphertexts in total.
   - NOT is free in all schemes (labels are relabelled).
3. **Oblivious transfer.** Bob obtains the label for each of his input bits using Chou–Orlandi OT over Z_p*, p = 2^255 − 19: the sender publishes `A = g^a`, the receiver replies `B = g^b` or `A·g^b` depending on the choice, and the sender encrypts the two labels under `H(B^a)` and `H((B/A)^a)`. Bob can derive only `H(A^b)`, matching his choice.
4. **Evaluation.** Bob evaluates gate by gate and decodes the outputs with Alice's output decoding bits.

## Complexity
| Quantity | Cost |
|---|---|
| Garbling / evaluation time | O(g) hash calls for g gates |
| Garbled-circuit size (half-gates) | 2·n_AND ciphertexts of 16 bytes |
| Garbled-circuit size (free-XOR) | 4·n_AND ciphertexts |
| Garbled-circuit size (classic) | 4·(n_AND + n_XOR) ciphertexts |
| Oblivious transfers | one per Bob input bit, O(n_B) modular exponentiations |
| Space | O(wires) labels |

Gadget AND-counts: adder n, comparator n, n×n multiplier n² + n·(2n−i) summed over adder rows (768 for 16×16 as measured).

## Usage
```python
from garbled_circuits import run_two_party
from garbled_circuits.circuit import millionaires
from garbled_circuits.primitives import int_to_bits

c = millionaires(16)
res = run_two_party(c, int_to_bits(48213, 16), int_to_bits(37990, 16), scheme="half_gates", seed=1)
print(res.outputs, res.total_bytes)   # [1], bytes exchanged
```

## Results
Output of `python examples/demo.py` (single run, timings include 32 or 16 base OTs):

| Circuit | Scheme | ANDs | XORs | Ciphertexts | Garbled bytes | Total bytes |
|---|---|---|---|---|---|---|
| 32-bit adder | classic | 32 | 161 | 772 | 12869 | 14949 |
| 32-bit adder | free-XOR | 32 | 161 | 128 | 2565 | 4645 |
| 32-bit adder | half-gates | 32 | 161 | 64 | 1541 | 3621 |
| 16×16 multiplier | classic | 768 | 2561 | 13316 | 213316 | 214372 |
| 16×16 multiplier | free-XOR | 768 | 2561 | 3072 | 49412 | 50468 |
| 16×16 multiplier | half-gates | 768 | 2561 | 1536 | 24836 | 25892 |

Half-gates shrink the multiplier's garbled circuit by about 8.6× versus the classic scheme.

Run tests with `python -m pytest -q` (26 tests, exhaustive on small widths).

## References
- A. C. Yao, "How to Generate and Exchange Secrets", FOCS 1986.
- M. Beaver, S. Micali, P. Rogaway, "The Round Complexity of Secure Protocols", STOC 1990 (point-and-permute).
- V. Kolesnikov, T. Schneider, "Improved Garbled Circuit: Free XOR Gates and Applications", ICALP 2008.
- S. Zahur, M. Rosulek, D. Evans, "Two Halves Make a Whole: Reducing Data Transfer in Garbled Circuits using Half Gates", EUROCRYPT 2015.
- T. Chou, C. Orlandi, "The Simplest Protocol for Oblivious Transfer", LATINCRYPT 2015.

*Educational code: labels come from a seeded PRNG and only semi-honest security is modelled.*

---

**Author:** Dr. Kainat Rizwan
