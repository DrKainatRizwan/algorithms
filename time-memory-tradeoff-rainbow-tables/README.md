# Time-Memory Trade-Off Cryptanalysis: Hellman, Distinguished-Point and Rainbow Tables

## Overview

Inverting a one-way function `f` on a space of size `N` can be done by brute force (time `N`, no memory) or by a full lookup table (memory `N`, instant). Time-memory trade-offs (TMTO) sit in between: an expensive offline phase condenses the function into chains of which only the endpoints are stored, and the online phase regenerates chains on demand. This project implements three classic schemes from scratch on a random-function model (`tmto/`):

| Module | Contents |
|---|---|
| `oneway.py` | Counted random-looking function on `bits`-bit integers, plus an integer/password bijection |
| `hellman.py` | Hellman tables with per-table step variants `F_i(x) = f(x) xor i` |
| `distinguished.py` | Rivest's distinguished-point chains with loop/over-length rejection |
| `rainbow.py` | Oechslin rainbow tables with column-dependent reductions |
| `analysis.py` | Hellman and rainbow coverage formulas, empirical success measurement |

## How it works

**Hellman.** A table has `m` chains of length `t`; only `(start, end)` is stored. To invert `y`, compute `y xor i`, iterate `F_i` up to `t` times and compare against the endpoints; on a match, rebuild the chain from its start and look for `x` with `f(x) = y`. Chains from different starts may merge, so `l` tables with different `i` are used; the matrix stopping rule `m t^2 = N` gives `T M^2 = N^2` trade-off.  An endpoint match without a true preimage is a *false alarm*.

**Distinguished points.** Chains stop at the first point with `d` low zero bits (mean length `2^d`). Online work needs one dictionary lookup per walk instead of one per step, and looping chains are discarded at build time.

**Rainbow tables.** Step `k` uses its own reduction `R_k`, so chains that collide in different columns never merge. One table of `m` chains of length `t` behaves like about `t` classic tables, at a lookup cost of up to `t(t-1)/2` evaluations. The success probability is `1 - prod_k (1 - m_k/N)` with `m_{k+1} = N(1 - exp(-m_k/N))`.

## Complexity

| Scheme | Precompute | Memory | Online (per query) |
|---|---|---|---|
| Hellman (l tables) | `l m t` | `l m` | `l t` evaluations (+ false alarms) |
| Distinguished points | `l m 2^d` | `l m` | `l 2^d` evaluations, `l` lookups |
| Rainbow (one table) | `m t` | `m` | `O(t^2)` evaluations, `t` lookups |

## Usage

```python
from tmto import OneWayFunction, RainbowAttack, measure_success

f = OneWayFunction(18)
attack = RainbowAttack(f, m=6000, t=100, tables=1, seed=1)
attack.precompute()
y = f(123456)
x = attack.invert(y)           # a preimage, or None
print(x is not None and f(x) == y)
print(measure_success(attack, f, trials=200).success_rate)
```

Run `python examples/demo.py` for the benchmark and `python -m pytest -q` for the tests.

## Results

Space `N = 2^18`, 200 random challenges (output of `examples/demo.py`; "theory" is Hellman's lower bound or Oechslin's formula):

| Attack | Memory (pairs) | Precompute f() | Success | Theory | Mean online f() |
|---|---|---|---|---|---|
| Hellman, m=256, t=32, l=8 | 2048 | 65,536 | 0.390 | 0.181 | 314 |
| Hellman, l=24 | 6144 | 196,608 | 0.705 | 0.450 | 641 |
| Distinguished points, d=5, l=24 | 6069 | 206,917 | 0.650 | - | 1424 |
| Rainbow, m=6000, t=100 | 2738 | 600,000 | 0.850 | 0.786 | 2034 |
| Rainbow, two tables | 5544 | 1,200,000 | 0.925 | - | 2653 |

Rainbow tables reach higher coverage with less stored data (merged chains are dropped), while Hellman's bound is conservative, as expected for a lower bound. The demo ends by recovering a preimage of a 16-bit digest of a short password.

## References

- M. Hellman, "A Cryptanalytic Time-Memory Trade-Off", IEEE Trans. Information Theory, 1980.
- D. Denning, *Cryptography and Data Security*, 1982 (distinguished points, attributed to R. Rivest).
- P. Oechslin, "Making a Faster Cryptanalytic Time-Memory Trade-Off", CRYPTO 2003.
- A. Biryukov, A. Shamir, "Cryptanalytic Time/Memory/Data Tradeoffs for Stream Ciphers", ASIACRYPT 2000.

---

**Author:** Dr. Kainat Rizwan
