# Robust Federated Learning with Secure Aggregation and Client Dropout Handling

## Overview

A from-scratch implementation of the pairwise-masking secure aggregation
protocol of Bonawitz et al. (2017), including recovery from clients that drop
out at any stage, wired into a federated logistic-regression trainer. The server
learns only the *sum* of the surviving clients' updates, never an individual
update. A bucketed variant adds Byzantine robustness on top of the secure sums.

Pure Python 3.12 + NumPy; no cryptography libraries.

| Module | Contents |
|--------|----------|
| `field.py` | Prime field `GF(2^255-19)`, Shamir sharing, Lagrange reconstruction |
| `crypto.py` | Diffie-Hellman, SHA-256 KDF, encrypt-then-MAC stream cipher, seeded PRG |
| `quantize.py` | L2 clipping, fixed-point encoding into `Z_{2^32}`, ring arithmetic |
| `protocol.py` | Client state machine and server for the four rounds, abort rules |
| `federated.py` | Synthetic non-IID data, local SGD, plain / secure / bucketed aggregators |

## How it works

1. **Advertise.** Client `u` creates two DH key pairs: `c` (share encryption)
   and `s` (mask agreement), plus a random self-mask seed `b_u`.
2. **Share.** `u` Shamir-shares `s_sk` and `b_u` with threshold `t` among all
   advertised clients, each share encrypted and authenticated under the pairwise
   `c`-key. The server only relays ciphertexts.
3. **Masked input.** `u` sends
   `y_u = x_u + PRG(b_u) + sum_{v>u} PRG(s_uv) - sum_{v<u} PRG(s_uv) (mod 2^32)`.
   Pairwise masks cancel between two surviving clients.
4. **Unmask.** The server lists the survivors `U3` (needs `|U3| >= t`). Every
   responding client reveals a share of `s_sk` for each *dropped* peer and a
   share of `b` for each *surviving* peer — never both for the same peer, so the
   server cannot unmask an individual input. The server reconstructs the dropped
   clients' `s_sk` (checked against the advertised public key), removes their
   dangling pairwise masks, and strips every survivor's self-mask.

Real-valued updates are clipped to L2 norm `C`, scaled by `2^16` and rounded.
`max_safe_clients` bounds the number of summands so the signed sum cannot wrap.

**Robust aggregation.** Secure aggregation hides individual updates, which rules
out per-update defences such as Krum. The `bucketed` aggregator shuffles clients
into `k` buckets, securely sums each, and takes the coordinate-wise median of
the `k` bucket means. A minority of attackers can corrupt at most a minority of
buckets, and the server still only sees sums over several clients.

## Complexity

With `n` clients, vector length `d`, threshold `t`:

| Quantity | Client | Server |
|----------|--------|--------|
| Computation | `O(n^2 + n d)` | `O(n^2 + n d)` per dropped client reconstruction `O(n t)` + PRG `O(n d)` |
| Communication | `O(n + d)` | `O(n^2 + n d)` |
| Storage | `O(n + d)` | `O(n^2 + n d)` |

Dropout recovery adds `O(|dropped| * |survivors| * d)` PRG expansion at the server.

## Usage

```python
import numpy as np
from secure_aggregation import run_protocol, encode, decode

rng = np.random.default_rng(0)
xs = {u: rng.normal(scale=0.3, size=50) for u in range(1, 21)}
enc = {u: encode(x, 2.0**16) for u, x in xs.items()}
res = run_protocol(enc, threshold=10, dropouts={"masked": {5, 6}, "unmask": {8}})
total = decode(res.total, 2.0**16)       # equals the sum over res.survivors
```

Federated training:

```python
from secure_aggregation import FLConfig, make_dataset, train
out = train(make_dataset(seed=0), FLConfig(aggregator="bucketed", n_buckets=10,
                                          byzantine_frac=0.2, clip=2.0))
print(out["accuracy"])
```

Run `python examples/demo.py` for the full demo and `python -m pytest -q` for the tests.

## Results

Output of `examples/demo.py` (30 clients, 10-dim logistic regression, non-IID shards):

| Experiment | Accuracy | Loss |
|-----------|----------|------|
| 20% dropout, plain mean (8 rounds) | 0.838 | 0.375 |
| 20% dropout, secure aggregation | 0.838 | 0.374 |
| No attack, secure mean (10 rounds) | 0.838 | 0.370 |
| 20% sign-flip attackers, plain / secure mean | 0.741 | 0.527 |
| 20% sign-flip attackers, bucketed median (10 buckets) | 0.795 | 0.467 |

Secure sums match the plaintext sum of survivors to within `4e-5` (pure
quantisation error). Protocol cost, `d = 100`, 10% dropping before masking:

| Clients | Time | Upload | Download |
|---------|------|--------|----------|
| 10 | 0.04 s | 16.5 kB | 14.8 kB |
| 20 | 0.16 s | 59.5 kB | 61.2 kB |
| 40 | 0.62 s | 225 kB | 248 kB |

The quadratic growth is visible. The bucketed median recovers part, not all, of
the accuracy lost to the attack; with 30% attackers it no longer helps.

## Limitations

This is a semi-honest simulation. The DH group is the multiplicative group mod
`2^255-19`, not a vetted prime-order group, and the protocol omits the
consistency-check round needed against an actively malicious server. Do not use
it for real data.

## References

- K. Bonawitz et al., "Practical Secure Aggregation for Privacy-Preserving
  Machine Learning", ACM CCS 2017.
- A. Shamir, "How to Share a Secret", Communications of the ACM, 1979.
- B. McMahan et al., "Communication-Efficient Learning of Deep Networks from
  Decentralized Data", AISTATS 2017.
- S. P. Karimireddy, L. He, M. Jaggi, "Byzantine-Robust Learning on
  Heterogeneous Datasets via Bucketing", ICLR 2022.

---

**Author:** Dr. Kainat Rizwan
