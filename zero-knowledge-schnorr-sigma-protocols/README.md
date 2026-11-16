# Zero-Knowledge Sigma Protocols: Schnorr, Chaum-Pedersen, OR-Proofs and Pedersen Range Proofs

## Overview

A from-scratch toolkit for Σ-protocols (three-move honest-verifier zero-knowledge proofs) over prime-order Schnorr groups, using only the standard library. It includes:

- a **generic linear-relation Σ-protocol** covering Schnorr, Chaum-Pedersen (DLEQ), Okamoto representation proofs and AND-composition with shared witnesses;
- a **witness extractor** (special soundness) and an **HVZK simulator**;
- **Fiat-Shamir** non-interactive proofs with a domain-separated, length-prefixed transcript and context binding;
- **CDS OR-composition**, used for bit proofs;
- **Pedersen commitments** with opening, equality and **range proofs** by bit decomposition;
- **Schnorr signatures**, small-exponent batch verification and a nonce-reuse key-recovery attack;
- empirical analysis tools: exact simulator-vs-real distributions on a toy group, soundness-error measurement and a broken-Fiat-Shamir forgery.

## How it works

For a statement `Y_i = ∏_j G_ij^{x_j}` the prover picks nonces `k`, sends `T_i = ∏ G_ij^{k_j}`, receives a challenge `c ∈ Z_q`, and answers `s_j = k_j + c·x_j`. The verifier checks `∏ G_ij^{s_j} = T_i · Y_i^c`.

- **Completeness:** follows directly from the algebra.
- **Special soundness:** two accepting transcripts with the same `T` and `c ≠ c'` give `x = (s − s')/(c − c')`.
- **HVZK:** choose `s` uniformly, set `T_i = ∏ G_ij^{s_j} · Y_i^{−c}`. The result is distributed exactly like a real transcript (checked by enumeration on a toy group).
- **Fiat-Shamir:** `c = H(domain, statement, context, T)`. Omitting `T` from the hash lets anyone forge proofs (demonstrated in `analysis.py`).
- **OR-proofs:** simulate every false branch with a chosen challenge, run the true branch honestly, and force the branch challenges to sum to the hash challenge.
- **Range proof for `0 ≤ m < 2^n`:** commit to each bit `C_i = g^{b_i} h^{r_i}`, prove `b_i ∈ {0,1}` with an OR-proof (`C_i = h^r` or `C_i/g = h^r`), and check `∏ C_i^{2^i} = C`. The top-bit randomness is chosen so that `Σ 2^i r_i = r`.
- **Groups:** safe primes `p = 2q + 1`; `g` and `h` are hashed into the order-`q` subgroup so `log_g h` is unknown.

## Complexity

| Operation | Time (modular exponentiations) | Space |
|-----------|-------------------------------|-------|
| Σ-proof, `m` rows × `n` witnesses | prove/verify `O(m·n)` | `O(m + n)` |
| OR-proof with `b` branches | `O(b)` times branch cost | `O(b)` |
| `n`-bit range proof | `O(n)` | `O(n)` |
| Batch verify of `N` signatures | `O(N)` exps (one shared comparison) | `O(1)` extra |

Each exponentiation costs `O(log² p)`-ish bit operations with schoolbook arithmetic.

## Usage

```python
import random
from zk_sigma import *

rng = random.Random(1)
G = standard_group(128)
x = G.random_scalar(rng)
rel = schnorr_relation(G, G.exp(G.g, x))
proof = prove(rel, [x], rng, context=b"login")
assert verify(rel, proof, context=b"login")

ped = Pedersen(G)
rp = RangeProver(ped, 16)
commitment, rproof = rp.prove(40000, G.random_scalar(rng), rng)
assert rp.verify(commitment, rproof)
```

Run `python examples/demo.py` for the full tour and `python -m pytest -q` for the tests.

## Results

Output of `examples/demo.py` (128-bit safe prime; pure-Python, single core):

| Item | Result |
|------|--------|
| Schnorr prove / verify | 0.076 ms / 0.089 ms |
| Extractor | recovers the witness from two transcripts |
| HVZK, toy group `(p,q)=(23,11)` | total-variation distance real vs simulated = 0.0 |
| Cheating prover (guess the challenge), 50,000 trials | 0.0911 success vs 1/q = 0.0909 |
| 8-bit range proof prove / verify | 1.7 ms / 2.2 ms |
| 16-bit range proof prove / verify | 3.6 ms / 4.6 ms |
| 32-bit range proof prove / verify | 6.8 ms / 7.5 ms |
| 50 signatures, individual vs batch | 6.6 ms vs 6.0 ms |
| Nonce reuse across two signatures | private key recovered |
| Forgery against Fiat-Shamir that omits `T` | accepted; 0 / 200 against the correct hash |

Note: the built-in 64/128-bit groups are for demonstration and testing only; they offer no real security.

## References

- C. P. Schnorr, "Efficient Signature Generation by Smart Cards", *J. Cryptology* 4, 1991.
- D. Chaum and T. P. Pedersen, "Wallet Databases with Observers", CRYPTO 1992.
- T. P. Pedersen, "Non-Interactive and Information-Theoretic Secure Verifiable Secret Sharing", CRYPTO 1991.
- R. Cramer, I. Damgård and B. Schoenmakers, "Proofs of Partial Knowledge and Simplified Design of Witness Hiding Protocols", CRYPTO 1994.
- A. Fiat and A. Shamir, "How to Prove Yourself", CRYPTO 1986.
- D. Bernhard, O. Pereira and B. Warinschi, "How Not to Prove Yourself: Pitfalls of the Fiat-Shamir Heuristic", ASIACRYPT 2012.
- I. Damgård, "On Σ-protocols", lecture notes, Aarhus University.

---

**Author:** Dr. Kainat Rizwan
