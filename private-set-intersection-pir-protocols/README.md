# Private Set Intersection and Private Information Retrieval

From-scratch implementations of two core privacy-preserving protocols, in pure Python with numpy only:

* **Private set intersection (PSI)**: two parties learn `A ∩ B` (or only `|A ∩ B|`) without revealing the rest of their sets.
* **Private information retrieval (PIR)**: a client reads record `i` from a database without the server(s) learning `i`.

## Overview

| Module | Contents |
|---|---|
| `numtheory.py` | Miller-Rabin, safe primes, hash-to-group, the prime-order quadratic-residue group |
| `paillier.py` | Paillier additively homomorphic encryption |
| `psi.py` | Naive hash PSI (and the dictionary attack on it), Diffie-Hellman PSI, DH cardinality PSI, Bloom-filter PSI |
| `database.py` | Fixed-length record database |
| `pir.py` | Trivial PIR, k-server XOR PIR, 2^d-server cube PIR, single-server Paillier PIR (linear and square-root layouts) |
| `audit.py` | Exact privacy audit by enumerating all random coins, plus communication cost models |

## How it works

**DH-PSI.** In a safe-prime group `p = 2q+1`, hash each item into the order-`q` subgroup. The client sends `H(x)^a`; the server returns `H(x)^(ab)` in the client's order and sends its own shuffled `H(y)^b`; the client computes `(H(y)^b)^a` and matches. Under DDH the blinded values reveal nothing about non-matching items. In the cardinality variant the server shuffles the doubly blinded list, so the client only counts matches.

**Naive hash PSI** sends `SHA-256(y)`. For a small universe (user ids, phone numbers) the receiver simply hashes every candidate: `dictionary_attack` shows this recovers the entire server set.

**XOR PIR.** For `k` servers the client draws `k-1` random subsets of `[n]` and a final one equal to their XOR with `e_i`. Each server returns the XOR of the selected records; XORing all answers cancels everything except record `i`. Any `k-1` servers see uniformly random subsets, so the scheme is private against `k-1` colluders; all `k` together learn `i`.

**Cube PIR.** Arrange `n = m^d` records in a `d`-cube. Pick random subsets `S_j ⊆ [m]` per dimension; server `b ∈ {0,1}^d` receives `S_j` or `S_j △ {i_j}` per dimension according to `b_j` and returns the XOR of the sub-cube. A record is counted by an odd number of servers iff it sits at coordinates `i`. Upload drops to `2^d · d · n^(1/d)` bits.

**Paillier PIR.** Records form an `R × C` matrix. The client uploads `Enc(e_col)` (`C` ciphertexts). For each row the server computes `∏_c Enc(e_c)^{rec[row, c]}`, a homomorphic dot product, and returns `R` ciphertexts per chunk. The client decrypts the target row. `R = 1` is linear; `R ≈ √n` balances to `O(√n)` both ways.

The privacy audit (`is_private`) enumerates the whole coin space for tiny `n` and checks the coalition's view distribution is *identical* for every index, proving the claims above exactly rather than statistically.

## Complexity

| Scheme | Servers | Upload | Download | Server work |
|---|---|---|---|---|
| DH-PSI | 1 | `O(|A|)` group elements | `O(|A| + |B|)` | `O(|A| + |B|)` modular exponentiations |
| Trivial PIR | 1 | 0 | `n·L` | 0 |
| XOR PIR | k | `k·n` bits | `k·L` | `O(n·L)` each |
| Cube PIR | 2^d | `2^d·d·n^(1/d)` | `2^d·L` | `O(n·L)` each |
| Paillier PIR (linear) | 1 | `n` ciphertexts | `⌈L/c⌉` ciphertexts | `n` exponentiations per chunk |
| Paillier PIR (sqrt) | 1 | `√n` ciphertexts | `√n·⌈L/c⌉` | same |

Space is linear in the database / set sizes throughout.

## Usage

```python
import random
from private_retrieval import SchnorrGroup, dh_psi, Database, CubePIR, PaillierPIR

group = SchnorrGroup.generate(128, seed=7)
res = dh_psi(["alice", "bob", "carol"], ["bob", "dave", "carol"], group)
print(sorted(res.intersection))          # [b'bob', b'carol']

db = Database.random(4096, 16, seed=1)
print(CubePIR(2).retrieve(db, 1234, random.Random(0)).record == db.record(1234))   # True
print(PaillierPIR(256, seed=3).retrieve(Database.random(256, 16), 77).upload_bits)
```

## Results

Output of `python examples/demo.py` (128-bit group for speed; not a secure parameter size):

```
PSI  |client|=150 |server|=200, true overlap = 22
naive-hash  card=  22 exact=True up=     0B down=  6400B     0.2 ms
bloom(1%)   card=  25 exact=False up=     0B down=   240B     1.1 ms
dh-psi      card=  22 exact=True up=  2400B down=  5600B    31.7 ms
dh-card     card=  22 exact=None up=  2400B down=  5600B    31.6 ms
dictionary attack on naive hashes recovers 200/200 server items

PIR (n=4096 records of 16 B)
xor-2   correct=True up=   8192b down=  256b
xor-4   correct=True up=  16384b down=  512b
cube-2  correct=True up=    512b down=  512b
cube-3  correct=True up=    384b down= 1024b
trivial download = 524288 bits

Paillier PIR (n=256, 16 B, 256-bit key)
linear  up=131072b down=512b   0.16 s
sqrt    up=  8192b down=8192b  0.08 s
```

Takeaways: Bloom-filter PSI is smallest but leaks membership probabilistically and returns false positives (3 of 25 here); naive hashing is broken on a small universe; DH-PSI is exact at the cost of one exponentiation per item. For PIR, cube PIR cuts upload from 8192 to 512 bits relative to 2-server XOR at the price of more servers, and the square-root layout cuts Paillier upload 16× at 16× the download.

## Tests

`python -m pytest -q` runs 30+ deterministic tests covering primality, group membership, PSI correctness and edge cases, the dictionary attack, Bloom false-positive rates, PIR correctness for every scheme and dimension, Paillier homomorphism, exact privacy audits (including the failure under full collusion) and cost models.

## References

* C. Meadows, "A More Efficient Cryptographic Matchmaking Protocol for Use in the Absence of a Continuously Available Third Party", IEEE S&P 1986.
* B. Huberman, M. Franklin, T. Hogg, "Enhancing Privacy and Trust in Electronic Communities", EC 1999.
* B. Chor, O. Goldreich, E. Kushilevitz, M. Sudan, "Private Information Retrieval", FOCS 1995 / JACM 1998.
* E. Kushilevitz, R. Ostrovsky, "Replication is Not Needed: Single Database, Computationally-Private Information Retrieval", FOCS 1997.
* P. Paillier, "Public-Key Cryptosystems Based on Composite Degree Residuosity Classes", EUROCRYPT 1999.
* B. Pinkas, T. Schneider, M. Zohner, "Scalable Private Set Intersection Based on OT Extension", ACM TOPS 2018.

---

**Author:** Dr. Kainat Rizwan
