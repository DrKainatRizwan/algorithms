# Local Differential Privacy: Frequency Estimation from Scratch

## Overview

In the local model of differential privacy each user perturbs their own value before it leaves the device, so the aggregator never sees raw data. This package implements the main *frequency oracles* from the LDP literature in pure NumPy, together with the tooling needed to use them properly: exact privacy auditing, variance theory, simplex post-processing, RAPPOR-style longitudinal protection and a prefix-tree heavy-hitter protocol for very large domains.

| Module | Contents |
|---|---|
| `ldp_frequency/oracles.py` | GRR (direct encoding), symmetric and optimised unary encoding (SUE/OUE), optimised local hashing (OLH), Hadamard response with a fast Walsh-Hadamard decoder |
| `ldp_frequency/rappor.py` | One-hot RAPPOR with permanent and instantaneous responses, longitudinal averaging attack |
| `ldp_frequency/postprocess.py` | Clip-and-normalise, simplex projection, Norm-Sub, base-cut |
| `ldp_frequency/analysis.py` | Oracle selection, variance tables, exact channel-matrix privacy audit, Zipf data, metrics |
| `ldp_frequency/heavy_hitters.py` | Prefix-tree heavy-hitter discovery over `2^b` domains |

## How it works

Every oracle is a randomiser whose report "supports" the true value with probability `p` and any other value with probability `q`. With `C(v)` the number of supporting reports among `n`,

```
f_hat(v) = (C(v)/n - q) / (p - q)          Var = pi (1 - pi) / (n (p - q)^2),   pi = q + f (p - q)
```

is unbiased. The oracles differ only in `(p, q)` and in communication:

* **GRR** keeps the value with `p = e^eps / (e^eps + d - 1)`, else reports a uniformly random other value. Best for `d < 3 e^eps + 2`; variance grows with `d`.
* **SUE / OUE** one-hot encode and flip bits. SUE (basic RAPPOR) uses `p = 1 - q = e^(eps/2)/(e^(eps/2)+1)`; OUE uses `p = 1/2`, `q = 1/(e^eps + 1)`, which minimises variance. Reports cost `d` bits.
* **OLH** hashes the value with a user-specific universal hash to `g = round(e^eps + 1)` buckets and applies GRR to the bucket. Variance matches OUE and is independent of `d`; a report is `(a, b, y)`.
* **Hadamard response** associates value `x` with row `x+1` of a `K x K` Hadamard matrix (`K` is the next power of two above `d`) and reports a column from the row's `+1` set w.p. `e^eps/(e^eps+1)`. Two different non-zero rows agree on exactly `K/4` columns, so `q = 1/2`. Reports need `log2 K` bits and the server decodes all items with one FWHT.
* **RAPPOR** adds a memoised permanent randomised response (`f`) below a fresh instantaneous one (`p`, `q`). The permanent step bounds the leakage of unlimited repeated reports by `eps_inf = 2 ln((1 - f/2)/(f/2))`.
* **Post-processing** projects the estimate onto the probability simplex (free in privacy terms) — Norm-Sub is shown to equal the Euclidean projection.
* **Heavy hitters** split users into levels; level `j` reports a length-`L_j` prefix over the surviving candidates plus a catch-all bucket, keeping the top `k` per level.

`audit()` computes the exact worst-case log-likelihood ratio of each randomiser's channel matrix, so the privacy claim of every oracle is checked numerically rather than assumed.

## Complexity

| Oracle | Client time | Report size | Server time | Server memory |
|---|---|---|---|---|
| GRR | O(1) | log d bits | O(n + d) | O(d) |
| OUE / SUE | O(d) | d bits | O(n d) | O(d) |
| OLH | O(1) | O(log g) + hash seed | O(n d) | O(d) |
| HR | O(1) expected | log K bits | O(n + K log K) | O(K) |

Simplex projection is O(d log d). Heavy hitters cost O(n) oracle work per level with candidate sets of size at most `k 2^step + 1`.

## Usage

```python
import numpy as np
from ldp_frequency import make_oracle, project_simplex, sample_users, zipf_distribution

rng = np.random.default_rng(0)
d, n, eps = 128, 50_000, 1.0
values = sample_users(zipf_distribution(d, 1.1), n, rng)

oracle = make_oracle("OLH", d, eps)
reports = oracle.perturb(values, rng)          # runs on the clients
estimate = project_simplex(oracle.estimate(reports))   # runs on the server
```

## Results

Output of `python examples/demo.py` (seeded, runs in about a second). Mean squared error of the frequency vector, `n = 50000`, `eps = 1`, Zipf(1.1) data:

| d | GRR | OUE | OLH | HR |
|---|---|---|---|---|
| 8 | 5.69e-05 | 2.31e-05 | 8.21e-05 | 8.76e-05 |
| 32 | 2.23e-04 | 7.79e-05 | 5.94e-05 | 9.96e-05 |
| 128 | 1.00e-03 | 7.32e-05 | 8.32e-05 | 7.48e-05 |
| 512 | 3.42e-03 | 7.42e-05 | 7.21e-05 | 8.28e-05 |

GRR degrades linearly in `d`, while OUE, OLH and HR stay flat; HR does so with `log2 K` bits per report (11 bits at `d = 1024`, against 1024 for OUE and 65 for OLH).

* Simplex projection (OLH, `d = 256`, `n = 30000`) lowers MSE from 1.19e-04 to 1.96e-05.
* RAPPOR (`f = 0.6`, `p = 0.7`, `q = 0.3`): `eps_inf = 1.695`, single-report `eps = 0.646`. An attacker averaging 300 rounds recovers the value 100% of the time without memoisation, but only 34.5% with it.
* Heavy hitters over a `2^20` domain (`n = 200000`, `eps = 4`) recover all four planted items (true 0.20 / 0.15 / 0.10 / 0.07, estimated 0.195 / 0.151 / 0.102 / 0.068) while querying at most 96 candidates per level.
* The privacy audit returns exactly `eps = 1.5` for GRR, SUE, OUE, OLH and HR configured with `eps = 1.5`.

Run the test-suite (39 tests) with `python -m pytest -q`.

## References

* S. L. Warner. Randomized response: a survey technique for eliminating evasive answer bias. *JASA*, 1965.
* Ú. Erlingsson, V. Pihur, A. Korolova. RAPPOR: Randomized Aggregatable Privacy-Preserving Ordinal Response. *CCS*, 2014.
* T. Wang, J. Blocki, N. Li, S. Jha. Locally Differentially Private Protocols for Frequency Estimation. *USENIX Security*, 2017.
* J. Acharya, Z. Sun, H. Zhang. Hadamard Response: Estimating Distributions Privately, Efficiently, and with Little Communication. *AISTATS*, 2019.
* T. Wang, N. Li, S. Jha. Locally Differentially Private Frequent Itemset Mining. *IEEE S&P*, 2018.
* T. Wang et al. Locally Differentially Private Heavy Hitter Identification. *IEEE TDSC*, 2021.
* R. Bassily, A. Smith. Local, Private, Efficient Protocols for Succinct Histograms. *STOC*, 2015.
* C. Dwork, A. Roth. The Algorithmic Foundations of Differential Privacy. 2014.

---

**Author:** Dr. Kainat Rizwan
