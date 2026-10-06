# Federated Averaging with Byzantine-Robust Aggregation

## Overview
A from-scratch NumPy simulator of synchronous federated learning in which a
fraction of clients is malicious. It implements a softmax-regression model,
IID and Dirichlet label-skew partitioning, four Byzantine attacks, and six
server-side aggregation rules (mean, coordinate-wise median, trimmed mean,
Krum, Multi-Krum, geometric median).

## How it works
Each round the server sends the global weights `w` to all clients. Honest
clients run local minibatch SGD and return the delta `w_local - w`. Byzantine
clients are omniscient: they see the honest deltas and craft their own
(sign flip, amplified negation, Gaussian noise, or "a little is enough", which
shifts the mean by `z` standard deviations per coordinate). The server
aggregates all deltas with the chosen rule and applies the result.

- **Mean** is optimal without attackers but a single bad update can move it arbitrarily.
- **Median / trimmed mean** act per coordinate; trimming drops the `f` largest and smallest values.
- **Krum** scores each update by the summed squared distance to its `n-f-2` nearest neighbours and picks the minimum; **Multi-Krum** averages the best `n-f`.
- **Geometric median** is computed with Weiszfeld iterations.

## Complexity
With `n` clients and model dimension `d`:

| Rule | Time | Space |
|------|------|-------|
| mean | O(nd) | O(d) |
| median / trimmed mean | O(nd log n) | O(nd) |
| Krum / Multi-Krum | O(n²d) | O(n²) |
| geometric median | O(T·nd) | O(nd) |

## Usage
```python
from byzantine_fedavg import *

X, y = make_blobs_dataset(4000, 10, 4, seed=7)
shards = partition_label_skew(y[:3200], 20, alpha=5.0)
sim = FederatedSimulator(SoftmaxModel(10, 4), X[:3200], y[:3200], shards,
                         X[3200:], y[3200:], trimmed_mean,
                         n_byzantine=4, attack=ATTACKS["gaussian"])
print(sim.run(15).final_accuracy)
```
Run `python examples/demo.py` for the full comparison and
`python -m pytest -q` for the tests.

## Results
20 clients (4 Byzantine), 15 rounds, test accuracy from `examples/demo.py`:

| Aggregator | none | sign_flip | scaled_negative | gaussian | lie |
|------------|------|-----------|-----------------|----------|-----|
| mean | 0.991 | 0.991 | 0.000 | 0.746 | 0.991 |
| median | 0.991 | 0.991 | 0.991 | 0.991 | 0.991 |
| trimmed_mean | 0.991 | 0.991 | 0.993 | 0.991 | 0.991 |
| krum | 0.993 | 0.993 | 0.993 | 0.993 | 0.993 |
| multi_krum | 0.991 | 0.991 | 0.991 | 0.991 | 0.991 |
| geomed | 0.991 | 0.991 | 0.991 | 0.991 | 0.991 |

Plain averaging collapses under amplified or noisy updates, while every robust
rule is unaffected. The task is easy and the honest updates are close to one
another, so the mild attacks (sign flip with 4/20 clients, LIE) do not hurt even the mean.

## References
- Blanchard et al., "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent", NeurIPS 2017.
- Yin et al., "Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates", ICML 2018.
- Baruch et al., "A Little Is Enough: Circumventing Defenses for Distributed Learning", NeurIPS 2019.
- McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data", AISTATS 2017.
- Pillutla et al., "Robust Aggregation for Federated Learning", 2019.

---

**Author:** Dr. Kainat Rizwan
