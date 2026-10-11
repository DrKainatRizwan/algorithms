# Graph Neural Network Message Passing from Scratch for Node Classification

## Overview

Pure NumPy implementations of three message-passing architectures — GCN, GraphSAGE (mean aggregator) and multi-head GAT — with hand-derived backward passes, dropout, softmax cross-entropy, Adam, early stopping, and a finite-difference gradient checker. A stochastic block model generator with class-conditional Gaussian features provides a semi-supervised node classification benchmark (few labels, weak features, homophilous edges), and an analysis module quantifies over-smoothing.

Package layout (`graph_message_passing/`):

| Module | Contents |
|---|---|
| `graph.py` | `Graph`, SBM generator, GCN / mean normalisation, split masks, homophily |
| `layers.py` | `GCNLayer`, `SAGELayer`, `GATLayer` with forward and backward |
| `model.py` | `GNN` stack, ReLU/dropout, cross-entropy, `Adam` |
| `train.py` | training loop with early stopping, evaluation, gradient check |
| `analysis.py` | Dirichlet energy, propagation depth probes |

## How it works

Each layer computes node updates from aggregated neighbour messages:

- **GCN** (Kipf & Welling): `H = Â X W + b`, with `Â = D^-1/2 (A + I) D^-1/2`. Backward: `dW = (ÂX)ᵀ dH`, `dX = Âᵀ dH Wᵀ`.
- **GraphSAGE**: `H = X W_s + P X W_n + b`, with `P` the row-normalised adjacency (neighbour mean).
- **GAT** (Veličković et al.): `z = XW`, `e_ij = LeakyReLU(a_sᵀ z_i + a_dᵀ z_j)`, `α_i· = softmax_j∈N(i)∪{i}(e_ij)`, `h_i = Σ_j α_ij z_j`. The backward pass differentiates through the masked softmax (`de = α ⊙ (dα − Σ α dα)`), the LeakyReLU and both attention vectors. Heads are concatenated.
- **MLP baseline**: a GCN layer with `Â = I`, so the same code path runs without structure.

Training is full batch: forward with inverted dropout, masked cross-entropy over the labelled nodes, backward, Adam with L2 weight decay; the best validation epoch (ties broken by validation loss) is restored. All gradients are validated against central differences in the test suite (relative error < 1e-4; the demo reports ~3e-8 for GAT).

## Complexity

With `n` nodes, `m` edges, `d` feature width, `L` layers (dense implementation as written):

- Time per layer: `O(n²d + nd²)` for GCN/SAGE (dense `Â`; `O(md + nd²)` with a sparse matrix); GAT `O(n²d + nd²)` per head for the score matrix.
- Space: `O(n²)` for propagation / attention matrices plus `O(nd)` activations per layer, and `O(d²)` parameters per layer.

## Usage

```python
from graph_message_passing import GNN, stochastic_block_model, split_masks, train, evaluate

g = stochastic_block_model([60] * 4, 0.12, 0.01, n_features=24, signal=0.35, seed=0)
tr, va, te = split_masks(g.y, per_class_train=5, n_val=60, seed=0)
model = GNN("gcn", g.adj, [24, 16, 4], dropout=0.5, seed=0)
train(model, g, tr, va, epochs=200)
print(evaluate(model, g, te))
```

Run the tests with `python -m pytest -q` and the demo with `python examples/demo.py`.

## Results

Demo graph: 240 nodes, 1105 edges, edge homophily 0.81, 20 labelled nodes (5 per class), 4 classes. Test accuracy over 5 seeds:

| Model | Params | Test acc (mean ± std) |
|---|---|---|
| MLP (no graph) | 468 | 0.565 ± 0.044 |
| GCN | 468 | 0.984 ± 0.005 |
| GraphSAGE | 916 | 0.972 ± 0.005 |
| GAT (2 heads) | 508 | 0.964 ± 0.023 |

Using the graph lifts accuracy by over 40 points. Pure propagation `Â^k X` shows the over-smoothing effect: Dirichlet energy falls from 1.785 (k=0) to 0.037 (k=32), and a nearest-centroid probe peaks at 0.983 for k=2–4, then collapses to 0.275 at k=32.

## References

- T. Kipf, M. Welling. *Semi-Supervised Classification with Graph Convolutional Networks*. ICLR 2017.
- W. Hamilton, R. Ying, J. Leskovec. *Inductive Representation Learning on Large Graphs*. NeurIPS 2017.
- P. Veličković et al. *Graph Attention Networks*. ICLR 2018.
- Q. Li, Z. Han, X.-M. Wu. *Deeper Insights into Graph Convolutional Networks for Semi-Supervised Learning*. AAAI 2018.
- D. Kingma, J. Ba. *Adam: A Method for Stochastic Optimization*. ICLR 2015.

---

**Author:** Dr. Kainat Rizwan
