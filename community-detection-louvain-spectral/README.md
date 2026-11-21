# Community Detection: Louvain, Label Propagation, Spectral Clustering and Girvan-Newman

## Overview

A from-scratch toolkit (NumPy only) for finding communities in weighted undirected graphs. It implements four classic approaches on one shared graph type, with modularity, NMI, ARI and conductance for evaluation, synthetic generators (planted partition / stochastic block model, ring of cliques) and checks for disconnected communities.

## How it works

- **Louvain** (`louvain.py`): phase 1 moves single nodes to the neighbouring community with the best modularity gain `k_i,in(c) - gamma * tot(c) * k_i / 2m` until no move helps; phase 2 collapses communities into super-nodes (internal weight becomes a self-loop, which preserves modularity exactly) and repeats. Every hierarchy level is returned. A resolution parameter `gamma` counters the resolution limit.
- **Label propagation** (`label_propagation.py`): every node adopts the label with the greatest incident weight in random order until each node holds a plurality label. Near-linear time but unstable on weak structure.
- **Spectral clustering** (`spectral.py`): bottom `k` eigenvectors of the normalised Laplacian, row-normalised, clustered with k-means++ (own implementation). `k` can be chosen by the eigengap.
- **Girvan-Newman** (`girvan_newman.py`): repeatedly delete the edge of highest Brandes betweenness and keep the component split with the best modularity.

## Complexity

| Method | Time | Space |
|---|---|---|
| Louvain | ~O(m) per pass, few passes in practice | O(n + m) |
| Label propagation | O(m) per sweep | O(n + m) |
| Spectral | O(n^3) dense eigendecomposition | O(n^2) |
| Girvan-Newman | O(m^2 n) | O(n + m) |

## Usage

```python
from graph_communities import louvain, planted_partition, nmi

g, truth = planted_partition([30] * 4, p_in=0.4, p_out=0.06, seed=11)
res = louvain(g, seed=0)
print(res.modularity, nmi(res.labels, truth))
```

Run `python -m pytest -q` for the 17 tests and `python examples/demo.py` for the benchmark below.

## Results (from `examples/demo.py`)

Planted partition, 4 blocks of 30 nodes, `p_in = 0.4`:

| p_out | Method | Q | NMI |
|---|---|---|---|
| 0.02 | Louvain / label-prop / spectral | 0.623 | 1.000 |
| 0.06 | Louvain | 0.445 | 1.000 |
| 0.06 | Label propagation | 0.203 | 0.471 |
| 0.06 | Spectral (k=4) | 0.445 | 1.000 |
| 0.10 | Louvain | 0.315 | 1.000 |
| 0.10 | Label propagation | 0.000 | 0.000 |
| 0.10 | Spectral (k=4) | 0.314 | 0.974 |

Resolution limit, ring of 30 cliques of size 5: Louvain at `gamma = 1` merges cliques into 17 communities (NMI 0.903, Q 0.886, higher than the true partition's 0.876); at `gamma = 2` it recovers all 30 (NMI 1.000). Girvan-Newman on 36 nodes takes about 0.13 s and matches Louvain (Q = 0.577); Louvain handles a 600-node, 6067-edge graph in about 45 ms.

## References

- V. Blondel, J.-L. Guillaume, R. Lambiotte, E. Lefebvre, "Fast unfolding of communities in large networks", J. Stat. Mech. (2008).
- M. Girvan, M. Newman, "Community structure in social and biological networks", PNAS (2002).
- U. Raghavan, R. Albert, S. Kumara, "Near linear time algorithm to detect community structures in large-scale networks", Phys. Rev. E (2007).
- A. Ng, M. Jordan, Y. Weiss, "On spectral clustering: analysis and an algorithm", NIPS (2001).
- S. Fortunato, M. Barthelemy, "Resolution limit in community detection", PNAS (2007).
- U. Brandes, "A faster algorithm for betweenness centrality", J. Math. Sociol. (2001).

---

**Author:** Dr. Kainat Rizwan
