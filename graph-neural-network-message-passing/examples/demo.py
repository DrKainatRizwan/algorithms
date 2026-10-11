"""Compare GCN, GraphSAGE, GAT and an MLP on a synthetic citation-like graph."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from graph_message_passing import (GNN, edge_homophily, evaluate, gradient_check,
                                   label_agreement_after_hops, smoothing_curve,
                                   split_masks, stochastic_block_model, train)


def main() -> None:
    g = stochastic_block_model([60] * 4, 0.12, 0.01, n_features=24, signal=0.35, seed=0)
    tr, va, te = split_masks(g.y, 5, 60, seed=0)
    print(f"graph: {g.n_nodes} nodes, {g.n_edges} edges, homophily {edge_homophily(g):.2f}, "
          f"{tr.sum()} labelled")

    chk = GNN("gat", g.adj, [24, 8, 4], dropout=0.0, seed=0)
    print(f"GAT gradient check max rel. error: {gradient_check(chk, g, tr):.2e}\n")

    print(f"{'model':<6}{'params':>8}{'test acc':>11}{'mean+-std (5 seeds)':>24}{'time(s)':>9}")
    for kind in ("mlp", "gcn", "sage", "gat"):
        accs, t0 = [], time.time()
        for seed in range(5):
            m = GNN(kind, g.adj, [24, 16, 4], dropout=0.5, seed=seed)
            train(m, g, tr, va, epochs=200)
            accs.append(evaluate(m, g, te))
        print(f"{kind:<6}{m.n_params():>8}{accs[0]:>11.3f}"
              f"{np.mean(accs):>14.3f} +- {np.std(accs):.3f}{time.time() - t0:>9.2f}")

    print("\nOver-smoothing (pure propagation A^k X):")
    curve = smoothing_curve(g, 32)
    for k in (0, 1, 2, 4, 8, 16, 32):
        print(f"  hops={k:>2}  dirichlet energy={curve[k]:.4f}  "
              f"centroid acc={label_agreement_after_hops(g, k):.3f}")


if __name__ == "__main__":
    main()
