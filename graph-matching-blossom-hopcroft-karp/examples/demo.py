"""Demo: compare matching algorithms and verify stable matchings."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from graph_matching import (
    blocking_pairs, brute_force_max_matching, edmonds_blossom, gale_shapley,
    greedy_matching, hopcroft_karp, konig_vertex_cover, kuhn_matching,
    random_bipartite, random_graph, random_preferences,
)
from graph_matching.bench import general_size, matching_size, timed


def main() -> None:
    print("== Bipartite matching (n=1500 per side) ==")
    for p in (0.002, 0.005):
        n = 1500
        adj = random_bipartite(n, n, p, seed=1)
        edges = sum(map(len, adj))
        hk, t_hk = timed(lambda: hopcroft_karp(adj, n), 2)
        ku, t_ku = timed(lambda: kuhn_matching(adj, n), 2)
        assert matching_size(hk) == matching_size(ku)
        l, r = konig_vertex_cover(adj, n, hk)
        print(f"p={p} edges={edges} matching={matching_size(hk)} "
              f"cover={len(l)+len(r)} HK={t_hk*1000:.1f}ms Kuhn={t_ku*1000:.1f}ms")

    print("== General graphs: blossom vs exact ==")
    for seed in range(3):
        adj = random_graph(16, 0.2, seed)
        m = edmonds_blossom(adj)
        print(f"seed={seed} blossom={general_size(m)} "
              f"greedy={general_size(greedy_matching(adj))} "
              f"exact={brute_force_max_matching(adj)}")
    adj = random_graph(400, 0.01, 7)
    m, t = timed(lambda: edmonds_blossom(adj), 2)
    print(f"n=400 blossom size={general_size(m)} in {t*1000:.1f}ms "
          f"(greedy={general_size(greedy_matching(adj))})")

    print("== Stable matching ==")
    a, b = random_preferences(200, seed=3)
    m, t = timed(lambda: gale_shapley(a, b), 2)
    ranks = [a[p].index(m[p]) + 1 for p in range(200)]
    print(f"n=200 blocking pairs={len(blocking_pairs(m, a, b))} "
          f"mean proposer rank={sum(ranks)/200:.2f} time={t*1000:.1f}ms")


if __name__ == "__main__":
    main()
