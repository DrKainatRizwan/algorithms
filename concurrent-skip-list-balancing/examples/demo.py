"""Demo: indexable skip list, balance statistics and a threaded stress test."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from skiplist_lab import GeometricLevels, HashLevels, SkipList, level_histogram, measure_search_cost, optimal_p
from skiplist_lab.analysis import expected_height
from skiplist_lab.benchmark import compare_with_sorted_list, threaded_stress


def main() -> None:
    sl = SkipList(GeometricLevels(seed=7))
    for k in range(0, 100, 3):
        sl.insert(k, f"v{k}")
    print("range [30, 50):", [k for k, _ in sl.range(30, 50)])
    print("rank(51) =", sl.rank(51), "| kth(10) =", sl.kth(10)[0], "| floor(50) =", sl.floor(50))

    print("\nSearch cost vs. theory (p = 0.5), mean comparisons per lookup")
    for n in (1_000, 10_000, 100_000):
        r = measure_search_cost(n, 0.5, seed=1)
        print(f"  n={n:>7}  height={r['height']:>2} (theory {expected_height(n):4.1f})  "
              f"comparisons={r['mean_comparisons']:6.2f}  bound={r['predicted_steps']:6.2f}")

    print("\nEffect of p at n = 20000")
    for p in (0.125, 0.25, 0.37, 0.5, 0.75):
        r = measure_search_cost(20_000, p, seed=1)
        print(f"  p={p:<5} comparisons={r['mean_comparisons']:6.2f}  bound={r['predicted_steps']:6.2f}")
    print("  analytically optimal p ~", optimal_p(20_000))

    big = SkipList(GeometricLevels(seed=3))
    for k in range(20_000):
        big.insert(k)
    print("\nTower height histogram:", level_histogram(big.tower_heights()))

    a, b = SkipList(HashLevels()), SkipList(HashLevels())
    for k in range(500):
        a.insert(k)
    for k in reversed(range(500)):
        b.insert(k)
    print("History independent (hash levels):", a.tower_heights() == b.tower_heights())

    c = compare_with_sorted_list(20_000)
    print(f"\n20k random inserts: skip list {c['skip_insert_s']:.3f}s, sorted array {c['array_insert_s']:.3f}s")

    res = threaded_stress(threads=4, ops=3000, key_space=100, seed=1)
    lst, net = res["list"], res["net"]
    ok = lst.keys() == [k for k, v in enumerate(net) if v == 1]
    print(f"\n4 threads x 3000 mixed ops: {res['ops_per_s']:.0f} ops/s, "
          f"CAS failures={res['cas_failures']}, final size={len(lst)}, consistent with op log: {ok}")


if __name__ == "__main__":
    main()
