"""Demo: run each DP family and report timings and sanity checks."""

import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dp_toolkit import *  # noqa: E402


def main() -> None:
    rng = random.Random(0)
    print("== Knapsack ==")
    w = [rng.randint(5, 60) for _ in range(100)]
    v = [rng.randint(10, 200) for _ in range(100)]
    r, t = timed(knapsack_01, w, v, 2000)
    print(f"0/1 n=100 W=2000 -> value {r.value}, weight {r.weight(w)}, {t*1e3:.1f} ms")
    r, t = timed(knapsack_unbounded, w[:10], v[:10], 2000)
    print(f"unbounded n=10 W=2000 -> value {r.value}, {t*1e3:.1f} ms")
    nums = [rng.randint(1000, 90000) for _ in range(60)]
    idx, t = timed(subset_sum_bitset, nums, sum(nums) // 2)
    print(f"bitset subset-sum n=60 -> {'found' if idx else 'none'} ({t*1e3:.1f} ms)")

    print("\n== Sequences ==")
    print("edit(kitten, sitting) =", edit_distance("kitten", "sitting"))
    a = "".join(rng.choice("ACGT") for _ in range(300))
    b = "".join(rng.choice("ACGT") for _ in range(300))
    d = edit_distance(a, b)
    s = edit_script(a, b)
    print(f"DNA 300x300: distance {d}, script ok={apply_script(a, s) == b}, banded={banded_edit_distance(a, b, d)}")
    x, tx = timed(lcs, list(a), list(b))
    y, ty = timed(lcs_hirschberg, list(a), list(b))
    print(f"LCS len {len(x)} (table {tx:.2f}s) vs Hirschberg len {len(y)} ({ty:.2f}s)")
    perm = [rng.randint(0, 10**6) for _ in range(100000)]
    out, t = timed(lis, perm)
    print(f"LIS of 100000 random ints: length {len(out)} in {t*1e3:.0f} ms")

    print("\n== Interval DP ==")
    dims = [rng.randint(5, 50) for _ in range(41)]
    (cost, split), t = timed(matrix_chain, dims)
    print(f"matrix chain n=40: {cost} mults, {t:.2f}s")
    print("small chain:", parenthesize(matrix_chain([10, 30, 5, 60])[1]), "cost", matrix_chain([10, 30, 5, 60])[0])
    print("optimal BST cost:", round(optimal_bst([.15, .1, .05, .1, .2], [.05, .1, .05, .05, .05, .1])[0], 3))
    print("burst balloons [3,1,5,8]:", burst_balloons([3, 1, 5, 8]))

    print("\n== Bitmask DP ==")
    nrng = np.random.default_rng(0)
    pts = nrng.random((12, 2))
    dist = np.linalg.norm(pts[:, None] - pts[None], axis=2)
    (best, tour), t = timed(held_karp, dist)
    print(f"Held-Karp 12 cities: tour length {best:.4f}, {t:.2f}s, tour {tour}")
    c = nrng.integers(1, 100, (12, 12)).astype(float)
    (best, _), t = timed(assignment_dp, c)
    print(f"assignment DP 12x12: cost {best:.0f}, {t:.2f}s")
    n = 9
    comp = [[int(i != j) for j in range(n)] for i in range(n)]
    print("Hamiltonian paths in K9:", count_hamiltonian_paths(comp))
    print("min set cover:", min_set_cover(5, [[0, 1, 2], [2, 3], [3, 4], [0, 4], [1, 3]]))


if __name__ == "__main__":
    main()
