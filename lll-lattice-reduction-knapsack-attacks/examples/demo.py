"""Demo: LLL/BKZ quality, knapsack density sweep, integer relations, LCG attack."""
import random
import sys
import time
from decimal import Decimal, getcontext
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lattice_reduction import (TruncatedLCG, bkz_reduce, hermite_factor, lll_reduce,
                               minimal_polynomial, predict_next, random_knapsack,
                               recover_lcg_states, solve_knapsack, svp)
from lattice_reduction.linalg import norm_sq


def qary_basis(n, q, rng):
    """Random q-ary lattice basis [[I, A], [0, qI]] of dimension 2n."""
    half = n
    rows = []
    for i in range(half):
        rows.append([int(i == j) for j in range(half)] + [rng.randrange(q) for _ in range(half)])
    for i in range(half):
        rows.append([0] * half + [q if i == j else 0 for j in range(half)])
    return rows


def main() -> None:
    rng = random.Random(2024)

    print("== Reduction quality on random q-ary lattices (dim 24, q=2^24) ==")
    basis = qary_basis(12, 2 ** 24, rng)
    t = time.perf_counter(); st = {}
    l = lll_reduce(basis, Fraction(99, 100), st)
    print(f"LLL  : |b1|={norm_sq(l[0]) ** .5:12.1f}  root-Hermite={hermite_factor(l):.4f}  "
          f"swaps={st['swaps']}  {time.perf_counter() - t:.2f}s")
    for beta in (6, 10):
        t = time.perf_counter(); st = {}
        z = bkz_reduce(basis, beta, stats=st)
        print(f"BKZ-{beta:<2}: |b1|={norm_sq(z[0]) ** .5:12.1f}  root-Hermite={hermite_factor(z):.4f}  "
              f"tours={st['tours']}  {time.perf_counter() - t:.2f}s")

    print("\n== Low-density subset-sum attack (n=26, 8 instances each) ==")
    print("density  LLL-solved  BKZ-8-solved")
    for dens in (0.6, 0.9, 1.0):
        ok_l = ok_b = 0
        for s in range(8):
            w, secret, target = random_knapsack(26, dens, random.Random(100 * s + int(dens * 100)))
            x = solve_knapsack(w, target)
            ok_l += x is not None
            if x is None:
                x = solve_knapsack(w, target, block_size=8)
            ok_b += x is not None
        print(f"  {dens:.2f}      {ok_l:2d}/8         {ok_b:2d}/8")

    print("\n== Minimal polynomials via integer relations ==")
    getcontext().prec = 150
    for name, alpha, deg in (("sqrt2+sqrt3", Decimal(2).sqrt() + Decimal(3).sqrt(), 4),
                             ("2^(1/3)+1", Decimal(2) ** (Decimal(1) / 3) + 1, 3),
                             ("golden ratio", (1 + Decimal(5).sqrt()) / 2, 2)):
        print(f"{name:13s} -> coeffs (const..lead) = {minimal_polynomial(alpha, deg, 60)}")

    print("\n== Truncated LCG state recovery (m=2^31, 8 outputs) ==")
    a, c, m = 1103515245, 12345, 2 ** 31
    print("hidden bits  success/10")
    for k in (10, 16, 20, 24, 27):
        good = 0
        for s in range(10):
            g = TruncatedLCG(a, c, m, k, 1000 + s)
            pairs = [g.next() for _ in range(8)]
            st = recover_lcg_states([p[1] for p in pairs], a, c, m, k)
            good += st == [p[0] for p in pairs]
        print(f"    {k:2d}         {good:2d}")
    g = TruncatedLCG(a, c, m, 16, 424242)
    pairs = [g.next() for _ in range(8)]
    st = recover_lcg_states([p[1] for p in pairs], a, c, m, 16)
    pred = predict_next(st, a, c, m, 3)
    actual = [g.next()[0] for _ in range(3)]
    print("predicted next states:", pred, "| actual:", actual, "| match:", pred == actual)

    print("\n== Exact SVP on a dim-12 q-ary lattice ==")
    small = qary_basis(6, 1009, rng)
    t = time.perf_counter()
    v = svp(small)
    print(f"shortest vector norm = {norm_sq(v) ** .5:.3f} (LLL gave "
          f"{norm_sq(lll_reduce(small)[0]) ** .5:.3f}) in {time.perf_counter() - t:.2f}s")


if __name__ == "__main__":
    main()
