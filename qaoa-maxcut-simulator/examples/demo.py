"""Demo: QAOA on 3-regular graphs, with depth scaling and classical comparison."""
import sys, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from qaoa_maxcut import *


def main() -> None:
    t0 = time.time()
    g = random_regular_graph(10, 3, seed=11)
    opt, _ = brute_force_maxcut(g)
    print(f"Graph: n={g.n}, m={g.m}, optimum cut={opt:.0f}")
    print(f"greedy={greedy_maxcut(g)[0]:.0f}  local-search={local_search_maxcut(g)[0]:.0f}")

    q1 = QAOA(g, 1)
    gs, bs = np.linspace(0, np.pi, 41), np.linspace(0, np.pi / 2, 41)
    land = landscape_p1(q1, gs, bs)
    i, j = np.unravel_index(land.argmax(), land.shape)
    print(f"p=1 grid best: {land.max():.4f} at gamma={gs[i]:.3f}, beta={bs[j]:.3f}; "
          f"closed form {p1_expectation(g, gs[i], bs[j]):.4f}")

    r = multistart(q1.energy, q1.gradient, 2, starts=6, seed=0)
    print(f"\n{'p':>2} {'<cut>':>8} {'ratio':>7} {'P(opt)':>8}")
    print(f"{1:>2} {r.value:8.4f} {r.value/opt:7.4f} {q1.success_probability(r.x):8.4f}")
    x = r.x
    for p in range(2, 6):
        q = QAOA(g, p)
        res = adam_ascent(q.energy, q.gradient, interp_init(x), lr=0.03, iters=250)
        x = res.x
        print(f"{p:>2} {res.value:8.4f} {res.value/opt:7.4f} {q.success_probability(x):8.4f}")
    print("\nTop bitstrings at p=5:", best_bitstring_stats(q, x, 3))
    mean, best = q.sample_energy(x, 2000, seed=1)
    print(f"2000 shots: mean cut {mean:.3f}, best cut {best:.0f}")

    ring = ring_graph(12)
    qr = QAOA(ring, 1)
    rr = multistart(qr.energy, qr.gradient, 2, starts=4, seed=1)
    print(f"\nRing(12) p=1 ratio {rr.value/ring.m:.4f} (theory 0.75)")
    print(f"elapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
