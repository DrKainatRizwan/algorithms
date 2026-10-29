"""Compare PSO and DE variants on standard benchmarks under an equal evaluation budget."""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from swarm_evolution_opt import compare_optimizers, summarize, wilcoxon_rank_sum  # noqa: E402

CONFIGS = {
    "PSO-global": ("pso", {"n_particles": 30}),
    "PSO-ring": ("pso", {"n_particles": 30, "topology": "ring"}),
    "PSO-constr": ("pso", {"n_particles": 30, "variant": "constriction", "c1": 2.05, "c2": 2.05}),
    "DE-rand1": ("de", {"pop_size": 40}),
    "DE-best1": ("de", {"pop_size": 40, "strategy": "best1"}),
    "jDE": ("de", {"pop_size": 40, "adaptive": True}),
}

t0 = time.time()
dim, seeds = 10, list(range(8))
for bench in ("sphere", "rosenbrock", "rastrigin", "ackley"):
    errs = compare_optimizers(CONFIGS, bench, dim, seeds, max_nfev=20000)
    stats = summarize(errs)
    print(f"\n== {bench} (d={dim}, 20000 evals, {len(seeds)} seeds) ==")
    for name, s in stats.items():
        print(f"{name:11s} mean={s['mean']:10.3e} median={s['median']:10.3e} best={s['best']:10.3e}")
    z, p = wilcoxon_rank_sum(errs["jDE"], errs["PSO-global"])
    print(f"jDE vs PSO-global: z={z:+.2f} p={p:.4f}")
print(f"\nelapsed {time.time() - t0:.1f}s")
