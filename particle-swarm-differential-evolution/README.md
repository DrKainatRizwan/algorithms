# Particle Swarm and Differential Evolution Optimizers

## Overview
From-scratch, numpy-only implementations of two population-based global optimisers for box-constrained continuous problems: Particle Swarm Optimisation (PSO) and Differential Evolution (DE). Both are fully vectorised over the population, count function evaluations exactly, and are seeded for reproducibility. A benchmark library (sphere, Rastrigin, Rosenbrock, Ackley, Griewank, Schwefel, Styblinski-Tang, ill-conditioned elliptic) and a multi-seed comparison harness with a rank-sum significance test are included.

Modules (`swarm_evolution_opt/`): `pso.py`, `de.py`, `benchmarks.py`, `runner.py`, `result.py`.

## How it works
**PSO.** Each particle keeps a velocity and its personal best. Per step
`v <- chi * (w v + c1 r1 (pbest - x) + c2 r2 (nbest - x))`, `x <- x + v`, where `nbest` is the best personal best in the particle's neighbourhood. Variants: constant inertia, linearly decaying inertia, and Clerc-Kennedy constriction. Topologies: global (gbest), ring (lbest, wrap-around) and von Neumann grid. Boundaries: clip (velocity zeroed), reflect, or random re-sampling. Velocity is clamped to a fraction of the domain width.

**DE.** For each target vector a mutant is built from scaled difference vectors (`rand/1`, `best/1`, `current-to-best/1`, `rand/2`, `best/2`), mixed with the target by binomial or exponential crossover, and kept if it is no worse (greedy selection). Out-of-range coordinates are bounced to the midpoint between the bound and the parent. With `adaptive=True`, every individual carries its own `(F, CR)` pair that is occasionally resampled, in the spirit of jDE.

## Complexity
Per iteration with population `N` and dimension `D`: time `O(N·D)` plus `N` objective evaluations (`O(N^2)` for the distinct-index sampling used by DE here, `O(N·k)` for ring PSO, `O(N)` for global PSO); space `O(N·D)`.

## Usage
```python
from swarm_evolution_opt import get_benchmark, pso_minimize, de_minimize

b = get_benchmark("rastrigin")
lo, hi = b.bounds(10)
r1 = pso_minimize(b.func, lo, hi, n_particles=30, topology="ring", max_iter=600, seed=0)
r2 = de_minimize(b.func, lo, hi, pop_size=40, adaptive=True, max_iter=500, seed=0)
print(r1.fun, r2.fun, r2.nfev)
```
Any vectorised objective mapping an `(n, d)` array to `n` values can be used.

## Results
`python examples/demo.py`: dimension 10, 20 000 evaluations per run, 8 seeds; mean final error `f - f*`.

| Function | PSO-global | PSO-ring | PSO-constr | DE-rand1 | DE-best1 | jDE |
|---|---|---|---|---|---|---|
| sphere | 2.0e-32 | 3.5e-18 | 8.8e-32 | 2.7e-15 | 2.4e-15 | 5.9e-18 |
| rosenbrock | 2.12 | 4.84 | 1.28 | 0.58 | 6.78 | 2.60 |
| rastrigin | 7.59 | 6.84 | 8.96 | 25.2 | 16.1 | 2.65 |
| ackley | 9.3e-15 | 2.5e-08 | 5.8e-15 | 4.5e-07 | 0.29 | 2.5e-08 |

No single method dominates: PSO converges fastest on separable unimodal and Ackley landscapes, plain DE/rand/1 is best on the curved Rosenbrock valley, and adaptive DE is clearly best on multimodal Rastrigin (jDE vs PSO-global rank-sum p = 0.021). The full demo runs in about 12 s.

Run the tests with `python -m pytest -q`.

## References
- Kennedy, Eberhart. *Particle Swarm Optimization*, IEEE ICNN, 1995.
- Shi, Eberhart. *A Modified Particle Swarm Optimizer*, IEEE CEC, 1998.
- Clerc, Kennedy. *The Particle Swarm: Explosion, Stability, and Convergence*, IEEE TEC, 2002.
- Storn, Price. *Differential Evolution - A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces*, J. Global Optimization, 1997.
- Brest et al. *Self-Adapting Control Parameters in Differential Evolution*, IEEE TEC, 2006.

---

**Author:** Dr. Kainat Rizwan
