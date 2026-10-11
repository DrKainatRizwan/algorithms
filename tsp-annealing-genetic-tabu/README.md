# Simulated Annealing vs Genetic Algorithm vs Tabu Search for the TSP

## Overview
From-scratch NumPy implementations of three metaheuristics for the Euclidean travelling salesman problem, plus the baselines needed to judge them: nearest-neighbour construction, 2-opt and Or-opt local search, and an exact Held-Karp solver for small instances.

## How it works
- **Simulated annealing** (`annealing.py`): random 2-opt reversals, Metropolis acceptance `exp(-Δ/T)`, geometric cooling; the start temperature is calibrated from sampled uphill moves.
- **Genetic algorithm** (`genetic.py`): tournament selection, order crossover (OX) or edge recombination (ERX), segment-reversal mutation, elitism, and a 2-opt polish of each generation's best individual.
- **Tabu search** (`tabu.py`): best-admissible 2-opt move each iteration; edges removed by a move are tabu for a tenure of ~√n iterations, with aspiration when a move beats the global best.
- **Baselines** (`local_search.py`, `instances.py`): nearest neighbour, first-improvement 2-opt with O(1) delta evaluation, Or-opt, Held-Karp DP and brute force.

## Complexity
| Method | Time | Space |
|---|---|---|
| Nearest neighbour | O(n²) | O(n) |
| 2-opt pass | O(n²) | O(n) |
| Annealing | O(I) moves, O(n) per accepted reversal | O(n) |
| Genetic algorithm | O(G·P·n) plus crossover cost | O(P·n) |
| Tabu search | O(I·n²) full neighbourhood (or O(I·k) sampled) | O(n²) move list |
| Held-Karp | O(n²·2ⁿ) | O(n·2ⁿ) |

## Usage
```python
from tsp_metaheuristics import random_instance, simulated_annealing, tabu_search, genetic_algorithm

inst = random_instance(50, seed=7)
r = simulated_annealing(inst, iterations=30_000, seed=0)
print(r.length, r.tour[:10])
```

Run `python examples/demo.py` for the comparison and `python -m pytest -q` for the tests.

## Results
From `examples/demo.py` (random uniform cities in the unit square, mean of 3 seeds, lower is better):

n = 11 against the Held-Karp optimum 2.8341: all three metaheuristics reach it (0.00% gap).

| Method | n=50 mean | n=50 s | n=100 mean | n=100 s |
|---|---|---|---|---|
| Nearest neighbour | 7.0072 | 0.00 | 9.0054 | 0.00 |
| NN + 2-opt | 6.2343 | 0.00 | 8.4474 | 0.02 |
| Annealing (30k iters) | 6.1875 | 0.19 | 8.8140 | 0.19 |
| Genetic (40 × 100 gens) | 6.5908 | 0.53 | 8.9738 | 1.00 |
| Tabu (150 iters, 300 sampled moves) | 6.2918 | 0.20 | 8.6214 | 0.20 |

At a fixed small budget, annealing wins on n=50, while on n=100 the budget is too small and plain NN+2-opt remains hard to beat; the GA is the slowest per unit of quality.

## References
- Kirkpatrick, Gelatt, Vecchi. *Optimization by Simulated Annealing*, Science 220, 1983.
- Glover. *Tabu Search – Part I*, ORSA Journal on Computing 1(3), 1989.
- Davis. *Applying Adaptive Algorithms to Epistatic Domains*, IJCAI 1985 (order crossover).
- Whitley, Starkweather, Fuquay. *Scheduling Problems and Traveling Salesmen: The Genetic Edge Recombination Operator*, ICGA 1989.
- Held, Karp. *A Dynamic Programming Approach to Sequencing Problems*, 1962.
- Croes. *A Method for Solving Traveling-Salesman Problems*, Operations Research 1958 (2-opt).

---

**Author:** Dr. Kainat Rizwan
