# Model-Based Reinforcement Learning: Dyna-Q, Dyna-Q+ and Prioritized Sweeping

## Overview

From-scratch NumPy implementations of tabular model-based reinforcement learning on gridworlds:

- **Q-learning** (model-free baseline)
- **Dyna-Q** – interleaves real experience with simulated updates drawn from a learned model
- **Dyna-Q+** – adds an exploration bonus so the agent re-checks transitions it has not tried for a long time (non-stationary worlds)
- **Prioritized sweeping** – plans from a priority queue ordered by Bellman error, propagating value backwards through predecessors
- **Value / policy iteration** on the exact model, used as ground truth

Package layout (`model_based_rl/`): `envs.py` (gridworld, blocking/shortcut mazes, exact model export), `model.py` (last-outcome and count-based learned models), `agents.py`, `dp.py`, `experiments.py`.

## How it works

After each real step `(s, a, r, s')` Dyna-Q does a Q-learning update, stores the transition in a model and then performs `n` planning updates on random previously seen `(s, a)` pairs using model-sampled outcomes. Dyna-Q+ replaces the planning reward with `r + κ√τ`, where `τ` is the number of real steps since `(s, a)` was last tried; untried actions in visited states are given self-loop model entries so planning can "wonder" whether they changed.

Prioritized sweeping pushes `(s, a)` onto a max-heap with priority `|E[r + γ max Q(s')] − Q(s, a)|` whenever it exceeds `θ`. Each planning step pops the largest-error pair, performs a full expected backup, and re-prioritises every known predecessor of the updated state, so credit flows backwards from the goal in few updates.

## Complexity

Let `S` states, `A` actions, `n` planning steps per real step.

| Method | Time per real step | Space |
|---|---|---|
| Q-learning | O(A) | O(SA) |
| Dyna-Q / Dyna-Q+ | O(nA) | O(SA) model + table |
| Prioritized sweeping | O(n · (A + p log q)) (p predecessors, q queue size) | O(SA) + queue |
| Value iteration | O(S²A) per sweep (dense) | O(S²A) |
| Policy iteration | O(S³) per evaluation | O(S²A) |

## Usage

```python
from model_based_rl import GridWorld, DynaQ, run_episodes, value_iteration
from model_based_rl.envs import dyna_maze

env = GridWorld(dyna_maze())
agent = DynaQ(env.n_states, 4, alpha=0.5, n_planning=10, seed=0)
steps = run_episodes(env, agent, n_episodes=50, seed=0)

P, R, terminal = env.transition_model()
V, Q, sweeps = value_iteration(P, R, terminal, gamma=0.95)
print(steps[:5], V[env.start])
```

Run `python examples/demo.py` for the demo and `python -m pytest -q` for the tests.

## Results

Dyna maze (6×9, γ=0.95, α=0.5, ε=0.1), mean over 10 seeds; the optimal path has 10 steps (V(start) = 0.6302 by value iteration, 14 sweeps).

| Agent | Steps ep. 1 | Steps ep. 2 | Steps ep. 5 | Mean steps, ep. 1–50 |
|---|---|---|---|---|
| Q-learning | 296.8 | 287.9 | 165.9 | 60.1 |
| Dyna-Q (n=5) | 304.5 | 90.3 | 24.3 | 22.1 |
| Dyna-Q (n=50) | 224.1 | 37.4 | 16.8 | 19.6 |
| Prioritized sweeping (n=5) | 296.8 | 39.7 | 15.6 | 20.4 |

Blocking maze (wall gap moves at step 1000, α=1, ε=0.1, n=10, 10 seeds), cumulative reward:

| Agent | At step 1000 | At step 3000 |
|---|---|---|
| Dyna-Q | 28.3 | 76.0 |
| Dyna-Q+ (κ=0.001) | 56.5 | 135.8 |

Planning cuts the learning cost by roughly 3× in this maze, and prioritized sweeping matches Dyna-Q with ten times fewer planning updates per step. With ε-greedy exploration alone, the greedy path can settle a few steps above optimal, so the tests only require near-optimal length and never-overestimated values.

## References

- R. S. Sutton, "Integrated architectures for learning, planning, and reacting based on approximating dynamic programming", ICML 1990.
- R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed., MIT Press, 2018, ch. 8.
- A. W. Moore and C. G. Atkeson, "Prioritized sweeping: reinforcement learning with less data and less time", Machine Learning 13, 1993.
- R. A. Howard, *Dynamic Programming and Markov Processes*, MIT Press, 1960.

---

**Author:** Dr. Kainat Rizwan
