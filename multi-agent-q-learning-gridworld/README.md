# Multi-Agent Reinforcement Learning: Independent Q-Learners in a Cooperative Gridworld

## Overview
A small, dependency-light testbed for decentralised cooperative learning. Several agents share one team reward while moving simultaneously on a grid with walls. Each agent runs its own tabular Q-learner (optionally hysteretic) and must discover coordination without communication. A centralised joint-action learner provides a reference upper bound.

Modules in `marl_gridworld/`:
- `env.py` – ASCII layouts, simultaneous moves, fixed-point collision resolution (same-cell, swaps, blocked chains), three observation models.
- `agents.py` – `IndependentQLearner` (standard and hysteretic), `JointQLearner`, `EpsilonSchedule`.
- `trainer.py` – training and greedy-evaluation loops.
- `scenarios.py` – `crossing`, `bay` (corridor swap with one passing bay) and `triangle`.
- `metrics.py` – BFS shortest paths, makespan lower bound, moving averages.

## How it works
Each step all agents pick one of {up, down, left, right, stay}. Moves into walls fail; agents targeting the same cell, swapping cells, or entering a cell held by a blocked agent are all blocked (computed by iterating to a fixed point). The team reward is `-0.05` per active agent per step, `+1` per arrival, `+3` when everyone has arrived and `-0.3` per blocked agent. Arrived agents leave the board.

Independent learners update `Q(o,a) += lr * (r + γ max Q(o',·) - Q(o,a))` using only their own observation. Because teammates are learning too, the environment looks non-stationary. Three remedies are compared:
1. **own** – the agent sees only its cell (no coordination possible in the `bay`).
2. **local** – plus a 4-bit mask of adjacent agents, enough to learn who yields.
3. **hysteretic** – negative TD errors use a smaller rate `β < α`, so a teammate's exploratory slips are forgiven (Matignon et al., 2007).
4. **joint** – one Q-table over the global state and the product of all actions.

Time-limit truncation bootstraps (only true success terminates the target).

## Complexity
- Environment step: O(n²) per resolution round, at most n rounds (n agents).
- Independent learners: O(n·|A|) per step; memory O(n · |O| · |A|) with |O| ≤ cells·2⁴.
- Joint learner: memory and action selection O(|S|ⁿ-ish states · |A|ⁿ) – exponential in n, which is why it is only a baseline.

## Usage
```python
from marl_gridworld import make_env, IndependentQLearner, train, evaluate

env = make_env("bay")
agents = [IndependentQLearner(5, seed=i) for i in range(env.n_agents)]
log = train(env, agents, episodes=1200, obs_mode="local")
print(evaluate(env, agents, episodes=3, obs_mode="local"))
```
Run `python examples/demo.py` for the comparison and `python -m pytest -q` for the 22 tests.

## Results
`python examples/demo.py` (1200 training episodes, mean of 2 seeds; train-succ is the last 200 training episodes with ε-exploration, the rest is greedy):

| scenario | learner | train-succ | greedy-succ | steps | collisions |
|---|---|---|---|---|---|
| crossing (bound 4) | independent/own | 0.97 | 1.00 | 6.0 | 0.00 |
| crossing | independent/local | 0.84 | 0.50 | 32.0 | 0.00 |
| crossing | hysteretic/local | 0.63 | 0.00 | 60.0 | 0.00 |
| crossing | joint | 1.00 | 1.00 | 7.0 | 0.00 |
| bay (bound 9) | independent/own | 0.00 | 0.00 | 60.0 | 0.00 |
| bay | independent/local | 0.96 | 1.00 | 13.5 | 0.00 |
| bay | hysteretic/local | 0.92 | 0.50 | 37.5 | 0.00 |
| bay | joint | 1.00 | 1.00 | 16.0 | 0.00 |

Observations: blind agents solve the open crossing but never the bay, where passing needs a yield; a neighbour mask fixes that. Some greedy policies stall in two-agent oscillations (greedy-succ 0.5 means one of two seeds), a known failure of independent learners; the hysteretic rate chosen here (β=0.1) was not uniformly better than plain updates with only 1200 episodes. The joint learner is the most reliable but scales exponentially.

## References
- Tan, M. (1993). Multi-agent reinforcement learning: independent vs. cooperative agents. ICML.
- Claus, C. & Boutilier, C. (1998). The dynamics of reinforcement learning in cooperative multiagent systems. AAAI.
- Matignon, L., Laurent, G. & Le Fort-Piat, N. (2007). Hysteretic Q-learning. IROS.
- Busoniu, L., Babuska, R. & De Schutter, B. (2008). A comprehensive survey of multiagent reinforcement learning. IEEE Trans. SMC-C.

---

**Author:** Dr. Kainat Rizwan
