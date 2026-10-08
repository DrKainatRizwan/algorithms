# Deep Q-Learning from Scratch in NumPy with Experience Replay on Gridworld Mazes

## Overview
A compact implementation of Deep Q-Networks (DQN) using only NumPy: a hand-written MLP with backpropagation, an Adam optimiser, uniform and prioritised experience replay, a target network (hard or Polyak updates), Double-DQN targets, and a randomly generated maze environment with a BFS oracle for the optimal path length.

## How it works
- **Environment** (`maze.py`): mazes come from randomised depth-first search (optionally with extra openings to create loops). The agent sees a one-hot cell encoding; rewards are +1 at the goal, -0.01 per step and -0.05 for bumping a wall.
- **Network** (`network.py`): ReLU MLP, analytic gradients (verified against finite differences in the tests), Adam with global-norm clipping.
- **Replay** (`replay.py`): circular buffer for uniform sampling; the prioritised variant samples with probability proportional to `|TD|^alpha` and corrects with importance weights.
- **Agent** (`agent.py`): epsilon-greedy with linear decay; regression target `y = r + γ (1-d) Q_target(s', argmax_a Q(s', a))` for Double DQN (or `max_a Q_target` for vanilla); Huber loss.
- **Training** (`train.py`): episode loop, greedy evaluation and path extraction.

## Complexity
- Maze generation: O(n) in cells; BFS: O(n).
- One learning step: O(B · P), with batch size B and P network parameters; replay memory O(C · d) for capacity C and observation size d. Prioritised sampling here is O(C) per batch (a sum-tree would give O(B log C)).

## Usage
```python
from deep_qlearning import *

maze = generate_maze(4, seed=3, extra_openings=2)
env = MazeEnv(maze, max_steps=150)
agent = DQNAgent(env.obs_dim, env.n_actions, DQNConfig(seed=1, eps_decay_steps=4000))
train(env, agent, episodes=150)
print(evaluate(env, agent), shortest_path_length(maze))
```
Run `python examples/demo.py` for the demo and `python -m pytest -q` for the tests.

## Results
9x9 grid maze (4x4 cells, 2 extra openings, optimal path = 16 steps), 150 training episodes, seed 1:

| Variant | Greedy success | Greedy steps | Mean return (last 20 eps) |
|---|---|---|---|
| Vanilla DQN | yes | 16 | +0.837 |
| Double DQN | yes | 16 | +0.835 |
| Prioritised replay | yes | 16 | +0.834 |

All variants recover a shortest path, each in roughly 2-3 s on a single CPU core.

## References
- Mnih et al., "Human-level control through deep reinforcement learning", Nature, 2015.
- van Hasselt, Guez, Silver, "Deep Reinforcement Learning with Double Q-learning", AAAI 2016.
- Schaul et al., "Prioritized Experience Replay", ICLR 2016.
- Lin, "Self-improving reactive agents based on reinforcement learning, planning and teaching", 1992.
- Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed.

---

**Author:** Dr. Kainat Rizwan
