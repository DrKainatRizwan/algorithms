"""Train DQN agents on a random maze and compare variants."""
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from deep_qlearning import (DQNAgent, DQNConfig, MazeEnv, generate_maze,
                            shortest_path_length, train, evaluate, greedy_path)


def run(maze, name, episodes=150, **kw):
    env = MazeEnv(maze, max_steps=150)
    agent = DQNAgent(env.obs_dim, env.n_actions, DQNConfig(seed=1, **kw))
    t = time.time()
    hist = train(env, agent, episodes)
    ev = evaluate(env, agent)
    print(f"{name:<14} success={ev['success']:.0f} steps={ev['steps']:.0f} "
          f"last-20 return={np.mean(hist['returns'][-20:]):+.3f} ({time.time()-t:.1f}s)")
    return env, agent


maze = generate_maze(4, seed=3, extra_openings=2)
print(maze.render())
opt = shortest_path_length(maze)
print(f"\nBFS optimal path length: {opt}\n")
kw = dict(eps_decay_steps=4000)
run(maze, "vanilla DQN", double=False, **kw)
env, agent = run(maze, "double DQN", double=True, **kw)
run(maze, "prioritized", prioritized=True, **kw)
path, ok = greedy_path(env, agent)
print("\nLearned greedy route (double DQN):")
print(maze.render(path))
