"""Training and evaluation loops."""
from __future__ import annotations

import numpy as np

from .agent import DQNAgent
from .maze import MazeEnv


def train(env: MazeEnv, agent: DQNAgent, episodes: int) -> dict[str, list[float]]:
    """Run ``episodes`` episodes; returns per-episode return, length and mean loss."""
    returns, lengths, losses = [], [], []
    for _ in range(episodes):
        s = env.reset()
        done, total, n, ls = False, 0.0, 0, []
        while not done:
            a = agent.act(s)
            s2, r, done = env.step(a)
            loss = agent.observe(s, a, r, s2, done)
            if loss is not None:
                ls.append(loss)
            s, total, n = s2, total + r, n + 1
        returns.append(total)
        lengths.append(float(n))
        losses.append(float(np.mean(ls)) if ls else float("nan"))
    return {"returns": returns, "lengths": lengths, "losses": losses}


def greedy_path(env: MazeEnv, agent: DQNAgent) -> tuple[list[tuple[int, int]], bool]:
    """Follow the greedy policy; returns visited cells and whether the goal was reached."""
    s = env.reset()
    path = [env.pos]
    done = False
    while not done:
        s, _, done = env.step(agent.act(s, greedy=True))
        path.append(env.pos)
    return path, env.pos == env.maze.goal


def evaluate(env: MazeEnv, agent: DQNAgent) -> dict[str, float]:
    """Greedy rollout statistics (the maze is deterministic, so one rollout suffices)."""
    path, ok = greedy_path(env, agent)
    return {"success": float(ok), "steps": float(len(path) - 1)}
