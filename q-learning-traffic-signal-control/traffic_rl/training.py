"""Training and evaluation loops for independent learners with a shared Q table."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .agents import EpsilonSchedule, StateEncoder, TabularAgent
from .controllers import Controller, TablePolicy
from .network import TrafficGrid


@dataclass
class EpisodeResult:
    mean_queue: float
    throughput: int
    rejected: int
    switches: int


@dataclass
class TrainingHistory:
    returns: list[float] = field(default_factory=list)
    mean_queue: list[float] = field(default_factory=list)
    epsilons: list[float] = field(default_factory=list)


def run_controller(env: TrafficGrid, controller: Controller, steps: int, seed: int) -> EpisodeResult:
    """Run a fixed controller for ``steps`` decisions on a freshly reset environment."""
    env.reset(seed)
    controller.reset()
    for _ in range(steps):
        env.step(controller.act(env))
    return EpisodeResult(env.mean_queue(), env.exited, env.rejected, env.switches)


def train(env: TrafficGrid, steps: int = 200, episodes: int = 150, algo: str = "q",
          alpha: float = 0.1, gamma: float = 0.9, schedule: EpsilonSchedule | None = None,
          encoder: StateEncoder | None = None, seed: int = 0) -> tuple[TabularAgent, TrainingHistory]:
    """Train a single Q table shared by all intersections (parameter sharing).

    Each episode uses a different traffic seed derived from ``seed``.
    """
    encoder = encoder or StateEncoder()
    schedule = schedule or EpsilonSchedule(1.0, 0.02, 0.96)
    agent = TabularAgent(encoder.n_states, 2, algo=algo, alpha=alpha, gamma=gamma, seed=seed)
    hist = TrainingHistory()
    for ep in range(episodes):
        agent.epsilon = schedule.value(ep)
        env.reset(seed * 10_000 + ep)
        states = [encoder.encode(env.observe(i)) for i in range(env.n)]
        total = 0.0
        for _ in range(steps):
            acts = [agent.act(s) for s in states]
            rewards = env.step(acts)
            nxt = [encoder.encode(env.observe(i)) for i in range(env.n)]
            for i in range(env.n):
                agent.update(states[i], acts[i], float(rewards[i]), nxt[i])
            states = nxt
            total += float(rewards.sum())
        hist.returns.append(total / env.n)
        hist.mean_queue.append(env.mean_queue())
        hist.epsilons.append(agent.epsilon)
    return agent, hist


def make_policy(agent: TabularAgent, encoder: StateEncoder | None = None) -> TablePolicy:
    return TablePolicy(agent.q.copy(), encoder or StateEncoder(), name=f"{agent.algo}-learning")


def evaluate(env: TrafficGrid, controller: Controller, steps: int, seeds: list[int]) -> dict[str, float]:
    """Average a controller's metrics over several evaluation seeds."""
    res = [run_controller(env, controller, steps, s) for s in seeds]
    return {
        "mean_queue": float(np.mean([r.mean_queue for r in res])),
        "throughput": float(np.mean([r.throughput for r in res])),
        "rejected": float(np.mean([r.rejected for r in res])),
        "switches": float(np.mean([r.switches for r in res])),
    }
