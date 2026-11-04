"""Training and evaluation loops for independent and joint learners."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .agents import EpsilonSchedule, IndependentQLearner, JointQLearner
from .env import CoopGridWorld


@dataclass
class TrainingLog:
    returns: list[float] = field(default_factory=list)
    successes: list[bool] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)
    collisions: list[int] = field(default_factory=list)

    def success_rate(self, last: int | None = None) -> float:
        s = self.successes[-last:] if last else self.successes
        return float(np.mean(s)) if s else 0.0


def train(env: CoopGridWorld, agents: list[IndependentQLearner] | JointQLearner,
          episodes: int, obs_mode: str = "local",
          schedule: EpsilonSchedule | None = None) -> TrainingLog:
    """Run ``episodes`` of epsilon-greedy training and return the log.

    ``agents`` is either a list of independent learners (one per agent) or a
    single :class:`JointQLearner` acting on the global observation.
    """
    schedule = schedule or EpsilonSchedule(steps=int(episodes * 0.7))
    joint = isinstance(agents, JointQLearner)
    log = TrainingLog()
    for ep in range(episodes):
        eps = schedule(ep)
        env.reset()
        total, done = 0.0, False
        while not done:
            if joint:
                obs = env.observe(0, "global")
                acts = agents.act(obs, eps)
            else:
                obs_all = [env.observe(i, obs_mode) for i in range(env.n_agents)]
                was_active = list(env.active)
                acts = [agents[i].act(obs_all[i], eps) if was_active[i] else 4
                        for i in range(env.n_agents)]
            _, r, done, info = env.step(acts)
            total += r
            terminal = info["success"]  # time-limit truncation bootstraps
            if joint:
                agents.update(obs, acts, r, env.observe(0, "global"), terminal)
            else:
                for i in range(env.n_agents):
                    if was_active[i]:
                        nobs = env.observe(i, obs_mode)
                        agents[i].update(obs_all[i], acts[i], r, nobs,
                                         terminal or not env.active[i])
        log.returns.append(total)
        log.successes.append(info["success"])
        log.lengths.append(env.t)
        log.collisions.append(env.collisions)
    return log


def evaluate(env: CoopGridWorld, agents: list[IndependentQLearner] | JointQLearner,
             episodes: int = 1, obs_mode: str = "local") -> dict[str, float]:
    """Greedy rollouts. The layout is deterministic, so one episode is exact
    unless ties are broken randomly; use several for a stable mean."""
    joint = isinstance(agents, JointQLearner)
    succ, steps, coll, rets = [], [], [], []
    for _ in range(episodes):
        env.reset()
        done, total = False, 0.0
        while not done:
            if joint:
                acts = agents.act(env.observe(0, "global"), 0.0)
            else:
                acts = [agents[i].greedy(env.observe(i, obs_mode)) if env.active[i]
                        else 4 for i in range(env.n_agents)]
            _, r, done, info = env.step(acts)
            total += r
        succ.append(info["success"])
        steps.append(env.t)
        coll.append(env.collisions)
        rets.append(total)
    return {"success": float(np.mean(succ)), "steps": float(np.mean(steps)),
            "collisions": float(np.mean(coll)), "return": float(np.mean(rets))}
