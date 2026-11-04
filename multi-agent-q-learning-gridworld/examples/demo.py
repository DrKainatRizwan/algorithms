"""Compare observation models and learners on the coordination scenarios."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from marl_gridworld import (IndependentQLearner, JointQLearner, evaluate,
                            make_env, optimal_makespan, train)

EPISODES, SEEDS = 1200, (0, 1)
CONFIGS = [("independent/own", "own", None), ("independent/local", "local", None),
           ("hysteretic/local", "local", 0.1), ("joint (central)", "global", "joint")]


def run(scn: str, mode: str, opt, seed: int):
    env = make_env(scn)
    if opt == "joint":
        ag = JointQLearner(env.n_agents, env.n_actions, seed=seed)
    else:
        ag = [IndependentQLearner(5, beta=opt, seed=100 * seed + i)
              for i in range(env.n_agents)]
    log = train(env, ag, EPISODES, obs_mode=mode)
    ev = evaluate(env, ag, 3, mode)
    return log.success_rate(200), ev


t0 = time.time()
for scn in ("crossing", "bay"):
    print(f"\n== {scn} (lower bound {optimal_makespan(make_env(scn))} steps) ==")
    print(f"{'learner':20s} train-succ  greedy-succ  steps  collisions")
    for name, mode, opt in CONFIGS:
        rs = [run(scn, mode, opt, s) for s in SEEDS]
        ts = np.mean([r[0] for r in rs])
        gs = np.mean([r[1]["success"] for r in rs])
        st = np.mean([r[1]["steps"] for r in rs])
        co = np.mean([r[1]["collisions"] for r in rs])
        print(f"{name:20s} {ts:10.2f} {gs:12.2f} {st:6.1f} {co:11.2f}")
print(f"\nelapsed {time.time() - t0:.1f}s")
