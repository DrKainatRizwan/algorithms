"""Demo: planning speeds up learning; Dyna-Q+ adapts to a blocked maze."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from model_based_rl import (DynaQ, DynaQPlus, GridWorld, PrioritizedSweeping,
                            QLearning, blocking_maze_layouts, run_episodes,
                            run_steps, value_iteration)
from model_based_rl.envs import dyna_maze


def main() -> None:
    env = GridWorld(dyna_maze())
    P, R, term = env.transition_model()
    V, Q, sweeps = value_iteration(P, R, term)
    print(env.render(Q.argmax(axis=1)))
    print(f"value iteration: {sweeps} sweeps, V(start)={V[env.start]:.4f}\n")

    print("Mean steps per episode (first 50 episodes, 10 seeds):")
    for name, mk in [
        ("Q-learning", lambda sd: QLearning(env.n_states, 4, alpha=0.5, seed=sd)),
        ("Dyna-Q n=5", lambda sd: DynaQ(env.n_states, 4, alpha=0.5, seed=sd, n_planning=5)),
        ("Dyna-Q n=50", lambda sd: DynaQ(env.n_states, 4, alpha=0.5, seed=sd, n_planning=50)),
        ("PrioSweep n=5", lambda sd: PrioritizedSweeping(env.n_states, 4, alpha=0.5, seed=sd, n_planning=5)),
    ]:
        t0 = time.time()
        curves = [run_episodes(GridWorld(dyna_maze()), mk(sd), 50, seed=sd) for sd in range(10)]
        c = np.mean(curves, axis=0)
        print(f"  {name:14s} ep1={c[0]:7.1f} ep2={c[1]:7.1f} ep5={c[4]:6.1f} "
              f"mean(1-50)={c.mean():6.1f} last10={c[-10:].mean():5.1f} ({time.time()-t0:.1f}s)")

    print("\nBlocking maze (cumulative reward at 3000 steps, wall switches at 1000):")
    before, after = blocking_maze_layouts()
    for name, cls, kw in [("Dyna-Q", DynaQ, {}), ("Dyna-Q+", DynaQPlus, {"kappa": 1e-3})]:
        finals = []
        for sd in range(10):
            e = GridWorld(before)
            ag = cls(e.n_states, 4, alpha=1.0, epsilon=0.1, seed=sd, n_planning=10, **kw)
            cum = run_steps(e, ag, 3000, seed=sd, switch_at=1000,
                            on_switch=lambda en: en.set_layout(after))
            finals.append((cum[999], cum[-1]))
        f = np.mean(finals, axis=0)
        print(f"  {name:8s} reward@1000={f[0]:6.1f}  reward@3000={f[1]:6.1f}")


if __name__ == "__main__":
    main()
