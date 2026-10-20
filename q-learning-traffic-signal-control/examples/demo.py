"""Train tabular signal controllers and compare them with classical baselines."""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from traffic_rl import (FixedTime, LongestQueueFirst, MaxPressure, RandomPolicy, TrafficGrid,  # noqa: E402
                        asymmetric_demand, evaluate, make_policy, rush_hour_demand, train,
                        uniform_demand)

STEPS = 240
SEEDS = [900, 901, 902, 903]


def compare(title, env, algos=("q",)):
    t0 = time.time()
    rows = [("fixed-time (20s)", FixedTime(20)), ("longest-queue", LongestQueueFirst()),
            ("max-pressure", MaxPressure()), ("random", RandomPolicy())]
    for algo in algos:
        agent, hist = train(env, steps=200, episodes=100, algo=algo, seed=1)
        rows.append((f"{algo} (trained)", make_policy(agent)))
        print(f"  [{algo}] training queue: first 10 eps {sum(hist.mean_queue[:10]) / 10:.2f} -> "
              f"last 10 eps {sum(hist.mean_queue[-10:]) / 10:.2f}")
    print(f"\n{title}")
    print(f"  {'controller':<18}{'mean queue':>11}{'throughput':>12}{'rejected':>10}{'switches':>10}")
    for name, ctl in rows:
        m = evaluate(env, ctl, STEPS, SEEDS)
        print(f"  {name:<18}{m['mean_queue']:>11.2f}{m['throughput']:>12.0f}"
              f"{m['rejected']:>10.0f}{m['switches']:>10.1f}")
    print(f"  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    compare("Single intersection, rush-hour demand", TrafficGrid(1, 1, rush_hour_demand()),
            algos=("q", "double_q", "expected_sarsa"))
    compare("2x2 grid, asymmetric demand", TrafficGrid(2, 2, asymmetric_demand()))
    compare("2x2 grid, uniform demand", TrafficGrid(2, 2, uniform_demand(0.14)))
