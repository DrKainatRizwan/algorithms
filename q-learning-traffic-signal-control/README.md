# Q-Learning Traffic Signal Controller on a Simulated Intersection Network

## Overview

A from-scratch study of reinforcement-learning traffic signal control. The package contains a
queue-based microscopic simulator of a grid of signalised intersections (saturation flow, lost
time at phase changes, minimum green, finite storage with spillback, turning movements) and
tabular temporal-difference agents (Q-learning, Double Q-learning, Expected SARSA) that learn
a shared switching policy for every intersection. Learned policies are compared with fixed-time,
longest-queue-first and max-pressure controllers. Only the standard library and NumPy are used.

## How it works

* **Network** (`traffic_rl/network.py`): each intersection has four single-lane approaches and two
  phases (NS, EW). Green approaches discharge at 0.5 veh/s; vehicles turn with probability 0.2,
  join the downstream queue one second later or leave the grid. A full downstream link blocks
  discharge (spillback). A switch costs 2 s of yellow with no discharge and is ignored before
  6 s of green. Decisions are taken every 5 s.
* **Demand** (`traffic_rl/demand.py`): Poisson arrivals at the boundary; uniform, asymmetric and
  alternating "rush hour" profiles.
* **State**: current phase, whether a switch is allowed, and binned NS / EW queue totals
  (144 discrete states). **Actions**: keep / switch.
  **Reward**: minus the mean local queue length over the decision interval.
* **Agents** (`traffic_rl/agents.py`): one Q table shared by all intersections (parameter sharing,
  independent learners). Updates:
  * Q-learning: `Q(s,a) += α (r + γ max Q(s',·) − Q(s,a))`
  * Double Q-learning: two tables, one chooses and the other evaluates the next action
  * Expected SARSA: expectation under the ε-greedy policy
  * Exploration decays exponentially per episode.
* **Controllers** (`traffic_rl/controllers.py`): fixed-time, longest-queue-first, max-pressure
  (upstream minus downstream queue), random and the learned table policy.

## Complexity

* Simulation: O(T · N) per episode for N intersections and T seconds (arrivals and discharge
  are constant work per approach); memory O(N).
* Learning: O(1) per TD update, so O(E · S · N) per training run for E episodes and S decisions;
  the Q table needs O(|states| · |actions|) = 144 × 2 entries, independent of network size.

## Usage

```python
from traffic_rl import TrafficGrid, rush_hour_demand, train, make_policy, evaluate, MaxPressure

env = TrafficGrid(2, 2, rush_hour_demand(), seed=0)
agent, history = train(env, steps=200, episodes=100, algo="q", seed=1)
policy = make_policy(agent)
print(evaluate(env, policy, 240, seeds=[900, 901]))
print(evaluate(env, MaxPressure(), 240, seeds=[900, 901]))
```

Run `python examples/demo.py` for the full comparison and `python -m pytest -q` for the tests.

## Results

Mean number of queued vehicles over a 1200 s run, averaged over 4 unseen traffic seeds
(lower is better; 100 training episodes of 1000 s each):

| Scenario | Fixed-time | Longest-queue | Max-pressure | Random | Q-learning |
|----------|-----------:|--------------:|-------------:|-------:|-----------:|
| 1 intersection, rush hour | 6.33 | 2.29 | 2.61 | 19.26 | 2.65 |
| 2x2 grid, asymmetric | 17.73 | 10.06 | 13.60 | 40.54 | 10.71 |
| 2x2 grid, uniform | 19.83 | 13.88 | 17.10 | 58.42 | 14.65 |

On the single intersection, Expected SARSA reached 2.56 and Double Q-learning 4.14 with the same
budget. The learned policy cuts the queue by 40–58% relative to fixed timing and is competitive
with the hand-designed heuristics; on the grid it improves on max-pressure, whose local rule
ignores lost time from frequent switching. Results depend on seeds and training budget.

## References

* Watkins & Dayan, "Q-learning", *Machine Learning* 8, 1992.
* van Hasselt, "Double Q-learning", NIPS 2010.
* van Seijen et al., "A theoretical and empirical analysis of Expected SARSA", ADPRL 2009.
* Varaiya, "Max pressure control of a network of signalized intersections", *Transportation Research Part C* 36, 2013.
* Wei et al., "A survey on traffic signal control methods", arXiv:1904.08117, 2019.
* Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed., 2018.

---

**Author:** Dr. Kainat Rizwan
