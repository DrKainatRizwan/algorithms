# Proximal Policy Optimization in Pure NumPy

## Overview

A from-scratch implementation of PPO-Clip (Schulman et al., 2017) for continuous
action spaces, using only NumPy. It includes hand-derived backpropagation for a
diagonal-Gaussian actor and a value critic, Generalised Advantage Estimation,
running observation normalisation, KL-based early stopping, gradient clipping,
and three small control environments (a 1-D double integrator, 2-D target
reaching, and an inverted pendulum). Every gradient is verified against finite
differences in the test suite.

## How it works

1. **Rollouts.** `n_envs` environment copies are stepped for `rollout_len` steps
   with the current stochastic policy `a ~ N(mu(s), exp(log_std)^2)`.
2. **GAE.** Advantages use `A_t = sum_k (gamma*lambda)^k delta_{t+k}` with
   `delta_t = r_t + gamma V(s_{t+1}) - V(s_t)`, computed backwards in O(T).
3. **Clipped surrogate.** With `r = pi_new(a|s) / pi_old(a|s)` the policy
   minimises `-E[min(r A, clip(r, 1-eps, 1+eps) A)] - c_ent * H`. Gradients flow
   only through samples where the unclipped term is the minimum; the analytic
   derivatives are `dlogp/dmu = (a-mu)/sigma^2` and
   `dlogp/dlog_sigma = ((a-mu)/sigma)^2 - 1`.
4. **Several epochs of minibatch Adam** over the same rollout, stopping early if
   the approximate KL exceeds `1.5 * target_kl`. The critic is regressed onto the
   GAE returns.
5. Observations are normalised with Chan's parallel running mean/variance.

Modules: `envs` (environments, `VecEnv`), `nets` (MLP, Adam, Gaussian policy,
value net, gradient check), `buffer` (rollouts, GAE, minibatches), `normalize`
(running statistics), `ppo` (agent, config, evaluation).

## Complexity

- Per iteration: `O(T * N * P)` for collection and `O(E * T * N * P)` for the
  update, where `T` is rollout length, `N` is number of envs, `E` is epochs and
  `P` is parameters of the networks (matrix-multiply cost per sample).
- GAE: `O(T * N)`. Memory: `O(T * N * (obs_dim + act_dim))` for the buffer plus
  `O(P)` for the networks and Adam moments.

## Usage

```python
import numpy as np
from ppo_control import PPOAgent, PPOConfig, ReachEnv2D, VecEnv, evaluate

venv = VecEnv(lambda seed: ReachEnv2D(seed), n_envs=8, seed=0)
agent = PPOAgent(venv, PPOConfig(seed=0))
agent.train(70, log_every=10)
print(evaluate(ReachEnv2D(1234), agent.act, episodes=20))
```

Run the demo with `python examples/demo.py` and the tests with `python -m pytest -q`.

## Results

Output of `examples/demo.py` (evaluation on 20 held-out episodes, mean undiscounted
return; higher is better). The policy-gradient finite-difference check reports a
maximum relative error of 4.1e-06.

| Task | Zero action | Random | PPO | Iterations | Wall time |
|------|------------:|-------:|----:|-----------:|----------:|
| PointMass (1-D double integrator) | -268.93 | -1523.16 | -12.87 | 40 | 1.5 s |
| Reach2D (target reaching) | -88.63 | -128.73 | 23.02 | 70 | 2.8 s |
| Pendulum balancing (start within 0.8 rad) | -819.54 | -971.57 | -96.14 | 80 | 5.8 s |

Swing-up from arbitrary angles is available through `PendulumEnv()` but needs a
much larger sample budget than the demo uses.

## References

- J. Schulman, F. Wolski, P. Dhariwal, A. Radford, O. Klimov, *Proximal Policy
  Optimization Algorithms*, arXiv:1707.06347, 2017.
- J. Schulman, P. Moritz, S. Levine, M. Jordan, P. Abbeel, *High-Dimensional
  Continuous Control Using Generalized Advantage Estimation*, ICLR 2016.
- L. Engstrom et al., *Implementation Matters in Deep Policy Gradients*, ICLR 2020.
- R. Sutton and A. Barto, *Reinforcement Learning: An Introduction*, 2nd ed., 2018.

---

**Author:** Dr. Kainat Rizwan
