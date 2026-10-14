# Policy Gradient Methods: REINFORCE and A2C in NumPy

## Overview
From-scratch implementations of two policy-gradient algorithms for CartPole-style control, using only NumPy: Monte-Carlo REINFORCE (with no baseline, a normalised-return baseline, or a learned value baseline) and synchronous advantage actor-critic (A2C) with generalised advantage estimation. Networks, backpropagation and Adam are hand-written and verified against finite differences. The package includes a CartPole physics simulator, a corridor sanity-check task and a vectorised environment wrapper with auto-reset.

## How it works
The policy gradient theorem gives ∇J(θ) = E[Σ_t ∇log π(a_t|s_t) · Ψ_t]. 

- **REINFORCE** uses Ψ_t = G_t − b(s_t), the discounted reward-to-go minus a baseline. Subtracting a baseline leaves the gradient unbiased but reduces variance substantially.
- **A2C** replaces G_t with the GAE(λ) advantage Â_t = Σ_l (γλ)^l δ_{t+l}, δ_t = r_t + γV(s_{t+1}) − V(s_t), computed over n-step rollouts from several parallel environments. The critic regresses onto Â_t + V(s_t). Time-limit truncation is bootstrapped from V(final state); true termination cuts the bootstrap.
- Both add an optional entropy bonus; the gradient of loss wrt logits is analytic: `-A(onehot - p) + c·p(log p + H)`.

## Complexity
Per update with batch size B, hidden width H, input d and A actions: time O(B·H·(d + A)) forward and backward, memory O(B·H) activations plus O(H(d+A)) parameters. GAE is O(T·N) for T steps in N environments.

## Usage
```python
from policy_gradient import A2C, CartPole, VectorEnv, Reinforce, evaluate_policy

agent = Reinforce(4, 2, hidden=32, lr=0.01, baseline="value", seed=0)
agent.train(CartPole(seed=0), episodes=250)
print(evaluate_policy(agent, CartPole(), episodes=5))

venv = VectorEnv(lambda s: CartPole(seed=s), n_envs=8)
a2c = A2C(4, 2, seed=0)
returns = a2c.train(venv, updates=300, n_steps=16)
```

Run `python examples/demo.py` for the benchmark and `python -m pytest -q` for the tests.

## Results
CartPole with a 200-step limit (maximum return 200); last-30-episode training return and greedy evaluation over 5 episodes, from `examples/demo.py`:

| Method | Train return | Greedy eval |
|--------|-------------:|------------:|
| REINFORCE, no baseline (2 seeds, 250 eps) | 104.0 | 104.5 |
| REINFORCE, normalised returns (2 seeds, 250 eps) | 200.0 | 200.0 |
| REINFORCE, value baseline (2 seeds, 250 eps) | 200.0 | 200.0 |
| A2C, 8 envs (3 seeds, 300 updates) | 121.0 | 139.3 |

Baselines clearly help REINFORCE. A2C is more seed-sensitive at this small budget (one seed reaches 200, others 106–112) and uses far more environment steps per update.

## References
- Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine Learning*, 8.
- Sutton, McAllester, Singh, Mansour (2000). Policy gradient methods for reinforcement learning with function approximation. *NeurIPS*.
- Mnih et al. (2016). Asynchronous methods for deep reinforcement learning. *ICML*.
- Schulman et al. (2016). High-dimensional continuous control using generalized advantage estimation. *ICLR*.
- Sutton & Barto (2018). *Reinforcement Learning: An Introduction*, 2nd ed.

---

**Author:** Dr. Kainat Rizwan
