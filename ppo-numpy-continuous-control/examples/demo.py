"""Train PPO on three toy control tasks and report returns before and after."""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ppo_control import (PendulumEnv, PointMassEnv, PPOAgent, PPOConfig, ReachEnv2D,
                         VecEnv, GaussianPolicy, evaluate, gradient_check)


def run(name, cls, iters, cfg_kw=None, env_kw=None):
    env_kw = env_kw or {}
    t0 = time.time()
    venv = VecEnv(lambda s: cls(s, **env_kw), 8, seed=0)
    agent = PPOAgent(venv, PPOConfig(seed=0, **(cfg_kw or {})))
    zero = evaluate(cls(1234, **env_kw), lambda o: np.zeros(cls.act_dim), 20)
    rand_rng = np.random.default_rng(0)
    rand = evaluate(cls(1234, **env_kw), lambda o: rand_rng.uniform(-1, 1, cls.act_dim), 20)
    print(f"\n== {name} ==")
    agent.train(iters, log_every=max(1, iters // 5))
    ret = evaluate(cls(1234, **env_kw), agent.act, 20)
    print(f"zero-action {zero:9.2f} | random {rand:9.2f} | PPO {ret:9.2f} | {time.time() - t0:.1f}s")
    return ret


def main():
    rng = np.random.default_rng(0)
    pol = GaussianPolicy(4, 2, (8,), rng)
    obs, act = rng.standard_normal((16, 4)), rng.standard_normal((16, 2))
    old, adv = pol.log_prob(obs, act) + 0.02, rng.standard_normal(16)

    def loss_and_grads():
        loss, grads, _, _ = pol.ppo_loss_grads(obs, act, old, adv, 0.2, 0.01)
        return loss, grads

    err = gradient_check(loss_and_grads, pol.params)
    print(f"policy-gradient check: max relative error {err:.2e}")
    run("PointMass (1-D double integrator)", PointMassEnv, 40)
    run("Reach2D (target reaching)", ReachEnv2D, 70)
    run("Pendulum balancing (start within 0.8 rad)", PendulumEnv, 80,
        dict(rollout_len=128, gamma=0.9, lr_pi=1e-3, epochs=10), dict(start_range=0.8))


if __name__ == "__main__":
    main()
