"""Synchronous advantage actor-critic (A2C) with GAE over vectorised environments."""

from __future__ import annotations

import numpy as np

from .envs import VectorEnv
from .networks import Adam, MLP, softmax
from .reinforce import policy_gradient_logits
from .returns import gae


class A2C:
    """Actor-critic with n-step rollouts, GAE(lambda) and an entropy bonus.

    The actor and critic are separate networks. Time-limit truncation is
    handled by bootstrapping from the critic's value of the final observation,
    while true termination cuts the bootstrap.
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64, gamma: float = 0.99,
                 lam: float = 0.95, lr: float = 0.005, ent_coef: float = 0.01,
                 vf_coef: float = 0.5, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)
        self.gamma, self.lam, self.ent_coef, self.vf_coef = gamma, lam, ent_coef, vf_coef
        self.actor = MLP([obs_dim, hidden, n_actions], self.rng, out_scale=0.1)
        self.critic = MLP([obs_dim, hidden, 1], self.rng)
        self.actor_opt = Adam(self.actor.params, lr=lr, clip_norm=0.5)
        self.critic_opt = Adam(self.critic.params, lr=lr, clip_norm=0.5)

    def probs(self, obs: np.ndarray) -> np.ndarray:
        return softmax(self.actor.forward(np.atleast_2d(obs)))

    def value(self, obs: np.ndarray) -> np.ndarray:
        return self.critic.forward(np.atleast_2d(obs))[:, 0]

    def act(self, obs: np.ndarray, greedy: bool = False) -> int:
        p = self.probs(obs)[0]
        return int(np.argmax(p)) if greedy else int(self.rng.choice(len(p), p=p))

    def _sample(self, obs: np.ndarray) -> np.ndarray:
        p = self.probs(obs)
        u = self.rng.random((len(p), 1))
        return (u > np.cumsum(p, axis=1)).sum(axis=1).clip(0, p.shape[1] - 1)

    def collect(self, venv: VectorEnv, obs: np.ndarray, n_steps: int):
        """Roll out ``n_steps`` in every sub-environment."""
        S, A, R, D, V, NV = [], [], [], [], [], []
        for _ in range(n_steps):
            a = self._sample(obs)
            v = self.value(obs)
            nxt, r, term, trunc, final = venv.step(a)
            nv = self.value(nxt)
            if trunc.any():  # bootstrap through time-limit truncation
                nv = np.where(trunc, self.value(final), nv)
            S.append(obs); A.append(a); R.append(r); D.append(term.astype(float))
            V.append(v); NV.append(nv)
            # a truncated step restarts in a new episode: cut the GAE trace via a
            # zero-valued continuation by folding the bootstrap into the reward
            if trunc.any():
                R[-1] = r + self.gamma * np.where(trunc, nv, 0.0)
                NV[-1] = np.where(trunc, 0.0, nv)
                D[-1] = np.where(trunc, 1.0, D[-1])
            obs = nxt
        return obs, tuple(np.array(x) for x in (S, A, R, D, V, NV))

    def update(self, S, A, R, D, V, NV) -> dict[str, float]:
        adv, targets = gae(R, V, NV, D, self.gamma, self.lam)
        T, N = R.shape
        s = S.reshape(T * N, -1)
        a = A.reshape(-1)
        adv_flat = adv.reshape(-1)
        adv_n = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
        logits, pacts = self.actor.forward(s, cache=True)
        pl, g = policy_gradient_logits(logits, a, adv_n, self.ent_coef)
        self.actor_opt.step(self.actor.backward(pacts, g))
        v, vacts = self.critic.forward(s, cache=True)
        err = v[:, 0] - targets.reshape(-1)
        self.critic_opt.step(self.critic.backward(vacts, (self.vf_coef * 2 * err / len(err))[:, None]))
        return {"policy_loss": pl, "value_loss": float((err ** 2).mean())}

    def train(self, venv: VectorEnv, updates: int, n_steps: int = 16) -> list[float]:
        """Run ``updates`` rollout/update cycles; returns finished-episode returns."""
        obs = venv.reset()
        for _ in range(updates):
            obs, batch = self.collect(venv, obs, n_steps)
            self.update(*batch)
        return list(venv.finished_returns)
