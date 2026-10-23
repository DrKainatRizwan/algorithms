"""Rollout storage and Generalised Advantage Estimation."""

from __future__ import annotations

import numpy as np


def compute_gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray,
                last_value: np.ndarray, gamma: float = 0.99, lam: float = 0.95
                ) -> tuple[np.ndarray, np.ndarray]:
    """GAE(lambda) over arrays of shape ``(T, n_envs)``.

    ``dones[t]`` marks that the transition at ``t`` ended an episode, so no
    bootstrapping happens across it. Returns ``(advantages, returns)``.
    """
    T = rewards.shape[0]
    adv = np.zeros_like(rewards, dtype=float)
    last = np.zeros_like(last_value, dtype=float)
    for t in reversed(range(T)):
        nxt = last_value if t == T - 1 else values[t + 1]
        nonterm = 1.0 - dones[t]
        delta = rewards[t] + gamma * nxt * nonterm - values[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
    return adv, adv + values


class RolloutBuffer:
    """Fixed-size on-policy buffer for ``T`` steps of ``n_envs`` environments."""

    def __init__(self, T: int, n_envs: int, obs_dim: int, act_dim: int) -> None:
        self.T, self.n_envs = T, n_envs
        self.obs = np.zeros((T, n_envs, obs_dim))
        self.act = np.zeros((T, n_envs, act_dim))
        self.logp = np.zeros((T, n_envs))
        self.rew = np.zeros((T, n_envs))
        self.done = np.zeros((T, n_envs))
        self.val = np.zeros((T, n_envs))
        self.ptr = 0
        self.adv: np.ndarray | None = None
        self.ret: np.ndarray | None = None

    def add(self, obs, act, logp, rew, done, val) -> None:
        i = self.ptr
        self.obs[i], self.act[i], self.logp[i] = obs, act, logp
        self.rew[i], self.done[i], self.val[i] = rew, done, val
        self.ptr += 1

    def finish(self, last_value: np.ndarray, gamma: float, lam: float) -> None:
        self.adv, self.ret = compute_gae(self.rew, self.val, self.done, last_value, gamma, lam)

    def minibatches(self, n_batches: int, rng: np.random.Generator, normalize_adv: bool = True):
        """Yield shuffled minibatch dicts covering every sample exactly once."""
        assert self.adv is not None and self.ret is not None, "call finish() first"
        N = self.T * self.n_envs
        flat = lambda x: x.reshape(N, *x.shape[2:])
        adv = flat(self.adv)
        if normalize_adv:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        data = dict(obs=flat(self.obs), act=flat(self.act), logp=flat(self.logp),
                    adv=adv, ret=flat(self.ret), val=flat(self.val))
        perm = rng.permutation(N)
        for idx in np.array_split(perm, n_batches):
            yield {k: v[idx] for k, v in data.items()}
