"""REINFORCE (Williams, 1992) with optional learned baseline."""

from __future__ import annotations

import numpy as np

from .networks import Adam, MLP, log_softmax, softmax
from .returns import discounted_returns, normalize


def policy_gradient_logits(logits: np.ndarray, actions: np.ndarray, weights: np.ndarray,
                           ent_coef: float = 0.0) -> tuple[float, np.ndarray]:
    """Loss ``-mean(w * log pi(a|s)) - ent_coef * mean(H)`` and its gradient wrt logits."""
    n = len(actions)
    logp_all = log_softmax(logits)
    p = np.exp(logp_all)
    logp = logp_all[np.arange(n), actions]
    ent = -(p * logp_all).sum(axis=1)
    loss = float(-(weights * logp).mean() - ent_coef * ent.mean())
    onehot = np.zeros_like(p)
    onehot[np.arange(n), actions] = 1.0
    grad = -weights[:, None] * (onehot - p)
    grad += ent_coef * p * (logp_all + ent[:, None])
    return loss, grad / n


class Reinforce:
    """Monte-Carlo policy gradient agent.

    Parameters
    ----------
    baseline : ``"none"``, ``"mean"`` (normalised returns) or ``"value"``
        (a learned state-value network subtracted from the return).
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 32, gamma: float = 0.99,
                 lr: float = 0.01, baseline: str = "value", ent_coef: float = 0.0,
                 seed: int = 0) -> None:
        if baseline not in ("none", "mean", "value"):
            raise ValueError(f"unknown baseline {baseline!r}")
        self.rng = np.random.default_rng(seed)
        self.gamma, self.baseline, self.ent_coef = gamma, baseline, ent_coef
        self.policy = MLP([obs_dim, hidden, n_actions], self.rng, out_scale=0.1)
        self.policy_opt = Adam(self.policy.params, lr=lr, clip_norm=5.0)
        self.value = MLP([obs_dim, hidden, 1], self.rng)
        self.value_opt = Adam(self.value.params, lr=lr * 2, clip_norm=5.0)

    def probs(self, obs: np.ndarray) -> np.ndarray:
        return softmax(self.policy.forward(np.atleast_2d(obs)))

    def act(self, obs: np.ndarray, greedy: bool = False) -> int:
        p = self.probs(obs)[0]
        if greedy:
            return int(np.argmax(p))
        return int(self.rng.choice(len(p), p=p))

    def run_episode(self, env) -> tuple[list, list, list]:
        obs = env.reset()
        states, actions, rewards = [], [], []
        while True:
            a = self.act(obs)
            nxt, r, term, trunc = env.step(a)
            states.append(obs); actions.append(a); rewards.append(r)
            obs = nxt
            if term or trunc:
                return states, actions, rewards

    def update(self, states, actions, rewards) -> dict[str, float]:
        """One gradient step from a single episode."""
        s = np.array(states)
        a = np.array(actions)
        G = discounted_returns(np.array(rewards, dtype=float), self.gamma)
        info: dict[str, float] = {}
        if self.baseline == "value":
            v, acts = self.value.forward(s, cache=True)
            v = v[:, 0]
            scale = max(1.0, float(np.abs(G).max()))
            err = v - G / scale
            self.value_opt.step(self.value.backward(acts, (2 * err / len(G))[:, None]))
            weights = normalize(G - v * scale) if len(G) > 1 else G - v * scale
            info["value_loss"] = float((err ** 2).mean())
        elif self.baseline == "mean":
            weights = normalize(G) if len(G) > 1 else G
        else:
            weights = G
        logits, pacts = self.policy.forward(s, cache=True)
        loss, g = policy_gradient_logits(logits, a, weights, self.ent_coef)
        self.policy_opt.step(self.policy.backward(pacts, g))
        info["policy_loss"] = loss
        return info

    def train(self, env, episodes: int) -> list[float]:
        """Train for ``episodes`` episodes; returns the per-episode returns."""
        history = []
        for _ in range(episodes):
            s, a, r = self.run_episode(env)
            self.update(s, a, r)
            history.append(float(sum(r)))
        return history
