"""PPO-Clip agent with GAE, observation normalisation and KL early stopping."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .buffer import RolloutBuffer
from .envs import Env, VecEnv
from .nets import Adam, GaussianPolicy, ValueNet, clip_grad_norm
from .normalize import RunningMeanStd


@dataclass
class PPOConfig:
    n_envs: int = 8
    rollout_len: int = 64
    epochs: int = 8
    n_minibatches: int = 4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    lr_pi: float = 3e-4
    lr_vf: float = 1e-3
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.03
    hidden: tuple[int, ...] = (64, 64)
    normalize_obs: bool = True
    anneal_lr: bool = True
    seed: int = 0


@dataclass
class PPOAgent:
    """Actor-critic PPO agent operating on a :class:`VecEnv`."""

    venv: VecEnv
    cfg: PPOConfig = field(default_factory=PPOConfig)

    def __post_init__(self) -> None:
        c = self.cfg
        self.rng = np.random.default_rng(c.seed)
        od, ad = self.venv.obs_dim, self.venv.act_dim
        self.policy = GaussianPolicy(od, ad, c.hidden, np.random.default_rng(c.seed + 1))
        self.value = ValueNet(od, c.hidden, np.random.default_rng(c.seed + 2))
        self.opt_pi = Adam(self.policy.params, lr=c.lr_pi)
        self.opt_vf = Adam(self.value.params, lr=c.lr_vf)
        self.obs_rms = RunningMeanStd((od,))
        self.obs = self.venv.reset()
        self.obs_rms.update(self.obs)
        self.limit = self.venv.act_limit
        self.history: list[dict] = []
        self.total_steps = 0

    def _norm(self, obs: np.ndarray) -> np.ndarray:
        return self.obs_rms.normalize(obs) if self.cfg.normalize_obs else obs

    def act(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Action (clipped to the env limits) for one raw observation."""
        a, _ = self.policy.sample(self._norm(obs[None]), self.rng, deterministic)
        return np.clip(a[0], -self.limit, self.limit)

    def collect(self) -> RolloutBuffer:
        c = self.cfg
        buf = RolloutBuffer(c.rollout_len, self.venv.n, self.venv.obs_dim, self.venv.act_dim)
        for _ in range(c.rollout_len):
            nobs = self._norm(self.obs)
            act, logp = self.policy.sample(nobs, self.rng)
            val = self.value.predict(nobs)
            next_obs, rew, done = self.venv.step(np.clip(act, -self.limit, self.limit))
            buf.add(nobs, act, logp, rew, done.astype(float), val)
            if c.normalize_obs:
                self.obs_rms.update(next_obs)
            self.obs = next_obs
        # episodes here end by time-limit; we treat them as terminal (no bootstrap)
        buf.finish(self.value.predict(self._norm(self.obs)), c.gamma, c.lam)
        self.total_steps += c.rollout_len * self.venv.n
        return buf

    def update(self, buf: RolloutBuffer) -> dict:
        c = self.cfg
        stats = dict(pi_loss=0.0, vf_loss=0.0, kl=0.0, clip_frac=0.0, epochs_run=0)
        n_mb = 0
        for epoch in range(c.epochs):
            kls = []
            for mb in buf.minibatches(c.n_minibatches, self.rng):
                loss, g, kl, cf = self.policy.ppo_loss_grads(
                    mb["obs"], mb["act"], mb["logp"], mb["adv"], c.clip, c.ent_coef)
                clip_grad_norm(g, c.max_grad_norm)
                self.opt_pi.step(g)
                vl, vg = self.value.loss_grads(mb["obs"], mb["ret"], mb["val"], clip=None)
                clip_grad_norm(vg, c.max_grad_norm / max(c.vf_coef, 1e-8))
                self.opt_vf.step(vg)
                kls.append(kl)
                stats["pi_loss"] += loss
                stats["vf_loss"] += vl
                stats["clip_frac"] += cf
                n_mb += 1
            stats["kl"] = float(np.mean(kls))
            stats["epochs_run"] = epoch + 1
            if c.target_kl is not None and stats["kl"] > 1.5 * c.target_kl:
                break
        for k in ("pi_loss", "vf_loss", "clip_frac"):
            stats[k] /= max(n_mb, 1)
        return stats

    def train(self, iterations: int, log_every: int = 0) -> list[dict]:
        c = self.cfg
        for it in range(iterations):
            if c.anneal_lr:
                frac = 1.0 - it / iterations
                self.opt_pi.lr = c.lr_pi * max(frac, 0.1)
                self.opt_vf.lr = c.lr_vf * max(frac, 0.1)
            stats = self.update(self.collect())
            recent = self.venv.finished[-self.venv.n:]
            stats["iter"] = it
            stats["steps"] = self.total_steps
            stats["mean_return"] = float(np.mean(recent)) if recent else float("nan")
            self.history.append(stats)
            if log_every and (it % log_every == 0 or it == iterations - 1):
                print(f"iter {it:3d} steps {self.total_steps:6d} return {stats['mean_return']:9.2f} "
                      f"kl {stats['kl']:.4f} clip {stats['clip_frac']:.2f} "
                      f"std {np.exp(self.policy.log_std).mean():.3f}")
        return self.history


def evaluate(env: Env, policy_fn, episodes: int = 10) -> float:
    """Average undiscounted return of ``policy_fn(obs) -> action`` on ``env``."""
    total = 0.0
    for _ in range(episodes):
        obs, done = env.reset(), False
        while not done:
            obs, r, done, _ = env.step(policy_fn(obs))
            total += r
    return total / episodes
