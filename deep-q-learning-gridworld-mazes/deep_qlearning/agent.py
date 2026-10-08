"""DQN agent: target network, epsilon-greedy exploration, optional Double DQN."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .network import MLP, Adam
from .replay import PrioritizedReplayBuffer, ReplayBuffer


@dataclass
class DQNConfig:
    hidden: tuple[int, ...] = (64,)
    gamma: float = 0.97
    lr: float = 2e-3
    batch_size: int = 64
    buffer_size: int = 20000
    warmup: int = 200
    train_every: int = 1
    target_sync: int = 200
    tau: float | None = None          # if set, use Polyak averaging instead of hard sync
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 6000
    double: bool = True
    prioritized: bool = False
    grad_clip: float | None = 5.0
    seed: int = 0


class DQNAgent:
    def __init__(self, obs_dim: int, n_actions: int, cfg: DQNConfig | None = None) -> None:
        self.cfg = cfg or DQNConfig()
        c = self.cfg
        sizes = [obs_dim, *c.hidden, n_actions]
        self.q = MLP(sizes, seed=c.seed)
        self.target = MLP(sizes, seed=c.seed + 1)
        self.target.copy_from(self.q)
        self.opt = Adam(self.q.params, lr=c.lr, clip=c.grad_clip)
        buf_cls = PrioritizedReplayBuffer if c.prioritized else ReplayBuffer
        self.buffer = buf_cls(c.buffer_size, obs_dim, seed=c.seed + 2)
        self.rng = np.random.default_rng(c.seed + 3)
        self.n_actions = n_actions
        self.steps = 0
        self.updates = 0

    def epsilon(self) -> float:
        c = self.cfg
        frac = min(1.0, self.steps / max(1, c.eps_decay_steps))
        return c.eps_start + frac * (c.eps_end - c.eps_start)

    def act(self, obs: np.ndarray, greedy: bool = False) -> int:
        if not greedy and self.rng.random() < self.epsilon():
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.q.forward(obs)[0]))

    def observe(self, s, a, r, s2, done) -> float | None:
        """Store a transition and possibly run a gradient step. Returns loss if updated."""
        self.buffer.add(s, a, r, s2, done)
        self.steps += 1
        c = self.cfg
        if len(self.buffer) < max(c.warmup, c.batch_size) or self.steps % c.train_every:
            return None
        return self.learn()

    def targets(self, r: np.ndarray, s2: np.ndarray, d: np.ndarray) -> np.ndarray:
        q_next_t = self.target.forward(s2)
        if self.cfg.double:
            best = np.argmax(self.q.forward(s2), axis=1)
            nxt = q_next_t[np.arange(len(best)), best]
        else:
            nxt = q_next_t.max(axis=1)
        return r + self.cfg.gamma * (1.0 - d) * nxt

    def learn(self) -> float:
        c = self.cfg
        s, a, r, s2, d = self.buffer.sample(c.batch_size)
        y = self.targets(r, s2, d)
        q = self.q.forward(s)
        idx = np.arange(len(a))
        td = q[idx, a] - y
        w = (self.buffer.last_weights if c.prioritized else np.ones_like(td))
        # Huber loss gradient, weighted, averaged over the batch
        g_td = np.clip(td, -1.0, 1.0) * w / len(a)
        grad_out = np.zeros_like(q)
        grad_out[idx, a] = g_td
        self.opt.step(self.q.backward(grad_out))
        if c.prioritized:
            self.buffer.update_priorities(td)  # type: ignore[union-attr]
        self.updates += 1
        if c.tau is not None:
            self.target.soft_update(self.q, c.tau)
        elif self.updates % c.target_sync == 0:
            self.target.copy_from(self.q)
        absd = np.abs(td)
        return float(np.where(absd <= 1, 0.5 * td ** 2, absd - 0.5).mean())
