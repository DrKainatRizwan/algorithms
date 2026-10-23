"""Tiny neural-network toolkit with hand-written backpropagation."""

from __future__ import annotations

from typing import Callable

import numpy as np

LOG_2PI = float(np.log(2 * np.pi))


class MLP:
    """Fully-connected net with tanh hidden layers and a linear output."""

    def __init__(self, sizes: list[int], rng: np.random.Generator,
                 out_scale: float = 1.0) -> None:
        self.W: list[np.ndarray] = []
        self.b: list[np.ndarray] = []
        for i, (m, n) in enumerate(zip(sizes[:-1], sizes[1:])):
            last = i == len(sizes) - 2
            gain = out_scale if last else np.sqrt(2.0)
            # orthogonal initialisation
            a = rng.standard_normal((m, n))
            q, r = np.linalg.qr(a if m >= n else a.T)
            q = q * np.sign(np.diag(r))
            w = q if m >= n else q.T
            self.W.append(np.ascontiguousarray(gain * w))
            self.b.append(np.zeros(n))
        self._cache: list[np.ndarray] = []

    @property
    def params(self) -> list[np.ndarray]:
        return [p for wb in zip(self.W, self.b) for p in wb]

    def forward(self, x: np.ndarray, store: bool = False) -> np.ndarray:
        h = x
        acts = [x]
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            h = h @ w + b
            if i < len(self.W) - 1:
                h = np.tanh(h)
            acts.append(h)
        if store:
            self._cache = acts
        return h

    def backward(self, dout: np.ndarray) -> list[np.ndarray]:
        """Backprop ``dL/d(output)``; returns grads aligned with :attr:`params`."""
        grads: list[np.ndarray] = [None] * (2 * len(self.W))  # type: ignore[list-item]
        d = dout
        for i in reversed(range(len(self.W))):
            grads[2 * i] = self._cache[i].T @ d
            grads[2 * i + 1] = d.sum(axis=0)
            if i > 0:
                d = (d @ self.W[i].T) * (1.0 - self._cache[i] ** 2)
        return grads


class Adam:
    """Adam optimiser over a list of parameter arrays (updated in place)."""

    def __init__(self, params: list[np.ndarray], lr: float = 3e-4,
                 betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8) -> None:
        self.params, self.lr, self.betas, self.eps = params, lr, betas, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads: list[np.ndarray]) -> None:
        self.t += 1
        b1, b2 = self.betas
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m *= b1
            m += (1 - b1) * g
            v *= b2
            v += (1 - b2) * g * g
            p -= self.lr * (m / (1 - b1 ** self.t)) / (np.sqrt(v / (1 - b2 ** self.t)) + self.eps)


def clip_grad_norm(grads: list[np.ndarray], max_norm: float) -> float:
    """Scale grads in place so their global L2 norm is at most ``max_norm``."""
    norm = float(np.sqrt(sum(float((g ** 2).sum()) for g in grads)))
    if norm > max_norm:
        scale = max_norm / (norm + 1e-12)
        for g in grads:
            g *= scale
    return norm


class GaussianPolicy:
    """Diagonal Gaussian policy: MLP mean and a state-independent log-std."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: tuple[int, ...] = (64, 64),
                 rng: np.random.Generator | None = None, init_log_std: float = -0.5) -> None:
        rng = rng or np.random.default_rng(0)
        self.net = MLP([obs_dim, *hidden, act_dim], rng, out_scale=0.01)
        self.log_std = np.full(act_dim, init_log_std)
        self.act_dim = act_dim

    @property
    def params(self) -> list[np.ndarray]:
        return [*self.net.params, self.log_std]

    def mean(self, obs: np.ndarray, store: bool = False) -> np.ndarray:
        return self.net.forward(obs, store=store)

    def log_prob(self, obs: np.ndarray, act: np.ndarray) -> np.ndarray:
        mu = self.mean(obs)
        z = (act - mu) / np.exp(self.log_std)
        return (-0.5 * z ** 2 - self.log_std - 0.5 * LOG_2PI).sum(axis=-1)

    def entropy(self) -> float:
        return float((self.log_std + 0.5 * (LOG_2PI + 1.0)).sum())

    def sample(self, obs: np.ndarray, rng: np.random.Generator,
               deterministic: bool = False) -> tuple[np.ndarray, np.ndarray]:
        mu = self.mean(obs)
        act = mu if deterministic else mu + np.exp(self.log_std) * rng.standard_normal(mu.shape)
        return act, self.log_prob(obs, act)

    def ppo_loss_grads(self, obs: np.ndarray, act: np.ndarray, old_logp: np.ndarray,
                       adv: np.ndarray, clip: float, ent_coef: float):
        """Clipped surrogate loss with entropy bonus and analytic gradients.

        Returns ``(loss, grads, approx_kl, clip_fraction)``.
        """
        n = obs.shape[0]
        mu = self.mean(obs, store=True)
        std = np.exp(self.log_std)
        z = (act - mu) / std
        logp = (-0.5 * z ** 2 - self.log_std - 0.5 * LOG_2PI).sum(axis=-1)
        ratio = np.exp(logp - old_logp)
        unclipped = ratio * adv
        clipped = np.clip(ratio, 1 - clip, 1 + clip) * adv
        loss = -np.minimum(unclipped, clipped).mean() - ent_coef * self.entropy()
        # gradient flows through the unclipped term only when it is the minimum
        active = (unclipped <= clipped).astype(float)
        dlogp = -(active * adv * ratio) / n                      # dL/dlogp, shape (n,)
        dmu = dlogp[:, None] * (z / std)                         # dlogp/dmu = z/std
        dls = (dlogp[:, None] * (z ** 2 - 1.0)).sum(axis=0) - ent_coef
        grads = [*self.net.backward(dmu), dls]
        approx_kl = float(((ratio - 1) - np.log(ratio)).mean())
        clip_frac = float((np.abs(ratio - 1) > clip).mean())
        return float(loss), grads, approx_kl, clip_frac


class ValueNet:
    """State-value function ``V(s)`` trained with squared error."""

    def __init__(self, obs_dim: int, hidden: tuple[int, ...] = (64, 64),
                 rng: np.random.Generator | None = None) -> None:
        rng = rng or np.random.default_rng(1)
        self.net = MLP([obs_dim, *hidden, 1], rng, out_scale=1.0)

    @property
    def params(self) -> list[np.ndarray]:
        return self.net.params

    def predict(self, obs: np.ndarray) -> np.ndarray:
        return self.net.forward(obs)[:, 0]

    def loss_grads(self, obs: np.ndarray, returns: np.ndarray, old_values: np.ndarray | None = None,
                   clip: float | None = None):
        """Mean-squared error (optionally PPO value-clipped) and its gradients."""
        n = obs.shape[0]
        v = self.net.forward(obs, store=True)[:, 0]
        err = v - returns
        loss = 0.5 * (err ** 2)
        d = err
        if clip is not None and old_values is not None:
            vc = old_values + np.clip(v - old_values, -clip, clip)
            loss_c = 0.5 * (vc - returns) ** 2
            use_c = loss_c > loss
            loss = np.maximum(loss, loss_c)
            # clipped branch has zero gradient wrt v once the clip is saturated
            sat = np.abs(v - old_values) > clip
            d = np.where(use_c & sat, 0.0, np.where(use_c, vc - returns, err))
        grads = self.net.backward((d / n)[:, None])
        return float(loss.mean()), grads


def gradient_check(f_and_grad: Callable[[], tuple[float, list[np.ndarray]]],
                   params: list[np.ndarray], eps: float = 1e-6, samples: int = 20,
                   seed: int = 0) -> float:
    """Max relative error between analytic and central finite-difference grads."""
    rng = np.random.default_rng(seed)
    _, grads = f_and_grad()
    worst = 0.0
    for p, g in zip(params, grads):
        flat = p.reshape(-1)
        gflat = g.reshape(-1)
        for idx in rng.choice(flat.size, size=min(samples, flat.size), replace=False):
            old = flat[idx]
            flat[idx] = old + eps
            lp, _ = f_and_grad()
            flat[idx] = old - eps
            lm, _ = f_and_grad()
            flat[idx] = old
            num = (lp - lm) / (2 * eps)
            denom = max(1e-8, abs(num) + abs(gflat[idx]))
            worst = max(worst, abs(num - gflat[idx]) / denom)
    return worst
