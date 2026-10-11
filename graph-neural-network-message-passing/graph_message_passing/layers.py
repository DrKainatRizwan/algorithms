"""Message-passing layers with hand-written backward passes.

Every layer exposes ``forward(x)``, ``backward(grad_out)`` (returning the
gradient w.r.t. its input) and dictionaries ``params`` / ``grads``.
"""
from __future__ import annotations

import numpy as np

from .graph import gcn_normalize, mean_normalize


def glorot(rng: np.random.Generator, fan_in: int, fan_out: int) -> np.ndarray:
    """Glorot/Xavier uniform initialisation."""
    lim = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-lim, lim, size=(fan_in, fan_out))


class Layer:
    """Base class holding parameters and gradients."""

    def __init__(self) -> None:
        self.params: dict[str, np.ndarray] = {}
        self.grads: dict[str, np.ndarray] = {}

    def forward(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def backward(self, grad: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class GCNLayer(Layer):
    """Graph convolution ``H = A_hat X W + b``.

    Passing ``identity=True`` replaces ``A_hat`` with ``I`` which yields a plain
    dense layer, used as a structure-blind MLP baseline.
    """

    def __init__(self, adj: np.ndarray, d_in: int, d_out: int,
                 rng: np.random.Generator, identity: bool = False) -> None:
        super().__init__()
        self.a_hat = np.eye(adj.shape[0]) if identity else gcn_normalize(adj)
        self.params = {"W": glorot(rng, d_in, d_out), "b": np.zeros(d_out)}
        self._ax: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._ax = self.a_hat @ x
        return self._ax @ self.params["W"] + self.params["b"]

    def backward(self, grad: np.ndarray) -> np.ndarray:
        self.grads = {"W": self._ax.T @ grad, "b": grad.sum(axis=0)}
        return self.a_hat.T @ (grad @ self.params["W"].T)


class SAGELayer(Layer):
    """GraphSAGE with mean aggregator: ``H = X W_s + mean_N(X) W_n + b``."""

    def __init__(self, adj: np.ndarray, d_in: int, d_out: int,
                 rng: np.random.Generator) -> None:
        super().__init__()
        self.p = mean_normalize(adj)
        self.params = {
            "Ws": glorot(rng, d_in, d_out),
            "Wn": glorot(rng, d_in, d_out),
            "b": np.zeros(d_out),
        }
        self._x: np.ndarray | None = None
        self._px: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        self._px = self.p @ x
        return x @ self.params["Ws"] + self._px @ self.params["Wn"] + self.params["b"]

    def backward(self, grad: np.ndarray) -> np.ndarray:
        self.grads = {
            "Ws": self._x.T @ grad,
            "Wn": self._px.T @ grad,
            "b": grad.sum(axis=0),
        }
        return grad @ self.params["Ws"].T + self.p.T @ (grad @ self.params["Wn"].T)


class _GATHead:
    """One attention head operating on a dense masked score matrix."""

    def __init__(self, mask: np.ndarray, d_in: int, d_out: int,
                 rng: np.random.Generator, slope: float) -> None:
        self.mask = mask
        self.slope = slope
        self.W = glorot(rng, d_in, d_out)
        self.a_src = glorot(rng, d_out, 1)[:, 0]
        self.a_dst = glorot(rng, d_out, 1)[:, 0]

    def forward(self, x: np.ndarray) -> np.ndarray:
        z = x @ self.W
        s = z @ self.a_src
        t = z @ self.a_dst
        pre = s[:, None] + t[None, :]
        e = np.where(pre > 0, pre, self.slope * pre)
        e = np.where(self.mask, e, -np.inf)
        e = e - e.max(axis=1, keepdims=True)
        w = np.exp(e)
        alpha = w / w.sum(axis=1, keepdims=True)
        self.cache = (x, z, pre, alpha)
        self.alpha = alpha
        return alpha @ z

    def backward(self, grad: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        x, z, pre, alpha = self.cache
        d_alpha = grad @ z.T
        dz = alpha.T @ grad
        de = alpha * (d_alpha - (alpha * d_alpha).sum(axis=1, keepdims=True))
        dpre = de * np.where(pre > 0, 1.0, self.slope) * self.mask
        ds = dpre.sum(axis=1)
        dt = dpre.sum(axis=0)
        da_src = z.T @ ds
        da_dst = z.T @ dt
        dz = dz + ds[:, None] * self.a_src[None, :] + dt[:, None] * self.a_dst[None, :]
        grads = {"W": x.T @ dz, "a_src": da_src, "a_dst": da_dst}
        return dz @ self.W.T, grads


class GATLayer(Layer):
    """Multi-head graph attention (Velickovic et al.), heads concatenated.

    Attention coefficients are a softmax over each node's neighbourhood
    (including itself) of ``LeakyReLU(a_s . z_i + a_d . z_j)``.
    """

    def __init__(self, adj: np.ndarray, d_in: int, d_out: int,
                 rng: np.random.Generator, heads: int = 2, slope: float = 0.2) -> None:
        super().__init__()
        if d_out % heads:
            raise ValueError("d_out must be divisible by the number of heads")
        mask = adj + np.eye(adj.shape[0]) > 0
        self.heads = [_GATHead(mask, d_in, d_out // heads, rng, slope) for _ in range(heads)]
        self.params = {}
        for k, h in enumerate(self.heads):
            self.params[f"W{k}"] = h.W
            self.params[f"as{k}"] = h.a_src
            self.params[f"ad{k}"] = h.a_dst
        self.params["b"] = np.zeros(d_out)

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = np.concatenate([h.forward(x) for h in self.heads], axis=1)
        return out + self.params["b"]

    def backward(self, grad: np.ndarray) -> np.ndarray:
        self.grads = {"b": grad.sum(axis=0)}
        width = grad.shape[1] // len(self.heads)
        dx = 0.0
        for k, h in enumerate(self.heads):
            dxk, g = h.backward(grad[:, k * width:(k + 1) * width])
            self.grads[f"W{k}"] = g["W"]
            self.grads[f"as{k}"] = g["a_src"]
            self.grads[f"ad{k}"] = g["a_dst"]
            dx = dx + dxk
        return dx

    def attention(self) -> list[np.ndarray]:
        """Attention matrices of the most recent forward pass, one per head."""
        return [h.alpha for h in self.heads]
