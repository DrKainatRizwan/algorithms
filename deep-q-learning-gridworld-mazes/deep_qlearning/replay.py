"""Experience replay buffers: uniform ring buffer and proportional prioritised replay."""
from __future__ import annotations

import numpy as np

Batch = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


class ReplayBuffer:
    """Fixed-capacity circular buffer of (s, a, r, s', done) transitions."""

    def __init__(self, capacity: int, obs_dim: int, seed: int = 0) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.s = np.zeros((capacity, obs_dim))
        self.a = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity)
        self.s2 = np.zeros((capacity, obs_dim))
        self.d = np.zeros(capacity)
        self.size = 0
        self.ptr = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def add(self, s: np.ndarray, a: int, r: float, s2: np.ndarray, done: bool) -> int:
        i = self.ptr
        self.s[i], self.a[i], self.r[i], self.s2[i], self.d[i] = s, a, r, s2, float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return i

    def _gather(self, idx: np.ndarray) -> Batch:
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx]

    def sample(self, batch_size: int) -> Batch:
        if self.size == 0:
            raise ValueError("buffer is empty")
        idx = self.rng.integers(0, self.size, size=batch_size)
        return self._gather(idx)


class PrioritizedReplayBuffer(ReplayBuffer):
    """Proportional prioritised replay (Schaul et al., 2016) with importance weights."""

    def __init__(self, capacity: int, obs_dim: int, alpha: float = 0.6,
                 beta: float = 0.4, eps: float = 1e-5, seed: int = 0) -> None:
        super().__init__(capacity, obs_dim, seed)
        self.alpha, self.beta, self.eps = alpha, beta, eps
        self.priorities = np.zeros(capacity)
        self.max_priority = 1.0
        self.last_idx: np.ndarray = np.zeros(0, dtype=np.int64)
        self.last_weights: np.ndarray = np.zeros(0)

    def add(self, s, a, r, s2, done) -> int:  # type: ignore[override]
        i = super().add(s, a, r, s2, done)
        self.priorities[i] = self.max_priority
        return i

    def sample(self, batch_size: int) -> Batch:
        if self.size == 0:
            raise ValueError("buffer is empty")
        p = self.priorities[: self.size] ** self.alpha
        p = p / p.sum()
        idx = self.rng.choice(self.size, size=batch_size, p=p)
        w = (self.size * p[idx]) ** (-self.beta)
        self.last_idx, self.last_weights = idx, w / w.max()
        return self._gather(idx)

    def update_priorities(self, td_errors: np.ndarray) -> None:
        """Set priorities of the most recently sampled batch from |TD error|."""
        pr = np.abs(td_errors) + self.eps
        self.priorities[self.last_idx] = pr
        self.max_priority = max(self.max_priority, float(pr.max()))
