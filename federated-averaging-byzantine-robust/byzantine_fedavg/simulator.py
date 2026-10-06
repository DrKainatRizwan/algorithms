"""Federated training loop with a configurable aggregator and adversary."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .model import SoftmaxModel


@dataclass
class SimulationResult:
    """Per-round test accuracy / loss and the final global weights."""

    accuracy: list[float] = field(default_factory=list)
    loss: list[float] = field(default_factory=list)
    weights: np.ndarray | None = None

    @property
    def final_accuracy(self) -> float:
        return self.accuracy[-1]


class FederatedSimulator:
    """Synchronous FedAvg where the first ``n_byzantine`` clients are malicious.

    Clients send *model deltas* (local weights minus global weights); the
    server aggregates deltas and applies them to the global model.
    """

    def __init__(
        self,
        model: SoftmaxModel,
        X: np.ndarray,
        y: np.ndarray,
        shards: list[np.ndarray],
        X_test: np.ndarray,
        y_test: np.ndarray,
        aggregator: Callable[..., np.ndarray],
        n_byzantine: int = 0,
        attack: Callable[..., np.ndarray] | None = None,
        local_epochs: int = 1,
        lr: float = 0.1,
        batch_size: int = 32,
        seed: int = 0,
    ) -> None:
        if n_byzantine < 0 or n_byzantine >= len(shards):
            raise ValueError("n_byzantine must be in [0, n_clients)")
        if n_byzantine and attack is None:
            raise ValueError("an attack is required when n_byzantine > 0")
        self.model, self.X, self.y = model, X, y
        self.shards = shards
        self.X_test, self.y_test = X_test, y_test
        self.aggregator = aggregator
        self.n_byz = n_byzantine
        self.attack = attack
        self.local_epochs, self.lr, self.batch_size = local_epochs, lr, batch_size
        self.rng = np.random.default_rng(seed)

    def _honest_deltas(self, w: np.ndarray) -> np.ndarray:
        deltas = []
        for shard in self.shards[self.n_byz :]:
            local = self.model.local_train(
                w, self.X[shard], self.y[shard],
                self.local_epochs, self.lr, self.batch_size, self.rng,
            )
            deltas.append(local - w)
        return np.stack(deltas)

    def run(self, rounds: int) -> SimulationResult:
        """Train for ``rounds`` communication rounds."""
        w = self.model.init_params()
        res = SimulationResult()
        for _ in range(rounds):
            honest = self._honest_deltas(w)
            if self.n_byz:
                bad = self.attack(honest, self.n_byz, self.rng)
                updates = np.vstack([bad, honest])
            else:
                updates = honest
            w = w + self.aggregator(updates, f=self.n_byz)
            loss, _ = self.model.loss_and_grad(w, self.X_test, self.y_test)
            res.loss.append(loss)
            res.accuracy.append(self.model.accuracy(w, self.X_test, self.y_test))
        res.weights = w
        return res
