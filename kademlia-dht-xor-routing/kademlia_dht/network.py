"""A deterministic in-memory network with RPC accounting and failure injection."""
from __future__ import annotations

import random
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .node import DHTNode


class Network:
    """Routes RPCs between :class:`DHTNode` objects.

    Offline nodes never answer (the caller observes a timeout, ``None``).
    Optionally a fraction of messages is dropped to model packet loss.
    """

    def __init__(self, seed: int = 0, loss: float = 0.0) -> None:
        self.nodes: dict[int, "DHTNode"] = {}
        self.online: set[int] = set()
        self.rng = random.Random(seed)
        self.loss = loss
        self.rpc_count = 0
        self.timeouts = 0

    def add(self, node: "DHTNode") -> None:
        self.nodes[node.id] = node
        self.online.add(node.id)

    def kill(self, nid: int) -> None:
        self.online.discard(nid)

    def revive(self, nid: int) -> None:
        if nid in self.nodes:
            self.online.add(nid)

    def alive_ids(self) -> list[int]:
        return sorted(self.online)

    def rpc(self, src: int, dst: int, method: str, *args: Any) -> Any:
        """Deliver an RPC; returns ``None`` on timeout."""
        self.rpc_count += 1
        if dst not in self.online or src not in self.online:
            self.timeouts += 1
            return None
        if self.loss and self.rng.random() < self.loss:
            self.timeouts += 1
            return None
        return getattr(self.nodes[dst], "handle_" + method)(src, *args)

    def true_closest(self, target: int, count: int) -> list[int]:
        """Ground truth: globally closest live nodes to ``target``."""
        return sorted(self.online, key=lambda n: n ^ target)[:count]
