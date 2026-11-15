"""Causal broadcast with vector clocks (the transport for operation-based CRDTs).

A message from ``s`` stamped ``V`` is deliverable at a node with delivered
clock ``D`` iff ``V[s] == D[s] + 1`` and ``V[k] <= D[k]`` for all ``k != s``.
Otherwise it waits in a hold-back buffer.  Duplicates are discarded.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .clocks import VectorClock


@dataclass(frozen=True)
class Message:
    sender: str
    clock: VectorClock
    payload: Any


class CausalBroadcaster:
    """One node's endpoint of a causal-order broadcast group."""

    def __init__(self, node_id: str, on_deliver: Callable[[Message], None] | None = None) -> None:
        self.node_id = node_id
        self.delivered = VectorClock()
        self.buffer: list[Message] = []
        self.log: list[Message] = []
        self._on_deliver = on_deliver

    def broadcast(self, payload: Any) -> Message:
        """Create a message for a local event and deliver it locally."""
        self.delivered = self.delivered.incremented(self.node_id)
        msg = Message(self.node_id, self.delivered, payload)
        self.log.append(msg)
        if self._on_deliver:
            self._on_deliver(msg)
        return msg

    def deliverable(self, msg: Message) -> bool:
        s = msg.sender
        if msg.clock.get(s) != self.delivered.get(s) + 1:
            return False
        return all(msg.clock.get(k) <= self.delivered.get(k) for k in msg.clock.nodes() if k != s)

    def receive(self, msg: Message) -> list[Message]:
        """Accept a network message; return everything newly delivered."""
        if msg.sender == self.node_id or msg.clock.get(msg.sender) <= self.delivered.get(msg.sender):
            return []  # duplicate
        if all(not (m.sender == msg.sender and m.clock == msg.clock) for m in self.buffer):
            self.buffer.append(msg)
        out: list[Message] = []
        progress = True
        while progress:
            progress = False
            for m in list(self.buffer):
                if self.deliverable(m):
                    self.buffer.remove(m)
                    self.delivered = self.delivered.incremented(m.sender)
                    self.log.append(m)
                    out.append(m)
                    if self._on_deliver:
                        self._on_deliver(m)
                    progress = True
        return out
