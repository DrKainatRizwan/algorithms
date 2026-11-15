"""Gossip-based anti-entropy simulation and lattice-law checking."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Sequence

from .crdts import CRDT
from .network import Network


@dataclass
class RunResult:
    converged: bool
    rounds: int
    messages: int
    dropped: int


class Cluster:
    """A set of replicas that exchange full state over a lossy network.

    ``mode='push'``: each round every replica sends its state to ``fanout``
    random peers.  ``mode='pushpull'``: a receiver of a push also replies with
    its own state, roughly halving convergence time.
    """

    def __init__(self, factory: Callable[[str], CRDT], n: int, *, seed: int = 0,
                 fanout: int = 1, mode: str = "push", drop_prob: float = 0.0,
                 dup_prob: float = 0.0, max_delay: int = 3) -> None:
        if mode not in ("push", "pushpull"):
            raise ValueError(mode)
        self.ids = [f"n{i}" for i in range(n)]
        self.replicas = {i: factory(i) for i in self.ids}
        self.net = Network(seed=seed, drop_prob=drop_prob, dup_prob=dup_prob, max_delay=max_delay)
        self.rng = random.Random(seed + 1)
        self.fanout = fanout
        self.mode = mode

    def gossip_round(self) -> None:
        """Every replica pushes state to random peers, then time advances one tick."""
        for src in self.ids:
            peers = [p for p in self.ids if p != src]
            for dst in self.rng.sample(peers, min(self.fanout, len(peers))):
                self.net.send(src, dst, ("push", self.replicas[src].copy()))
        self.tick()

    def tick(self) -> None:
        for src, dst, (kind, state) in self.net.advance():
            self.replicas[dst].merge(state)
            if kind == "push" and self.mode == "pushpull":
                self.net.send(dst, src, ("reply", self.replicas[dst].copy()))

    def converged(self) -> bool:
        keys = [r.canonical() for r in self.replicas.values()]
        return all(k == keys[0] for k in keys[1:])

    def run_until_converged(self, max_rounds: int = 200) -> RunResult:
        for r in range(1, max_rounds + 1):
            if self.converged() and self.net.in_flight() == 0 and r > 1:
                return self._result(True, r - 1)
            self.gossip_round()
            if self.converged():
                return self._result(True, r)
        return self._result(self.converged(), max_rounds)

    def _result(self, ok: bool, rounds: int) -> RunResult:
        return RunResult(ok, rounds, self.net.stats.sent, self.net.stats.dropped)


def check_lattice_laws(samples: Sequence[CRDT]) -> dict[str, bool]:
    """Test commutativity, associativity and idempotence of ``merge`` on samples."""
    comm = assoc = idem = True
    for a in samples:
        idem &= a.joined(a) == a
        for b in samples:
            comm &= a.joined(b) == b.joined(a)
            for c in samples:
                assoc &= a.joined(b).joined(c) == a.joined(b.joined(c))
    return {"commutative": comm, "associative": assoc, "idempotent": idem}
