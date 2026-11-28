"""Kademlia node: RPC handlers, iterative lookup, store/get, join and refresh."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .ids import hash_to_id, random_id_in_bucket
from .network import Network
from .routing import RoutingTable


@dataclass
class LookupResult:
    """Outcome of an iterative lookup."""

    closest: list[int]
    value: Any = None
    found: bool = False
    rounds: int = 0
    rpcs: int = 0
    queried: list[int] = field(default_factory=list)


class DHTNode:
    """A single Kademlia peer."""

    def __init__(self, node_id: int, network: Network, bits: int = 32,
                 k: int = 8, alpha: int = 3) -> None:
        self.id = node_id
        self.net = network
        self.bits = bits
        self.k = k
        self.alpha = alpha
        self.table = RoutingTable(node_id, bits, k)
        self.storage: dict[int, Any] = {}
        network.add(self)

    # ---- contact bookkeeping -------------------------------------------------
    def _touch(self, nid: int) -> None:
        """Note a live contact; if its bucket is full, evict only if the oldest is dead."""
        stale = self.table.update(nid)
        if stale is not None:
            if self.net.rpc(self.id, stale, "ping") is None:
                self.table.remove(stale)
                self.table.update(nid)
            else:
                self.table.update(stale)  # still alive -> move to tail

    # ---- RPC handlers (receiver side) ---------------------------------------
    def handle_ping(self, sender: int) -> bool:
        self._touch(sender)
        return True

    def handle_find_node(self, sender: int, target: int) -> list[int]:
        self._touch(sender)
        return self.table.closest(target, self.k, exclude=sender)

    def handle_find_value(self, sender: int, key: int) -> tuple[Any, list[int]]:
        self._touch(sender)
        if key in self.storage:
            return self.storage[key], []
        return None, self.table.closest(key, self.k, exclude=sender)

    def handle_store(self, sender: int, key: int, value: Any) -> bool:
        self._touch(sender)
        self.storage[key] = value
        return True

    # ---- iterative lookup ---------------------------------------------------
    def lookup(self, target: int, find_value: bool = False) -> LookupResult:
        """Iterative node/value lookup with parallelism ``alpha``.

        Each round queries the ``alpha`` closest not-yet-queried candidates.
        The lookup ends when the ``k`` closest known candidates have all
        replied (or a value is found).
        """
        start_rpcs = self.net.rpc_count
        dist = lambda n: n ^ target  # noqa: E731
        shortlist: set[int] = set(self.table.closest(target, self.k))
        queried: set[int] = set()
        failed: set[int] = set()
        order: list[int] = []
        rounds = 0
        value, found = None, False
        while True:
            live = sorted((n for n in shortlist if n not in failed), key=dist)
            top = live[: self.k]
            pending = [n for n in top if n not in queried][: self.alpha]
            if not pending:
                break
            rounds += 1
            for nid in pending:
                queried.add(nid)
                order.append(nid)
                if find_value:
                    resp = self.net.rpc(self.id, nid, "find_value", target)
                else:
                    resp = self.net.rpc(self.id, nid, "find_node", target)
                if resp is None:
                    failed.add(nid)
                    self.table.remove(nid)
                    continue
                self._touch(nid)
                if find_value:
                    val, closer = resp
                    if val is not None:
                        value, found = val, True
                        break
                else:
                    closer = resp
                for c in closer:
                    if c != self.id:
                        shortlist.add(c)
            if found:
                break
        live = sorted((n for n in shortlist if n not in failed), key=dist)[: self.k]
        return LookupResult(live, value, found, rounds,
                            self.net.rpc_count - start_rpcs, order)

    # ---- public API ----------------------------------------------------------
    def put(self, key: str | int, value: Any) -> int:
        """Store ``value`` at the ``k`` nodes closest to the key. Returns replicas written."""
        kid = key if isinstance(key, int) else hash_to_id(key, self.bits)
        res = self.lookup(kid)
        written = 0
        for nid in res.closest:
            if self.net.rpc(self.id, nid, "store", kid, value):
                written += 1
        if self.id ^ kid < (res.closest[-1] ^ kid if res.closest else 1 << self.bits) \
                or len(res.closest) < self.k:
            self.storage[kid] = value
            written += 1
        return written

    def get(self, key: str | int) -> LookupResult:
        """Retrieve a value; checks the local store first."""
        kid = key if isinstance(key, int) else hash_to_id(key, self.bits)
        if kid in self.storage:
            return LookupResult([self.id], self.storage[kid], True)
        return self.lookup(kid, find_value=True)

    def bootstrap(self, contact: int) -> None:
        """Join the network through a known ``contact``."""
        self._touch(contact)
        self.lookup(self.id)
        # Refresh every bucket farther away than the closest populated one.
        populated = [i for i, b in enumerate(self.table.buckets) if b.entries]
        if populated:
            for i in range(populated[0] + 1, self.bits):
                self.refresh_bucket(i)

    def refresh_bucket(self, index: int) -> None:
        rng = self.net.rng
        self.lookup(random_id_in_bucket(self.id, index, rng))

    def republish(self) -> int:
        """Re-store every held key at its current k closest nodes."""
        n = 0
        for kid, val in list(self.storage.items()):
            res = self.lookup(kid)
            for nid in res.closest:
                if self.net.rpc(self.id, nid, "store", kid, val):
                    n += 1
        return n
