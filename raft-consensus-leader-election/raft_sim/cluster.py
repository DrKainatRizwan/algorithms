"""Cluster harness: drives nodes, network and fault injection in lockstep."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from .invariants import InvariantChecker
from .network import Network
from .node import LEADER, RaftConfig, RaftNode


class Cluster:
    """A simulated Raft cluster advanced one millisecond per :meth:`step`."""

    def __init__(self, n: int = 5, seed: int = 0, config: Optional[RaftConfig] = None,
                 drop_rate: float = 0.0, min_delay: int = 2, max_delay: int = 15,
                 check_invariants: bool = True):
        self.config = config or RaftConfig()
        ids = list(range(n))
        self.nodes: Dict[int, RaftNode] = {i: RaftNode(i, ids, self.config, seed) for i in ids}
        self.net = Network(seed, min_delay, max_delay, drop_rate)
        self.now = 0
        self.down: set = set()
        self.checker = InvariantChecker() if check_invariants else None
        self.events: List[Tuple[int, str]] = []

    # ------------------------------------------------------------- simulation
    def step(self) -> None:
        self.now += 1
        for dst, msg in self.net.deliver_due(self.now):
            if dst not in self.down:
                self.nodes[dst].receive(msg, self.now)
        for i, node in self.nodes.items():
            if i not in self.down:
                before = node.role
                node.tick(self.now)
                if before != LEADER and node.role == LEADER:
                    self.events.append((self.now, f"node {i} elected for term {node.current_term}"))
        for i, node in self.nodes.items():
            for dst, msg in node.outbox:
                self.net.send(i, dst, msg, self.now)
            node.outbox.clear()
        if self.checker:
            self.checker.check(self)

    def run(self, ms: int) -> None:
        for _ in range(ms):
            self.step()

    def run_until(self, pred: Callable[["Cluster"], bool], max_ms: int = 5000) -> bool:
        for _ in range(max_ms):
            if pred(self):
                return True
            self.step()
        return pred(self)

    # ---------------------------------------------------------------- queries
    def leader(self) -> Optional[RaftNode]:
        """The live leader with the highest term, if any."""
        leaders = [n for i, n in self.nodes.items() if i not in self.down and n.role == LEADER]
        return max(leaders, key=lambda n: n.current_term) if leaders else None

    def stable_leader(self) -> Optional[RaftNode]:
        """A leader that a majority of live nodes currently follow."""
        ldr = self.leader()
        if ldr is None:
            return None
        votes = sum(1 for i, n in self.nodes.items()
                    if i not in self.down and n.current_term == ldr.current_term
                    and n.leader_id == ldr.id)
        return ldr if votes >= ldr.majority else None

    def wait_for_leader(self, max_ms: int = 5000) -> Optional[RaftNode]:
        self.run_until(lambda c: c.stable_leader() is not None, max_ms)
        return self.stable_leader()

    # ------------------------------------------------------------ interaction
    def submit(self, command: Any) -> Optional[Tuple[int, int]]:
        """Send a command to the current leader; return ``(leader, index)``."""
        ldr = self.leader()
        if ldr is None:
            return None
        idx = ldr.propose(command, self.now)
        return (ldr.id, idx) if idx is not None else None

    def wait_committed(self, index: int, nodes: Optional[List[int]] = None,
                       max_ms: int = 3000) -> bool:
        ids = nodes if nodes is not None else [i for i in self.nodes if i not in self.down]
        return self.run_until(lambda c: all(c.nodes[i].commit_index >= index for i in ids), max_ms)

    def crash(self, node_id: int) -> None:
        self.down.add(node_id)
        self.events.append((self.now, f"node {node_id} crashed"))

    def restart(self, node_id: int) -> None:
        self.down.discard(node_id)
        self.nodes[node_id].restart(self.now)
        self.events.append((self.now, f"node {node_id} restarted"))

    def partition(self, *groups: List[int]) -> None:
        self.net.partition(groups)
        self.events.append((self.now, f"partition {list(map(list, groups))}"))

    def heal(self) -> None:
        self.net.heal()
        self.events.append((self.now, "partition healed"))

    def converged(self) -> bool:
        """All live nodes hold identical logs and state machines."""
        live = [n for i, n in self.nodes.items() if i not in self.down]
        return all(n.log == live[0].log and n.state == live[0].state
                   and n.commit_index == live[0].commit_index for n in live)
